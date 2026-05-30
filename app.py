from __future__ import annotations

import re
from io import BytesIO

import streamlit as st

from src.config import get_settings
from src.document_loader import SUPPORTED_EXTENSIONS, load_file_bytes
from src.graph_rag import KnowledgeGraph, build_knowledge_graph
from src.health import build_health_report
from src.models import Chunk, RawDocument
from src.rag import ANSWER_MODES, answer_question, compare_documents
from src.splitter import split_documents
from src.suggestions import suggest_questions
from src.vector_store import SearchIndex


st.set_page_config(page_title="TrustDoc AI", page_icon="TD", layout="wide")


def main() -> None:
    settings = get_settings()
    _inject_css()

    st.title("TrustDoc AI")
    st.caption("A trust-aware RAG chatbot for real-world documents, with citations, confidence signals, gaps, and conflict checks.")

    with st.sidebar:
        st.header("Knowledge Base")
        available_embedding_providers = ["Local TF-IDF"]
        if settings.google_api_key:
            available_embedding_providers.append("Google Gemini")
        if settings.openai_api_key:
            available_embedding_providers.append("OpenAI")
        default_embedding_provider = "Google Gemini" if settings.google_api_key else "Local TF-IDF"
        embedding_provider = st.selectbox(
            "Embedding provider",
            available_embedding_providers,
            index=available_embedding_providers.index(default_embedding_provider),
            help="Choose Local TF-IDF for no API calls, Gemini for Google semantic search, or OpenAI for OpenAI semantic search.",
        )
        answer_providers = ["Local extractive"]
        if settings.google_api_key:
            answer_providers.append("Gemini")
        if settings.openai_api_key:
            answer_providers.append("OpenAI")
        default_answer_provider = "Gemini" if settings.google_api_key else "Local extractive"
        answer_provider = st.selectbox(
            "Answer provider",
            answer_providers,
            index=answer_providers.index(default_answer_provider),
            help="Gemini/OpenAI synthesize direct answers from retrieved evidence. Local extractive quotes the strongest evidence.",
        )
        conflict_modes = ["Fast heuristic"]
        if settings.google_api_key:
            conflict_modes.append("Gemini reasoning")
        if settings.openai_api_key:
            conflict_modes.append("OpenAI reasoning")
        default_conflict_mode = "Gemini reasoning" if settings.google_api_key else "Fast heuristic"
        conflict_mode = st.selectbox(
            "Conflict detection",
            conflict_modes,
            index=conflict_modes.index(default_conflict_mode),
            help="Use an LLM to judge whether retrieved evidence truly contradicts, or use the fast local heuristic.",
        )
        retrieval_mode = st.selectbox(
            "Retrieval mode",
            ["Vector RAG", "Graph-Enhanced RAG"],
            help="Graph-Enhanced RAG expands vector results with chunks connected by shared entities and topics.",
        )
        rewrite_queries = st.toggle(
            "Rewrite vague questions",
            value=True,
            help="Clarifies vague questions before retrieval while preserving the original question for the final answer.",
        )
        mode = st.selectbox("Answer mode", list(ANSWER_MODES.keys()))
        st.divider()
        uploaded_files = st.file_uploader(
            "Upload documents",
            type=[ext.replace(".", "") for ext in SUPPORTED_EXTENSIONS],
            accept_multiple_files=True,
        )
        _cache_uploaded_files(uploaded_files)
        cached_uploads = st.session_state.get("uploaded_file_cache", [])
        if cached_uploads:
            st.caption("Ready to index: " + ", ".join(file["name"] for file in cached_uploads))
        _render_index_freshness_warning()
        process_clicked = st.button("Build knowledge base", type="primary", use_container_width=True)
        clear_clicked = st.button("Clear session", use_container_width=True)

    if clear_clicked:
        st.session_state.clear()
        st.rerun()

    if process_clicked:
        documents = _load_documents()
        if not documents:
            st.warning("Upload at least one supported document before building the knowledge base.")
        else:
            with st.spinner("Extracting text, chunking documents, and building the retriever..."):
                chunks = split_documents(
                    documents,
                    chunk_size=settings.default_chunk_size,
                    chunk_overlap=settings.default_chunk_overlap,
                )
                if not chunks:
                    st.error("No readable text was found. If this is a scanned PDF, OCR is needed before upload.")
                    return
                index = SearchIndex.build(chunks, settings, embedding_provider=embedding_provider)
                st.session_state["documents"] = documents
                st.session_state["chunks"] = chunks
                st.session_state["index"] = index
                st.session_state["graph"] = build_knowledge_graph(chunks)
                st.session_state["health"] = build_health_report(documents, chunks)
                st.session_state["suggestions"] = suggest_questions(chunks, settings)
                if index.fallback_reason:
                    st.warning(
                        f"{embedding_provider} embeddings failed, so the app fell back to Local TF-IDF. "
                        f"Reason: {index.fallback_reason}"
                    )
                else:
                    st.success(f"Knowledge base ready using {index.backend}.")

    index: SearchIndex | None = st.session_state.get("index")
    chunks: list[Chunk] = st.session_state.get("chunks", [])
    graph: KnowledgeGraph | None = st.session_state.get("graph")

    left, right = st.columns([0.64, 0.36], gap="large")
    with left:
        st.subheader("Ask Documents")
        if not index:
            st.info("Upload documents from the sidebar and build the knowledge base to start.")
        if "pending_question" in st.session_state:
            st.session_state["question_input"] = st.session_state.pop("pending_question")
        query = st.text_input(
            "Question",
            key="question_input",
            placeholder="Example: What are the refund deadlines and exceptions?",
        )
        col_a, col_b = st.columns([0.5, 0.5])
        ask_clicked = col_a.button("Ask", type="primary", use_container_width=True, disabled=not index)
        compare_clicked = col_b.button("Compare documents", use_container_width=True, disabled=not index)
        submitted_question = st.session_state.get("question_input", "").strip()

        if (ask_clicked or compare_clicked) and index and not submitted_question:
            st.warning("Enter a question before asking the documents.")
        elif ask_clicked and index:
            st.session_state["last_response"] = answer_question(
                submitted_question,
                index,
                settings,
                mode,
                conflict_mode=conflict_mode,
                retrieval_mode=retrieval_mode,
                graph=graph,
                answer_provider=answer_provider,
                rewrite_queries=rewrite_queries,
            )
        elif compare_clicked and index:
            st.session_state["last_response"] = compare_documents(
                submitted_question,
                index,
                settings,
                conflict_mode=conflict_mode,
                retrieval_mode=retrieval_mode,
                graph=graph,
                answer_provider=answer_provider,
                rewrite_queries=rewrite_queries,
            )

        if "last_response" in st.session_state:
            _render_response(st.session_state["last_response"])

    with right:
        _render_health_panel()
        _render_graph_panel()
        _render_suggestions()
        _render_chunk_preview(chunks)


def _cache_uploaded_files(uploaded_files: list[object] | None) -> None:
    if uploaded_files is None:
        return
    st.session_state["uploaded_file_cache"] = [
        {"name": uploaded_file.name, "data": uploaded_file.getvalue()}
        for uploaded_file in uploaded_files
    ]


def _load_documents() -> list[RawDocument]:
    documents: list[RawDocument] = []
    for uploaded_file in st.session_state.get("uploaded_file_cache", []):
        documents.extend(load_file_bytes(uploaded_file["name"], BytesIO(uploaded_file["data"])))
    return documents


def _render_index_freshness_warning() -> None:
    uploaded_files = st.session_state.get("uploaded_file_cache", [])
    if not uploaded_files or "chunks" not in st.session_state:
        return

    indexed_sources = {chunk.source for chunk in st.session_state.get("chunks", [])}
    uploaded_names = {uploaded_file["name"] for uploaded_file in uploaded_files}
    missing_sources = sorted(uploaded_names - indexed_sources)
    if missing_sources:
        st.warning(
            "Uploaded file not indexed yet. Click Build knowledge base again to include: "
            + ", ".join(missing_sources)
        )


def _render_response(response) -> None:
    if response.prompt_injection_warnings:
        _render_prompt_injection_alert(response.prompt_injection_warnings)

    st.markdown("### Answer")
    trust = response.trust
    score_col, label_col = st.columns([0.25, 0.75])
    score_col.metric("Trust score", f"{trust.score}%")
    label_col.info(f"{trust.label}: {' '.join(trust.reasons)}")
    st.markdown(response.answer)
    if response.rewritten_query:
        st.caption(f"Retrieval query used: {response.rewritten_query}")

    if trust.gaps:
        with st.expander("Knowledge gaps", expanded=True):
            for gap in trust.gaps:
                st.write(f"- {gap}")

    if trust.conflict_warnings:
        with st.expander("Possible conflicts", expanded=True):
            for warning in trust.conflict_warnings:
                st.warning(warning)

    if response.retrieved:
        with st.expander("Source citations", expanded=True):
            for number, item in enumerate(response.retrieved, start=1):
                _render_citation_card(number, item)


def _render_prompt_injection_alert(warnings) -> None:
    st.error(
        f"Prompt injection detected: {len(warnings)} suspicious instruction-like "
        "item(s) were blocked before answer generation."
    )
    with st.expander("Blocked prompt-injection details", expanded=True):
        st.warning(
            "The app treated this text as untrusted document/user content, removed flagged document "
            "sentences from the generation context, and answered only from remaining evidence."
        )
        for warning in warnings:
            st.markdown(f"**{warning.citation}**")
            st.caption(f"{warning.detector} detector | score {warning.score:.2f}")
            st.write(warning.reason)
            st.code(_shorten(warning.text, 320), language="text")


def _render_citation_card(number: int, item) -> None:
    source = item.chunk.source
    location = f"Page {item.chunk.page}" if item.chunk.page is not None else f"Chunk {item.chunk.chunk_index + 1}"
    excerpt = _supporting_excerpt(item.chunk.text)
    with st.container(border=True):
        top_left, top_right = st.columns([0.76, 0.24])
        top_left.markdown(f"**[{number}] {source}**")
        top_right.metric("Relevance", f"{item.score:.2f}")
        st.caption(location)
        st.markdown("**Supporting excerpt**")
        st.info(excerpt)
        with st.expander("View full retrieved text"):
            st.write(_clean_chunk_text(item.chunk.text))


def _render_health_panel() -> None:
    st.subheader("Document Health")
    health = st.session_state.get("health")
    if not health:
        st.caption("No report yet.")
        return
    c1, c2, c3 = st.columns(3)
    c1.metric("Docs", health.document_count)
    c2.metric("Chunks", health.chunk_count)
    c3.metric("Words", health.total_words)
    st.write("Top topics: " + (", ".join(health.top_terms) if health.top_terms else "None found"))
    if health.duplicate_ratio > 0.15:
        st.warning(f"Duplicate content ratio is {health.duplicate_ratio:.0%}.")
    if health.short_document_count:
        st.warning(f"{health.short_document_count} document pages or files have very little text.")
    if health.scanned_or_low_text_sources:
        st.warning("Some PDF pages may be scanned or low text: " + ", ".join(health.scanned_or_low_text_sources[:4]))


def _render_graph_panel() -> None:
    graph = st.session_state.get("graph")
    if not graph:
        return
    st.subheader("Knowledge Graph")
    c1, c2 = st.columns(2)
    c1.metric("Entities", graph.entity_count)
    c2.metric("Edges", graph.edge_count)


def _render_suggestions() -> None:
    st.subheader("Suggested Questions")
    suggestions = st.session_state.get("suggestions", [])
    if not suggestions:
        st.caption("Suggestions appear after indexing.")
        return
    for index, suggestion in enumerate(suggestions):
        if st.button(suggestion, key=f"suggestion_{index}", use_container_width=True):
            st.session_state["pending_question"] = suggestion
            st.rerun()


def _render_chunk_preview(chunks: list[Chunk]) -> None:
    st.subheader("Indexed Evidence")
    if not chunks:
        st.caption("No chunks indexed yet.")
        return
    with st.expander("Preview chunks"):
        for chunk in chunks[:5]:
            st.markdown(f"**{chunk.citation}**")
            st.write(chunk.text[:450] + ("..." if len(chunk.text) > 450 else ""))


def _inject_css() -> None:
    st.markdown(
        """
        <style>
        .stApp { background: #f7f8fb; }
        [data-testid="stSidebar"] { background: #ffffff; }
        div[data-testid="stMetric"] {
            background: #ffffff;
            border: 1px solid #e5e7eb;
            padding: 0.75rem;
            border-radius: 8px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _supporting_excerpt(text: str, limit: int = 420) -> str:
    sentences = _readable_sentences(text)
    if not sentences:
        return _shorten(_clean_chunk_text(text), limit)
    excerpt = ""
    for sentence in sentences:
        candidate = f"{excerpt} {sentence}".strip()
        if len(candidate) > limit and excerpt:
            break
        excerpt = candidate
    return _shorten(excerpt, limit)


def _readable_sentences(text: str) -> list[str]:
    sentences: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip().strip("#").strip()
        if not line or _looks_like_heading(line):
            continue
        parts = re.split(r"(?<=[.!?])\s+", " ".join(line.split()))
        sentences.extend(part for part in parts if len(part.split()) >= 4)
    return sentences


def _clean_chunk_text(text: str) -> str:
    return "\n".join(line.strip() for line in text.splitlines() if line.strip())


def _shorten(text: str, limit: int) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3].rstrip() + "..."


def _looks_like_heading(text: str) -> bool:
    words = text.split()
    if not words or len(words) > 9:
        return False
    if text.endswith((".", "?", "!")):
        return False
    title_words = sum(1 for word in words if word[:1].isupper())
    return text.isupper() or title_words >= max(1, len(words) // 2)


if __name__ == "__main__":
    main()
