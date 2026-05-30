"""Answer generation for TrustDoc AI."""

from __future__ import annotations

import re

from src.answer_highlighter import highlight_key_facts
from src.config import Settings
from src.conflicts import analyze_conflicts
from src.context_compressor import compress_retrieved_context
from src.graph_rag import KnowledgeGraph, graph_enhance_retrieval
from src.models import RAGResponse, RetrievedChunk, TrustReport
from src.prompt_injection import (
    UNTRUSTED_EVIDENCE_INSTRUCTION,
    apply_prompt_injection_guard,
    detect_prompt_injection_in_question,
    should_refuse_user_prompt,
)
from src.query_rewriter import rewrite_query
from src.reranker import rerank_and_select
from src.trust import evaluate_trust
from src.vector_store import SearchIndex


RETRIEVAL_POOL_MULTIPLIER = 4


ANSWER_MODES = {
    "Simple Answer": "Answer in 3 to 5 concise sentences.",
    "Detailed Explanation": "Explain the answer carefully with relevant details.",
    "Bullet Summary": "Answer as short bullet points.",
    "Action Items": "Extract concrete next steps or required actions.",
    "Email Draft": "Write a professional email draft using only the evidence.",
    "Study Notes": "Turn the answer into compact study notes.",
    "Interview Prep": "Create interview-style talking points and likely follow-up questions.",
}


def answer_question(
    question: str,
    index: SearchIndex,
    settings: Settings,
    mode: str,
    use_openai_chat: bool = False,
    conflict_mode: str = "Fast heuristic",
    retrieval_mode: str = "Vector RAG",
    graph: KnowledgeGraph | None = None,
    answer_provider: str = "Local extractive",
    rewrite_queries: bool = True,
) -> RAGResponse:
    user_prompt_warnings = detect_prompt_injection_in_question(question, settings)
    if should_refuse_user_prompt(question, user_prompt_warnings):
        return _prompt_injection_refusal(user_prompt_warnings, mode)

    retrieval_query = rewrite_query(question, index.chunks, settings, enabled=rewrite_queries)
    candidate_limit = _candidate_limit(index, settings.max_context_chunks)
    retrieved = index.search(retrieval_query, top_k=candidate_limit)
    if retrieval_mode == "Graph-Enhanced RAG":
        retrieved = graph_enhance_retrieval(
            retrieval_query,
            retrieved,
            graph,
            top_k=candidate_limit,
        )
    retrieved = _select_final_evidence(retrieval_query, retrieved, index, settings.max_context_chunks)
    overview_question = _is_overview_question(question)
    if overview_question and not retrieved:
        retrieved = index.overview(top_k=settings.max_context_chunks)

    guarded_retrieved, prompt_injection_warnings = apply_prompt_injection_guard(
        question,
        retrieved,
        settings,
        user_warnings=user_prompt_warnings,
    )
    api_retrieved = compress_retrieved_context(question, guarded_retrieved, mode)
    if overview_question:
        answer = _answer_overview(guarded_retrieved)
    elif answer_provider == "Gemini" and settings.google_api_key:
        try:
            answer = _answer_with_gemini(question, api_retrieved, settings, mode)
        except Exception as exc:
            answer = _answer_locally(question, guarded_retrieved, mode)
            answer += f"\n\nNote: Gemini generation failed, so a local extractive answer was used. Error: {exc}"
    elif (answer_provider == "OpenAI" or use_openai_chat) and settings.openai_api_key:
        try:
            answer = _answer_with_openai(question, api_retrieved, settings, mode)
        except Exception as exc:
            answer = _answer_locally(question, guarded_retrieved, mode)
            answer += f"\n\nNote: OpenAI generation failed, so a local extractive answer was used. Error: {exc}"
    else:
        answer = _answer_locally(question, guarded_retrieved, mode)

    conflict_warnings = analyze_conflicts(question, guarded_retrieved, settings, conflict_mode)
    trust = evaluate_trust(
        question,
        guarded_retrieved,
        answer,
        conflict_warnings=conflict_warnings,
        prompt_injection_warnings=prompt_injection_warnings,
    )
    if trust.should_refuse:
        answer = (
            "I could not find enough support in the uploaded documents to answer this reliably. "
            "Try uploading a more relevant document or asking a narrower question."
        )
    else:
        answer = highlight_key_facts(answer)

    rewritten_query = retrieval_query if retrieval_query != question else None
    return RAGResponse(
        answer=answer,
        retrieved=guarded_retrieved,
        trust=trust,
        mode=mode,
        rewritten_query=rewritten_query,
        prompt_injection_warnings=prompt_injection_warnings,
    )


def _candidate_limit(index: SearchIndex, max_context_chunks: int) -> int:
    return min(len(index.chunks), max(12, max_context_chunks * RETRIEVAL_POOL_MULTIPLIER))


def _prompt_injection_refusal(warnings, mode: str) -> RAGResponse:
    return RAGResponse(
        answer=(
            "I cannot follow that instruction because it looks like a prompt-injection attempt. "
            "Ask a question about the uploaded documents, and I will answer with citations from the evidence."
        ),
        retrieved=[],
        trust=TrustReport(
            score=0,
            label="Prompt injection blocked",
            reasons=["The user prompt was an instruction-only request to change assistant behavior."],
            gaps=["Ask a factual question about the uploaded documents."],
            conflict_warnings=[],
            should_refuse=True,
        ),
        mode=mode,
        prompt_injection_warnings=warnings,
    )


def _select_final_evidence(
    retrieval_query: str,
    retrieved: list[RetrievedChunk],
    index: SearchIndex,
    max_context_chunks: int,
) -> list[RetrievedChunk]:
    if not retrieved:
        return []

    candidates = _add_adjacent_context(retrieved, index)
    max_context_chars = max(2400, max_context_chunks * 950)
    return rerank_and_select(
        retrieval_query,
        candidates,
        max_chunks=max_context_chunks,
        max_context_chars=max_context_chars,
    )


def _add_adjacent_context(retrieved: list[RetrievedChunk], index: SearchIndex) -> list[RetrievedChunk]:
    """Add nearby chunks as candidates, but let the reranker decide if they survive."""
    chunks_by_location = {(chunk.source, chunk.chunk_index): chunk for chunk in index.chunks}
    candidates_by_id = {item.chunk.id: item for item in retrieved}

    for item in retrieved[:3]:
        for offset in (-1, 1):
            neighbor = chunks_by_location.get((item.chunk.source, item.chunk.chunk_index + offset))
            if neighbor and neighbor.id not in candidates_by_id:
                candidates_by_id[neighbor.id] = RetrievedChunk(chunk=neighbor, score=item.score * 0.72)

    return list(candidates_by_id.values())


def compare_documents(
    question: str,
    index: SearchIndex,
    settings: Settings,
    use_openai_chat: bool = False,
    conflict_mode: str = "Fast heuristic",
    retrieval_mode: str = "Vector RAG",
    graph: KnowledgeGraph | None = None,
    answer_provider: str = "Local extractive",
    rewrite_queries: bool = True,
) -> RAGResponse:
    compare_question = f"Compare the documents and identify similarities, differences, conflicts, and gaps: {question}"
    return answer_question(
        compare_question,
        index=index,
        settings=settings,
        mode="Detailed Explanation",
        use_openai_chat=use_openai_chat,
        conflict_mode=conflict_mode,
        retrieval_mode=retrieval_mode,
        graph=graph,
        answer_provider=answer_provider,
        rewrite_queries=rewrite_queries,
    )


def _answer_with_openai(question: str, retrieved: list[RetrievedChunk], settings: Settings, mode: str) -> str:
    from openai import OpenAI

    client = OpenAI(api_key=settings.openai_api_key)
    context = _format_context(retrieved)
    instruction = ANSWER_MODES.get(mode, ANSWER_MODES["Simple Answer"])
    response = client.chat.completions.create(
        model=settings.openai_chat_model,
        temperature=0.1,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are TrustDoc AI, a careful retrieval-augmented assistant. "
                    f"{UNTRUSTED_EVIDENCE_INSTRUCTION} "
                    "Start with a direct answer to the user's question in the first sentence. "
                    "Use only the provided document evidence. If the evidence is insufficient, say what is missing. "
                    "Bold the most important factual values, such as durations, deadlines, amounts, dates, and percentages. "
                    "Cite sources inline using bracketed citation numbers like [1]."
                ),
            },
            {"role": "user", "content": f"Question: {question}\n\nMode: {instruction}\n\nEvidence:\n{context}"},
        ],
    )
    return response.choices[0].message.content or ""


def _answer_with_gemini(question: str, retrieved: list[RetrievedChunk], settings: Settings, mode: str) -> str:
    from google import genai

    client = genai.Client(api_key=settings.google_api_key)
    instruction = ANSWER_MODES.get(mode, ANSWER_MODES["Simple Answer"])
    prompt = f"""
You are TrustDoc AI, a careful retrieval-augmented assistant.

{UNTRUSTED_EVIDENCE_INSTRUCTION}

Start with a direct answer to the user's question in the first sentence.
Use only the provided document evidence.
If the evidence is insufficient, say exactly what is missing.
Bold the most important factual values, such as durations, deadlines, amounts, dates, and percentages.
Cite sources inline using bracketed citation numbers like [1].

Question:
{question}

Answer mode:
{instruction}

Evidence:
{_format_context(retrieved)}
""".strip()
    response = client.models.generate_content(model=settings.gemini_reasoning_model, contents=prompt)
    return response.text or ""


def _answer_locally(question: str, retrieved: list[RetrievedChunk], mode: str) -> str:
    if not retrieved:
        return "I could not find relevant evidence in the uploaded documents."

    evidence_lines = []
    for number, item in enumerate(retrieved[:4], start=1):
        evidence_lines.append(f"[{number}] {item.chunk.citation}: {_shorten(item.chunk.text, 420)}")

    if mode == "Bullet Summary":
        bullets = "\n".join(f"- {sentence}" for sentence in _best_sentences(question, retrieved, limit=5))
        return f"Direct answer:\n{bullets}\n\nEvidence:\n" + "\n".join(f"- {line}" for line in evidence_lines)
    if mode == "Action Items":
        return _extract_action_items(retrieved)
    if mode == "Email Draft":
        return _draft_email(question, retrieved)
    if mode == "Study Notes":
        return "Study notes:\n" + "\n".join(f"- {line}" for line in evidence_lines)
    if mode == "Interview Prep":
        return _interview_prep(question, retrieved)

    direct_answer = " ".join(_best_sentences(question, retrieved, limit=3))
    return f"Direct answer: {direct_answer}\n\nEvidence:\n" + "\n\n".join(evidence_lines)


def _answer_overview(retrieved: list[RetrievedChunk]) -> str:
    if not retrieved:
        return "I could not find readable indexed content to summarize."

    source_count = len({item.chunk.source for item in retrieved})
    lines = [
        f"- **{item.chunk.source}**: {_shorten(_first_informative_text(item.chunk.text), 260)}"
        for item in retrieved[:6]
    ]
    noun = "document" if source_count == 1 else "documents"
    return (
        f"The indexed {noun} appear to be about these main topics:\n\n"
        + "\n".join(lines)
        + "\n\nThis is an overview based on representative indexed chunks. Ask a more specific question for a more precise answer."
    )


def _format_context(retrieved: list[RetrievedChunk]) -> str:
    return "\n\n".join(
        f"[{number}] {item.chunk.citation}\nRelevance: {item.score:.3f}\n{item.chunk.text}"
        for number, item in enumerate(retrieved, start=1)
    )


def _extract_action_items(retrieved: list[RetrievedChunk]) -> str:
    action_words = ("must", "required", "submit", "provide", "complete", "notify", "send", "review")
    lines = []
    for item in retrieved:
        for sentence in _sentences(item.chunk.text):
            if any(word in sentence.lower() for word in action_words):
                lines.append(f"- {sentence.strip()} ({item.chunk.citation})")
    if not lines:
        return "No clear action items were found. Relevant evidence:\n" + "\n".join(
            f"- {_shorten(item.chunk.text, 240)} ({item.chunk.citation})" for item in retrieved[:3]
        )
    return "Action items found:\n" + "\n".join(lines[:8])


def _draft_email(question: str, retrieved: list[RetrievedChunk]) -> str:
    evidence = " ".join(_shorten(item.chunk.text, 180) for item in retrieved[:3])
    return (
        "Subject: Question About Document Details\n\n"
        "Hello,\n\n"
        f"I reviewed the available document evidence related to: {question}\n\n"
        f"The relevant document text appears to say: {evidence}\n\n"
        "Could you please confirm whether this interpretation is correct and whether any additional details apply?\n\n"
        "Thank you."
    )


def _interview_prep(question: str, retrieved: list[RetrievedChunk]) -> str:
    evidence = "\n".join(f"- {_shorten(item.chunk.text, 260)} ({item.chunk.citation})" for item in retrieved[:3])
    return (
        "Talking points:\n"
        f"{evidence}\n\n"
        "Possible follow-up questions:\n"
        f"- What source supports the answer to '{question}'?\n"
        "- What exceptions or edge cases are mentioned?\n"
        "- What information is not covered by the documents?"
    )


def _sentences(text: str) -> list[str]:
    return [sentence for sentence, _ in _sentence_records(text)]


def _sentence_records(text: str) -> list[tuple[str, str]]:
    records: list[tuple[str, str]] = []
    current_heading = ""
    for raw_line in text.splitlines():
        line = raw_line.strip().strip("#").strip()
        if not line:
            continue
        if _looks_like_heading(line):
            current_heading = line
            continue
        normalized = " ".join(line.split())
        parts = re.split(r"(?<=[.!?])\s+", normalized)
        records.extend((part.strip(), current_heading) for part in parts if part.strip())
    if records:
        return records
    normalized = " ".join(text.split())
    return [(part.strip(), "") for part in re.split(r"(?<=[.!?])\s+", normalized) if part.strip()]


def _best_sentences(question: str, retrieved: list[RetrievedChunk], limit: int) -> list[str]:
    question_terms = set(_important_terms(question))
    intent_terms = {
        "date",
        "deadline",
        "exception",
        "exceptions",
        "limitation",
        "limitations",
        "process",
        "requirement",
        "requirements",
        "required",
        "step",
        "steps",
        "window",
        "when",
    }
    domain_terms = question_terms - intent_terms
    scored: list[tuple[float, str]] = []
    for item_index, item in enumerate(retrieved[:4]):
        for sentence_index, (sentence, heading_text) in enumerate(_sentence_records(item.chunk.text)):
            heading_terms = set(_important_terms(heading_text))
            sentence_terms = set(_important_terms(sentence))
            overlap = len(question_terms & sentence_terms) if question_terms else 0
            heading_overlap = len(question_terms & heading_terms) if question_terms else 0
            domain_overlap = len(domain_terms & (sentence_terms | heading_terms))
            if domain_terms and not domain_overlap:
                continue
            if question_terms and not overlap and not heading_overlap:
                continue
            score = (
                (overlap * 3)
                + (heading_overlap * 3)
                + _intent_boost(question_terms, sentence, heading_terms)
                + item.score
                - (item_index * 0.5)
                - (sentence_index * 0.05)
            )
            if len(sentence.split()) >= 5:
                scored.append((score, f"{sentence} ({item.chunk.citation})"))

    scored.sort(key=lambda pair: pair[0], reverse=True)
    sentences = [sentence for _, sentence in scored[:limit]]
    if sentences:
        return sentences
    return [f"{_shorten(item.chunk.text, 220)} ({item.chunk.citation})" for item in retrieved[:limit]]


def _shorten(text: str, limit: int) -> str:
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def _important_terms(text: str) -> list[str]:
    stop = {
        "about",
        "after",
        "before",
        "could",
        "document",
        "documents",
        "given",
        "should",
        "request",
        "requests",
        "their",
        "there",
        "these",
        "those",
        "what",
        "when",
        "where",
        "which",
        "who",
        "why",
        "would",
    }
    terms: list[str] = []
    for word in re.findall(r"[a-zA-Z0-9]+", text.lower()):
        if len(word) <= 3 or word in stop:
            continue
        terms.append(word)
        if len(word) > 4 and word.endswith("s"):
            terms.append(word[:-1])
    return terms


def _headings(text: str) -> list[str]:
    headings: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip().strip("#").strip()
        if _looks_like_heading(line):
            headings.append(line)
    return headings


def _intent_boost(question_terms: set[str], sentence: str, heading_terms: set[str]) -> float:
    sentence_lower = sentence.lower()
    boost = 0.0
    if {"window", "deadline", "date", "when"} & question_terms:
        if re.search(r"\bwithin\b.*\b\d+\b.*\bday", sentence_lower):
            boost += 3.0
        if re.search(r"\b\d+\b.*\bday", sentence_lower):
            boost += 1.0
    if {"step", "steps", "process", "required", "requirements"} & question_terms:
        if any(word in sentence_lower for word in ("must", "required", "submit", "provide", "complete")):
            boost += 2.0
    if {"exception", "limitation"} & question_terms:
        if {"exception", "limitation", "non", "refundable"} & heading_terms:
            boost += 4.0
        if any(word in sentence_lower for word in ("except", "unless", "not", "may not", "non refundable")):
            boost += 2.0
    return boost


def _looks_like_heading(text: str) -> bool:
    words = text.split()
    if not words or len(words) > 9:
        return False
    if text.endswith((".", "?", "!")):
        return False
    title_words = sum(1 for word in words if word[:1].isupper())
    return text.isupper() or title_words >= max(1, len(words) // 2)


def _is_overview_question(question: str) -> bool:
    normalized = " ".join(question.lower().split())
    words = normalized.rstrip("?.!").split()
    overview_phrases = (
        "what is this document about",
        "what is the document about",
        "what is the given document about",
        "what are these documents about",
        "summarize this document",
        "summarize the document",
        "summarize the uploaded document",
        "give me an overview",
        "document summary",
    )
    if any(phrase in normalized for phrase in overview_phrases):
        return True
    return normalized.startswith("what is ") and normalized.endswith(" about") and len(words) <= 8


def _first_informative_text(text: str) -> str:
    pieces = [piece.strip("# ").strip() for piece in text.replace("\n", ". ").split(".") if piece.strip()]
    for piece in pieces:
        if len(piece.split()) >= 6:
            return piece
    return text
