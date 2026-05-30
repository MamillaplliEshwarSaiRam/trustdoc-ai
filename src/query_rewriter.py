"""Query rewriting for clearer retrieval."""

from __future__ import annotations

import re
from collections import Counter

from src.config import Settings
from src.models import Chunk
from src.prompt_injection import UNTRUSTED_EVIDENCE_INSTRUCTION


VAGUE_TERMS = {
    "it",
    "this",
    "that",
    "they",
    "them",
    "there",
    "these",
    "those",
    "details",
    "info",
    "information",
    "stuff",
    "thing",
    "things",
}


def rewrite_query(
    question: str,
    chunks: list[Chunk],
    settings: Settings,
    enabled: bool = True,
) -> str:
    question = " ".join(question.split())
    if not enabled or not question:
        return question
    if not _looks_vague(question):
        return question

    if settings.google_api_key:
        try:
            rewritten = _rewrite_with_gemini(question, chunks, settings)
            if rewritten:
                return rewritten
        except Exception:
            pass

    return _rewrite_locally(question, chunks)


def _rewrite_with_gemini(question: str, chunks: list[Chunk], settings: Settings) -> str:
    from google import genai

    client = genai.Client(api_key=settings.google_api_key)
    response = client.models.generate_content(
        model=settings.gemini_reasoning_model,
        contents=_build_prompt(question, chunks),
    )
    return _clean_rewrite(response.text or "", original=question)


def _build_prompt(question: str, chunks: list[Chunk]) -> str:
    context = "\n".join(
        f"- {chunk.citation}: {_shorten(chunk.text, 260)}"
        for chunk in chunks[:8]
    )
    return f"""
Rewrite the user's question into a clear retrieval query for a RAG system.

{UNTRUSTED_EVIDENCE_INSTRUCTION}

Original question:
{question}

Document context:
{context}

Rules:
- Preserve the user's intent.
- Resolve vague references like "this", "it", "details", or "requirements" using the document context.
- Do not answer the question.
- Return one plain-text rewritten query only.
""".strip()


def _rewrite_locally(question: str, chunks: list[Chunk]) -> str:
    subject = _best_subject(question, chunks)
    if not subject:
        return question
    lowered = question.lower()
    if subject.lower() in lowered:
        return question
    return f"{question} in {subject}"


def _looks_vague(question: str) -> bool:
    terms = re.findall(r"[a-zA-Z0-9]+", question.lower())
    if len(terms) <= 4:
        return True
    return any(term in VAGUE_TERMS for term in terms)


def _best_subject(question: str, chunks: list[Chunk]) -> str:
    question_terms = set(_terms(question)) - VAGUE_TERMS
    candidates: Counter[str] = Counter()
    for chunk in chunks[:8]:
        source = re.sub(r"\.(pdf|md|txt|docx)$", "", chunk.source, flags=re.IGNORECASE)
        source = source.replace("_", " ").replace("-", " ").strip()
        _score_candidate(source, question_terms, candidates, weight=1)
        for line in chunk.text.splitlines():
            clean = line.strip("# -\t ")
            if _looks_like_heading(clean):
                _score_candidate(clean, question_terms, candidates, weight=3)
    if not candidates:
        return ""
    subject, score = candidates.most_common(1)[0]
    return subject if score >= 2 else ""


def _score_candidate(candidate: str, question_terms: set[str], candidates: Counter[str], weight: int) -> None:
    if not candidate:
        return
    candidate_terms = set(_terms(candidate))
    overlap = question_terms & candidate_terms
    if overlap:
        candidates[candidate] += len(overlap) * weight


def _terms(text: str) -> list[str]:
    terms = []
    for word in re.findall(r"[a-zA-Z0-9]+", text.lower()):
        if len(word) <= 3:
            continue
        terms.append(word)
        if len(word) > 4 and word.endswith("s"):
            terms.append(word[:-1])
    return terms


def _clean_rewrite(text: str, original: str) -> str:
    cleaned = text.strip().strip('"').strip("'")
    cleaned = re.sub(r"^rewritten query:\s*", "", cleaned, flags=re.IGNORECASE).strip()
    cleaned = cleaned.splitlines()[0].strip() if cleaned else ""
    if not cleaned or len(cleaned) > 220:
        return original
    return cleaned


def _looks_like_heading(text: str) -> bool:
    words = text.split()
    if not words or len(words) > 10:
        return False
    if text.endswith((".", "?", "!")):
        return False
    title_words = sum(1 for word in words if word[:1].isupper())
    return text.isupper() or title_words >= max(1, len(words) // 2)


def _shorten(text: str, limit: int) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3].rstrip() + "..."
