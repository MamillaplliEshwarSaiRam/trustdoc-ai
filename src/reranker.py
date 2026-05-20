"""Local reranking and adaptive context selection for retrieved chunks."""

from __future__ import annotations

import math
import re
from collections import Counter

from src.models import RetrievedChunk


STOP_WORDS = {
    "about",
    "after",
    "also",
    "before",
    "between",
    "could",
    "document",
    "documents",
    "given",
    "should",
    "their",
    "there",
    "these",
    "those",
    "through",
    "under",
    "until",
    "where",
    "which",
    "while",
    "would",
}


def rerank_and_select(
    query: str,
    candidates: list[RetrievedChunk],
    max_chunks: int,
    max_context_chars: int,
) -> list[RetrievedChunk]:
    """Rerank candidates locally, then keep only the useful evidence window."""
    if not candidates:
        return []

    deduped = _dedupe_by_chunk(candidates)
    reranked = _rerank(query, deduped)
    return _adaptive_select(reranked, max_chunks=max_chunks, max_context_chars=max_context_chars)


def _dedupe_by_chunk(candidates: list[RetrievedChunk]) -> list[RetrievedChunk]:
    best_by_id: dict[str, RetrievedChunk] = {}
    for item in candidates:
        current = best_by_id.get(item.chunk.id)
        if current is None or item.score > current.score:
            best_by_id[item.chunk.id] = item
    return list(best_by_id.values())


def _rerank(query: str, candidates: list[RetrievedChunk]) -> list[RetrievedChunk]:
    query_terms = _terms(query)
    max_base_score = max((item.score for item in candidates), default=0.0) or 1.0
    bm25_scores = _bm25_scores(query_terms, candidates)

    reranked: list[RetrievedChunk] = []
    for index, item in enumerate(candidates):
        base_score = max(item.score, 0.0) / max_base_score
        lexical_score = bm25_scores[index]
        structure_score = _structure_score(query, query_terms, item.chunk.text)
        final_score = (0.70 * base_score) + (0.22 * lexical_score) + (0.08 * structure_score)
        reranked.append(RetrievedChunk(chunk=item.chunk, score=min(final_score, 1.0)))

    return sorted(reranked, key=lambda item: item.score, reverse=True)


def _adaptive_select(
    reranked: list[RetrievedChunk],
    max_chunks: int,
    max_context_chars: int,
) -> list[RetrievedChunk]:
    if not reranked:
        return []

    selected: list[RetrievedChunk] = []
    used_chars = 0
    top_score = reranked[0].score
    previous_score = top_score
    min_score = max(0.12, top_score * 0.38)

    for item in reranked:
        if len(selected) >= max_chunks:
            break

        text_size = len(item.chunk.text)
        if selected and used_chars + text_size > max_context_chars:
            continue

        if selected and item.score < min_score:
            break

        if len(selected) >= 2 and previous_score - item.score > 0.28:
            break

        selected.append(item)
        used_chars += text_size
        previous_score = item.score

    if not selected and reranked[0].score > 0:
        return [reranked[0]]
    return selected


def _bm25_scores(query_terms: list[str], candidates: list[RetrievedChunk]) -> list[float]:
    if not query_terms:
        return [0.0 for _ in candidates]

    query_counter = Counter(query_terms)
    document_terms = [
        _terms(f"{item.chunk.source.replace('_', ' ').replace('-', ' ')} {item.chunk.text}")
        for item in candidates
    ]
    document_count = max(len(document_terms), 1)
    average_length = sum(len(terms) for terms in document_terms) / document_count
    average_length = average_length or 1.0

    document_frequency: Counter[str] = Counter()
    for terms in document_terms:
        document_frequency.update(set(terms))

    k1 = 1.5
    b = 0.75
    raw_scores: list[float] = []
    for terms in document_terms:
        term_counts = Counter(terms)
        document_length = len(terms) or 1
        score = 0.0
        for term, query_weight in query_counter.items():
            frequency = term_counts.get(term, 0)
            if not frequency:
                continue
            idf = max(0.0, _bm25_idf(document_count, document_frequency.get(term, 0)))
            denominator = frequency + k1 * (1 - b + b * (document_length / average_length))
            score += query_weight * idf * ((frequency * (k1 + 1)) / denominator)
        raw_scores.append(score)

    max_score = max(raw_scores, default=0.0)
    if max_score <= 0:
        return [0.0 for _ in raw_scores]
    return [score / max_score for score in raw_scores]


def _bm25_idf(document_count: int, document_frequency: int) -> float:
    return math.log(1 + ((document_count - document_frequency + 0.5) / (document_frequency + 0.5)))


def _structure_score(query: str, query_terms: list[str], text: str) -> float:
    score = 0.0
    query_term_set = set(query_terms)
    lower_query = query.lower()
    lower_text = text.lower()

    for line in text.splitlines()[:8]:
        stripped = line.strip()
        heading_text = stripped.strip("#").strip()
        if _looks_like_heading(stripped) and query_term_set & set(_terms(heading_text)):
            score += 0.45
            break

    if any(word in lower_query for word in ("when", "deadline", "date", "long", "window")):
        if re.search(r"\b(within|before|after|by|days?|months?|business day)\b", lower_text):
            score += 0.25

    if any(word in lower_query for word in ("who", "eligible", "eligibility", "requirement", "requirements")):
        if re.search(r"\b(eligible|must|required|may|can|cannot|role|employee|customer|vendor)\b", lower_text):
            score += 0.25

    if any(word in lower_query for word in ("exception", "excluded", "not refundable", "cannot")):
        if re.search(r"\b(exception|except|exclude|excluded|not|cannot|non-refundable)\b", lower_text):
            score += 0.25

    if re.search(r"(^|\n)\s*(?:[-*]|\d+[.)])\s+", text):
        score += 0.10

    return min(score, 1.0)


def _terms(text: str) -> list[str]:
    return [
        word
        for word in re.findall(r"[a-zA-Z0-9]+", text.lower())
        if len(word) > 2 and word not in STOP_WORDS
    ]


def _looks_like_heading(text: str) -> bool:
    stripped = text.strip()
    if stripped.startswith("#"):
        return True
    if not stripped or len(stripped) > 90:
        return False
    if stripped.endswith(":"):
        return True
    words = stripped.split()
    title_words = sum(1 for word in words if word[:1].isupper())
    return len(words) <= 8 and title_words >= max(1, len(words) // 2)
