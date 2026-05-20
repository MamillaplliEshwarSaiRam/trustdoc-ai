"""Adaptive local context compression before LLM generation."""

from __future__ import annotations

import math
import re
from collections import Counter

from src.models import Chunk, RetrievedChunk


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

BROAD_QUERY_TERMS = {
    "summarize",
    "summary",
    "overview",
    "compare",
    "comparison",
    "conflict",
    "conflicts",
    "difference",
    "differences",
    "similarities",
    "study",
    "notes",
}

CONDITION_TERMS = {
    "after",
    "before",
    "except",
    "exception",
    "exceptions",
    "excluding",
    "unless",
    "if",
    "only",
    "provided",
    "required",
    "must",
    "cannot",
    "not",
    "however",
}


def compress_retrieved_context(
    question: str,
    retrieved: list[RetrievedChunk],
    mode: str,
) -> list[RetrievedChunk]:
    """Return a smaller evidence set for API prompts while preserving citations."""
    if not retrieved or not _should_compress(question, mode):
        return retrieved

    sentence_limit = _sentence_limit(mode)
    compressed: list[RetrievedChunk] = []
    for item in retrieved:
        compressed_text = _compress_text(question, item.chunk.text, sentence_limit)
        if not compressed_text:
            compressed.append(item)
            continue
        compressed.append(
            RetrievedChunk(
                chunk=Chunk(
                    id=item.chunk.id,
                    source=item.chunk.source,
                    text=compressed_text,
                    page=item.chunk.page,
                    chunk_index=item.chunk.chunk_index,
                ),
                score=item.score,
            )
        )
    return compressed


def _should_compress(question: str, mode: str) -> bool:
    question_terms = set(_terms(question))
    lower_question = question.lower()
    if question_terms & BROAD_QUERY_TERMS:
        return False
    if "what is the document about" in lower_question or "what is this about" in lower_question:
        return False
    if mode in {"Detailed Explanation", "Study Notes"} and len(question_terms) <= 4:
        return False
    return True


def _sentence_limit(mode: str) -> int:
    if mode == "Conflict Analysis":
        return 5
    if mode in {"Action Items", "Bullet Summary"}:
        return 4
    if mode in {"Email Draft", "Interview Prep"}:
        return 3
    return 3


def _compress_text(question: str, text: str, sentence_limit: int) -> str:
    records = _sentence_records(text)
    if len(text) <= 260 or len(records) <= sentence_limit:
        return text

    query_terms = _terms(question)
    scores = _score_sentences(query_terms, records)
    chosen_indices = _choose_sentence_indices(scores, records, sentence_limit)
    if not chosen_indices:
        return _shorten(text, 520)

    heading = _first_relevant_heading(records, chosen_indices)
    selected = [records[index][1] for index in sorted(chosen_indices)]
    compressed = "\n".join(([heading] if heading else []) + selected)
    compressed = _shorten(compressed, 900)
    if len(compressed) < 80:
        return _shorten(text, 520)
    return compressed


def _sentence_records(text: str) -> list[tuple[str, str]]:
    records: list[tuple[str, str]] = []
    current_heading = ""
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        clean_line = line.strip("#").strip()
        if _looks_like_heading(line):
            current_heading = clean_line
            continue
        parts = re.split(r"(?<=[.!?])\s+", " ".join(clean_line.split()))
        for part in parts:
            sentence = part.strip()
            if sentence:
                records.append((current_heading, sentence))
    if records:
        return records

    normalized = " ".join(text.split())
    return [("", sentence.strip()) for sentence in re.split(r"(?<=[.!?])\s+", normalized) if sentence.strip()]


def _score_sentences(query_terms: list[str], records: list[tuple[str, str]]) -> list[float]:
    sentence_terms = [_terms(f"{heading} {sentence}") for heading, sentence in records]
    bm25 = _bm25_scores(query_terms, sentence_terms)
    query_term_set = set(query_terms)

    scores: list[float] = []
    for index, ((heading, sentence), terms) in enumerate(zip(records, sentence_terms)):
        sentence_lower = sentence.lower()
        heading_overlap = len(query_term_set & set(_terms(heading)))
        exact_overlap = len(query_term_set & set(terms))
        number_boost = _number_overlap(query_terms, terms) * 0.2
        condition_boost = 0.16 if _has_condition_signal(sentence_lower) else 0.0
        position_penalty = min(index * 0.015, 0.12)
        scores.append(bm25[index] + (heading_overlap * 0.14) + (exact_overlap * 0.04) + number_boost + condition_boost - position_penalty)
    return scores


def _choose_sentence_indices(
    scores: list[float],
    records: list[tuple[str, str]],
    sentence_limit: int,
) -> set[int]:
    ranked = sorted(range(len(scores)), key=lambda index: scores[index], reverse=True)
    positive = [index for index in ranked if scores[index] > 0]
    chosen = set(positive[:sentence_limit])

    for index in list(chosen):
        for neighbor in (index - 1, index + 1):
            if neighbor < 0 or neighbor >= len(records) or len(chosen) >= sentence_limit + 1:
                continue
            if _has_condition_signal(records[neighbor][1].lower()):
                chosen.add(neighbor)

    return chosen


def _first_relevant_heading(records: list[tuple[str, str]], chosen_indices: set[int]) -> str:
    for index in sorted(chosen_indices):
        heading = records[index][0].strip()
        if heading:
            return heading
    return ""


def _bm25_scores(query_terms: list[str], documents: list[list[str]]) -> list[float]:
    if not query_terms or not documents:
        return [0.0 for _ in documents]

    query_counter = Counter(query_terms)
    document_count = max(len(documents), 1)
    average_length = sum(len(document) for document in documents) / document_count or 1.0
    document_frequency: Counter[str] = Counter()
    for document in documents:
        document_frequency.update(set(document))

    raw_scores: list[float] = []
    k1 = 1.5
    b = 0.75
    for document in documents:
        term_counts = Counter(document)
        document_length = len(document) or 1
        score = 0.0
        for term, query_weight in query_counter.items():
            frequency = term_counts.get(term, 0)
            if not frequency:
                continue
            idf = math.log(1 + ((document_count - document_frequency[term] + 0.5) / (document_frequency[term] + 0.5)))
            denominator = frequency + k1 * (1 - b + b * (document_length / average_length))
            score += query_weight * idf * ((frequency * (k1 + 1)) / denominator)
        raw_scores.append(score)

    max_score = max(raw_scores, default=0.0)
    if max_score <= 0:
        return [0.0 for _ in raw_scores]
    return [score / max_score for score in raw_scores]


def _terms(text: str) -> list[str]:
    return [
        word
        for word in re.findall(r"[a-zA-Z0-9]+", text.lower())
        if len(word) > 2 and word not in STOP_WORDS
    ]


def _number_overlap(query_terms: list[str], text_terms: list[str]) -> float:
    query_numbers = {term for term in query_terms if term.isdigit()}
    if not query_numbers:
        return 0.0
    return len(query_numbers & set(text_terms)) / len(query_numbers)


def _has_condition_signal(sentence_lower: str) -> bool:
    return any(re.search(rf"\b{re.escape(term)}\b", sentence_lower) for term in CONDITION_TERMS)


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


def _shorten(text: str, limit: int) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3].rstrip() + "..."
