"""Document collection analysis for demo-friendly diagnostics."""

from __future__ import annotations

from collections import Counter

from src.models import Chunk, HealthReport, RawDocument


STOP_WORDS = {
    "about",
    "after",
    "also",
    "because",
    "before",
    "between",
    "could",
    "document",
    "during",
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


def build_health_report(documents: list[RawDocument], chunks: list[Chunk]) -> HealthReport:
    all_words = _tokenize(" ".join(document.text for document in documents))
    text_fingerprints = Counter(_fingerprint(chunk.text) for chunk in chunks)
    duplicate_count = sum(count - 1 for count in text_fingerprints.values() if count > 1)
    low_text_sources = sorted({doc.source for doc in documents if doc.page is not None and len(doc.text) < 80})
    short_document_count = sum(1 for doc in documents if len(_tokenize(doc.text)) < 40)

    return HealthReport(
        document_count=len({doc.source for doc in documents}),
        chunk_count=len(chunks),
        total_words=len(all_words),
        duplicate_ratio=duplicate_count / max(len(chunks), 1),
        short_document_count=short_document_count,
        scanned_or_low_text_sources=low_text_sources,
        top_terms=[term for term, _ in Counter(all_words).most_common(8)],
    )


def _tokenize(text: str) -> list[str]:
    words = []
    for raw in text.lower().replace("/", " ").replace("-", " ").split():
        word = "".join(char for char in raw if char.isalnum())
        if len(word) > 3 and word not in STOP_WORDS:
            words.append(word)
    return words


def _fingerprint(text: str) -> str:
    words = _tokenize(text)
    return " ".join(words[:80])

