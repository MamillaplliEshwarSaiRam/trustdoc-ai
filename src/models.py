"""Shared data models for the RAG pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class RawDocument:
    source: str
    text: str
    page: int | None = None


@dataclass(frozen=True)
class Chunk:
    id: str
    source: str
    text: str
    page: int | None = None
    chunk_index: int = 0

    @property
    def citation(self) -> str:
        if self.page is None:
            return f"{self.source}, chunk {self.chunk_index + 1}"
        return f"{self.source}, page {self.page}"


@dataclass(frozen=True)
class RetrievedChunk:
    chunk: Chunk
    score: float


@dataclass
class HealthReport:
    document_count: int
    chunk_count: int
    total_words: int
    duplicate_ratio: float
    short_document_count: int
    scanned_or_low_text_sources: list[str] = field(default_factory=list)
    top_terms: list[str] = field(default_factory=list)


@dataclass
class TrustReport:
    score: int
    label: str
    reasons: list[str]
    gaps: list[str]
    conflict_warnings: list[str]
    should_refuse: bool


@dataclass
class RAGResponse:
    answer: str
    retrieved: list[RetrievedChunk]
    trust: TrustReport
    mode: str
    rewritten_query: str | None = None
