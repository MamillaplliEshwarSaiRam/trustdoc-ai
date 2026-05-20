"""A lightweight retriever with pure-Python TF-IDF, OpenAI, or Gemini embeddings."""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass

from src.config import Settings
from src.models import Chunk, RetrievedChunk


STOP_WORDS = {
    "about",
    "after",
    "also",
    "because",
    "before",
    "between",
    "could",
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
    "your",
}


@dataclass
class SearchIndex:
    chunks: list[Chunk]
    backend: str
    fallback_reason: str | None = None
    _idf: dict[str, float] | None = None
    _vectors: list[dict[str, float]] | None = None
    _norms: list[float] | None = None
    _embedding_matrix: list[list[float]] | None = None
    _client: object | None = None
    _embedding_model: str | None = None
    _provider: str | None = None

    @classmethod
    def build(cls, chunks: list[Chunk], settings: Settings, embedding_provider: str = "Local TF-IDF") -> "SearchIndex":
        if embedding_provider == "OpenAI" and settings.openai_api_key:
            try:
                return cls._build_openai(chunks, settings)
            except Exception as exc:
                index = cls._build_tfidf(chunks)
                index.fallback_reason = _safe_error_message(exc)
                return index
        if embedding_provider == "Google Gemini" and settings.google_api_key:
            try:
                return cls._build_gemini(chunks, settings)
            except Exception as exc:
                index = cls._build_tfidf(chunks)
                index.fallback_reason = _safe_error_message(exc)
                return index
        return cls._build_tfidf(chunks)

    @classmethod
    def _build_tfidf(cls, chunks: list[Chunk]) -> "SearchIndex":
        tokenized = [_tokens(chunk.text) for chunk in chunks]
        document_frequency: defaultdict[str, int] = defaultdict(int)
        for terms in tokenized:
            for term in set(terms):
                document_frequency[term] += 1

        document_count = max(len(chunks), 1)
        idf = {
            term: math.log((1 + document_count) / (1 + frequency)) + 1
            for term, frequency in document_frequency.items()
        }
        vectors = [_tfidf_vector(terms, idf) for terms in tokenized]
        norms = [_norm(vector) for vector in vectors]
        return cls(chunks=chunks, backend="Local TF-IDF", _idf=idf, _vectors=vectors, _norms=norms)

    @classmethod
    def _build_openai(cls, chunks: list[Chunk], settings: Settings) -> "SearchIndex":
        from openai import OpenAI

        client = OpenAI(api_key=settings.openai_api_key)
        texts = [chunk.text for chunk in chunks]
        response = client.embeddings.create(model=settings.openai_embedding_model, input=texts)
        matrix = [item.embedding for item in response.data]
        return cls(
            chunks=chunks,
            backend="OpenAI embeddings",
            _embedding_matrix=matrix,
            _client=client,
            _embedding_model=settings.openai_embedding_model,
            _provider="openai",
        )

    @classmethod
    def _build_gemini(cls, chunks: list[Chunk], settings: Settings) -> "SearchIndex":
        from google import genai

        client = genai.Client(api_key=settings.google_api_key)
        texts = [chunk.text for chunk in chunks]
        matrix = _gemini_embed_texts(client, settings.gemini_embedding_model, texts)
        return cls(
            chunks=chunks,
            backend="Google Gemini embeddings",
            _embedding_matrix=matrix,
            _client=client,
            _embedding_model=settings.gemini_embedding_model,
            _provider="gemini",
        )

    def search(self, query: str, top_k: int = 6) -> list[RetrievedChunk]:
        if not self.chunks:
            return []
        if self.backend == "OpenAI embeddings":
            return self._search_openai(query, top_k)
        if self.backend == "Google Gemini embeddings":
            return self._search_gemini(query, top_k)
        return self._search_tfidf(query, top_k)

    def overview(self, top_k: int = 6) -> list[RetrievedChunk]:
        """Return representative chunks for broad summary questions."""
        representative: list[RetrievedChunk] = []
        seen_sources: set[str] = set()
        for chunk in self.chunks:
            if chunk.source in seen_sources:
                continue
            representative.append(RetrievedChunk(chunk=chunk, score=0.82))
            seen_sources.add(chunk.source)
            if len(representative) >= top_k:
                break
        if len(representative) < min(top_k, len(self.chunks)):
            used_ids = {item.chunk.id for item in representative}
            for chunk in self.chunks:
                if chunk.id not in used_ids:
                    representative.append(RetrievedChunk(chunk=chunk, score=0.72))
                if len(representative) >= top_k:
                    break
        return representative

    def _search_tfidf(self, query: str, top_k: int) -> list[RetrievedChunk]:
        assert self._idf is not None
        assert self._vectors is not None
        assert self._norms is not None
        query_vector = _tfidf_vector(_tokens(query), self._idf)
        query_norm = _norm(query_vector)
        scores = [
            _cosine(query_vector, query_norm, vector, norm)
            for vector, norm in zip(self._vectors, self._norms)
        ]
        return self._rank(scores, top_k)

    def _search_openai(self, query: str, top_k: int) -> list[RetrievedChunk]:
        assert self._client is not None
        assert self._embedding_matrix is not None
        response = self._client.embeddings.create(model=self._embedding_model, input=[query])
        query_vector = response.data[0].embedding
        query_norm = math.sqrt(sum(value * value for value in query_vector))
        scores = [_dense_cosine(query_vector, query_norm, vector) for vector in self._embedding_matrix]
        return self._rank(scores, top_k)

    def _search_gemini(self, query: str, top_k: int) -> list[RetrievedChunk]:
        assert self._client is not None
        assert self._embedding_matrix is not None
        query_vector = _gemini_embed_texts(self._client, self._embedding_model, [query])[0]
        query_norm = math.sqrt(sum(value * value for value in query_vector))
        scores = [_dense_cosine(query_vector, query_norm, vector) for vector in self._embedding_matrix]
        return self._rank(scores, top_k)

    def _rank(self, scores: list[float], top_k: int) -> list[RetrievedChunk]:
        ranked_indices = sorted(range(len(scores)), key=lambda index: scores[index], reverse=True)[:top_k]
        return [
            RetrievedChunk(chunk=self.chunks[index], score=float(scores[index]))
            for index in ranked_indices
            if float(scores[index]) > 0
        ]


def _tokens(text: str) -> list[str]:
    terms = [
        word
        for word in re.findall(r"[a-zA-Z0-9]+", text.lower())
        if len(word) > 3 and word not in STOP_WORDS
    ]
    bigrams = [f"{terms[index]} {terms[index + 1]}" for index in range(len(terms) - 1)]
    return terms + bigrams


def _tfidf_vector(terms: list[str], idf: dict[str, float]) -> dict[str, float]:
    counts = Counter(term for term in terms if term in idf)
    total = max(sum(counts.values()), 1)
    return {term: (count / total) * idf[term] for term, count in counts.items()}


def _norm(vector: dict[str, float]) -> float:
    return math.sqrt(sum(value * value for value in vector.values()))


def _cosine(left: dict[str, float], left_norm: float, right: dict[str, float], right_norm: float) -> float:
    if left_norm == 0 or right_norm == 0:
        return 0.0
    if len(left) > len(right):
        left, right = right, left
    dot = sum(value * right.get(term, 0.0) for term, value in left.items())
    return dot / (left_norm * right_norm)


def _dense_cosine(left: list[float], left_norm: float, right: list[float]) -> float:
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    dot = sum(left_value * right_value for left_value, right_value in zip(left, right))
    return dot / (left_norm * right_norm)


def _gemini_embed_texts(client: object, model: str, texts: list[str]) -> list[list[float]]:
    embeddings: list[list[float]] = []
    for text in texts:
        response = client.models.embed_content(model=model, contents=text)
        embedding = getattr(response, "embedding", None)
        if embedding is None and getattr(response, "embeddings", None):
            embedding = response.embeddings[0]
        values = getattr(embedding, "values", None)
        if values is None:
            raise RuntimeError("Gemini embedding response did not include embedding values.")
        embeddings.append(list(values))
    return embeddings


def _safe_error_message(error: Exception) -> str:
    message = str(error).strip()
    if not message:
        message = error.__class__.__name__
    return message[:500]
