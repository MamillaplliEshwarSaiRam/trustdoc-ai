"""Lightweight graph-enhanced retrieval for document relationships."""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field

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
    "during",
    "given",
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


@dataclass
class KnowledgeGraph:
    chunks_by_id: dict[str, Chunk]
    entity_to_chunk_ids: dict[str, set[str]]
    chunk_id_to_entities: dict[str, set[str]]
    related_entities: dict[str, Counter[str]] = field(default_factory=dict)

    @property
    def entity_count(self) -> int:
        return len(self.entity_to_chunk_ids)

    @property
    def edge_count(self) -> int:
        return sum(len(counter) for counter in self.related_entities.values()) // 2


def build_knowledge_graph(chunks: list[Chunk]) -> KnowledgeGraph:
    chunks_by_id = {chunk.id: chunk for chunk in chunks}
    entity_to_chunk_ids: dict[str, set[str]] = defaultdict(set)
    chunk_id_to_entities: dict[str, set[str]] = {}
    related_entities: dict[str, Counter[str]] = defaultdict(Counter)

    for chunk in chunks:
        entities = _extract_entities(chunk.text)
        chunk_id_to_entities[chunk.id] = entities
        for entity in entities:
            entity_to_chunk_ids[entity].add(chunk.id)

        ordered = sorted(entities)
        for index, left in enumerate(ordered):
            for right in ordered[index + 1 :]:
                related_entities[left][right] += 1
                related_entities[right][left] += 1

    return KnowledgeGraph(
        chunks_by_id=chunks_by_id,
        entity_to_chunk_ids=dict(entity_to_chunk_ids),
        chunk_id_to_entities=chunk_id_to_entities,
        related_entities=dict(related_entities),
    )


def graph_enhance_retrieval(
    query: str,
    base_retrieved: list[RetrievedChunk],
    graph: KnowledgeGraph | None,
    top_k: int,
) -> list[RetrievedChunk]:
    if graph is None or not base_retrieved:
        return base_retrieved

    scores: dict[str, float] = {item.chunk.id: item.score for item in base_retrieved}
    query_entities = _match_graph_entities(query, graph)

    for entity in query_entities:
        _boost_chunks_for_entity(scores, graph, entity, boost=0.18)
        for related, weight in graph.related_entities.get(entity, {}).most_common(6):
            _boost_chunks_for_entity(scores, graph, related, boost=min(0.10, 0.025 * weight))

    for item in base_retrieved[:3]:
        for entity in graph.chunk_id_to_entities.get(item.chunk.id, set()):
            for related, weight in graph.related_entities.get(entity, {}).most_common(4):
                _boost_chunks_for_entity(scores, graph, related, boost=min(0.06, 0.015 * weight))

    ranked = sorted(scores.items(), key=lambda pair: pair[1], reverse=True)
    enhanced: list[RetrievedChunk] = []
    for chunk_id, score in ranked[:top_k]:
        chunk = graph.chunks_by_id.get(chunk_id)
        if chunk:
            enhanced.append(RetrievedChunk(chunk=chunk, score=min(score, 1.0)))
    return enhanced


def _boost_chunks_for_entity(
    scores: dict[str, float],
    graph: KnowledgeGraph,
    entity: str,
    boost: float,
) -> None:
    for chunk_id in graph.entity_to_chunk_ids.get(entity, set()):
        scores[chunk_id] = scores.get(chunk_id, 0.0) + boost


def _match_graph_entities(query: str, graph: KnowledgeGraph) -> set[str]:
    query_entities = _extract_entities(query)
    query_text = _normalize(query)
    matches = set()
    for entity in graph.entity_to_chunk_ids:
        if entity in query_entities or entity in query_text:
            matches.add(entity)
    return matches


def _extract_entities(text: str, limit: int = 14) -> set[str]:
    entities: Counter[str] = Counter()

    for line in text.splitlines():
        stripped = line.strip("# -\t ")
        if _looks_like_heading(stripped):
            for term in _term_candidates(stripped):
                entities[term] += 4

    for phrase in re.findall(r"\b[A-Z][A-Za-z0-9]+(?:\s+(?:[A-Z][A-Za-z0-9]+|and|of|for|to|in)){0,4}", text):
        normalized = _normalize(phrase)
        if _valid_entity(normalized):
            entities[normalized] += 3

    for term in _term_candidates(text):
        entities[term] += 1

    return {entity for entity, _ in entities.most_common(limit)}


def _term_candidates(text: str) -> list[str]:
    words = [
        word
        for word in re.findall(r"[a-zA-Z0-9]+", text.lower())
        if len(word) > 3 and word not in STOP_WORDS
    ]
    single_terms = [word for word in words if not word.isdigit()]
    bigrams = [
        f"{words[index]} {words[index + 1]}"
        for index in range(len(words) - 1)
        if not words[index].isdigit() and not words[index + 1].isdigit()
    ]
    return single_terms[:18] + bigrams[:12]


def _looks_like_heading(text: str) -> bool:
    if not text:
        return False
    if len(text) < 4 or len(text) > 90:
        return False
    title_words = sum(1 for word in text.split() if word[:1].isupper())
    return text.isupper() or title_words >= max(1, len(text.split()) // 2)


def _valid_entity(entity: str) -> bool:
    if len(entity) < 4:
        return False
    if entity in STOP_WORDS:
        return False
    return not entity.isdigit()


def _normalize(text: str) -> str:
    return " ".join(re.findall(r"[a-zA-Z0-9]+", text.lower()))
