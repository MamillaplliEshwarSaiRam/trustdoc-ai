"""Structure-aware recursive text chunking for citation-friendly retrieval."""

from __future__ import annotations

import hashlib
import re

from src.models import Chunk, RawDocument


HEADING_PATTERN = re.compile(
    r"(?m)^(#{1,6}\s+.+|[A-Z][A-Z0-9 ,:;/&()'\-]{6,}|(?:\d+(?:\.\d+)*)\s+[A-Z][^\n]{4,})$"
)
SENTENCE_PATTERN = re.compile(r"(?<=[.!?])\s+")


def split_documents(
    documents: list[RawDocument],
    chunk_size: int = 900,
    chunk_overlap: int = 140,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    for document in documents:
        sections = _split_by_structure(document.text)
        texts = _recursive_chunks(sections, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        for index, text in enumerate(texts):
            chunk_id = hashlib.sha1(
                f"{document.source}:{document.page}:{index}:{text[:80]}".encode("utf-8")
            ).hexdigest()[:16]
            chunks.append(
                Chunk(
                    id=chunk_id,
                    source=document.source,
                    page=document.page,
                    text=text,
                    chunk_index=index,
                )
            )
    return chunks


def _split_by_structure(text: str) -> list[str]:
    text = _normalize_newlines(text)
    if not text:
        return []

    matches = list(HEADING_PATTERN.finditer(text))
    if not matches:
        return _paragraph_blocks(text)

    sections: list[str] = []
    intro = text[: matches[0].start()].strip()
    if intro:
        sections.extend(_paragraph_blocks(intro))

    for index, match in enumerate(matches):
        start = match.start()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        section = text[start:end].strip()
        if section:
            sections.append(section)
    return sections


def _recursive_chunks(sections: list[str], chunk_size: int, chunk_overlap: int) -> list[str]:
    chunks: list[str] = []
    for section in sections:
        section = _compact_lines(section)
        if not section:
            continue
        if len(section) <= chunk_size:
            chunks.append(section)
            continue
        chunks.extend(_split_long_section(section, chunk_size, chunk_overlap))
    return _merge_short_chunks(chunks, chunk_size)


def _split_long_section(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    paragraphs = _paragraph_blocks(text)
    if len(paragraphs) > 1:
        return _pack_units(paragraphs, chunk_size, chunk_overlap)

    sentences = _sentence_blocks(text)
    if len(sentences) > 1:
        return _pack_units(sentences, chunk_size, chunk_overlap)

    return _hard_split(text, chunk_size, chunk_overlap)


def _pack_units(units: list[str], chunk_size: int, chunk_overlap: int) -> list[str]:
    chunks: list[str] = []
    current: list[str] = []
    current_length = 0

    for unit in units:
        unit = unit.strip()
        if not unit:
            continue
        if len(unit) > chunk_size:
            if current:
                chunks.append("\n\n".join(current).strip())
                current = []
                current_length = 0
            chunks.extend(_split_long_section(unit, chunk_size, chunk_overlap))
            continue

        separator_length = 2 if current else 0
        if current and current_length + separator_length + len(unit) > chunk_size:
            chunks.append("\n\n".join(current).strip())
            current = _overlap_units(current, chunk_overlap)
            current_length = len("\n\n".join(current))

        current.append(unit)
        current_length += separator_length + len(unit)

    if current:
        chunks.append("\n\n".join(current).strip())
    return chunks


def _overlap_units(units: list[str], chunk_overlap: int) -> list[str]:
    overlap: list[str] = []
    total = 0
    for unit in reversed(units):
        if total + len(unit) > chunk_overlap and overlap:
            break
        overlap.insert(0, unit)
        total += len(unit)
    return overlap


def _merge_short_chunks(chunks: list[str], chunk_size: int) -> list[str]:
    merged: list[str] = []
    for chunk in chunks:
        if not merged:
            merged.append(chunk)
            continue
        previous = merged[-1]
        if len(previous) < chunk_size * 0.35 and len(previous) + len(chunk) + 2 <= chunk_size:
            merged[-1] = f"{previous}\n\n{chunk}"
        else:
            merged.append(chunk)
    return merged


def _hard_split(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    text = " ".join(text.split())
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = max(end - chunk_overlap, start + 1)
    return chunks


def _paragraph_blocks(text: str) -> list[str]:
    return [block.strip() for block in re.split(r"\n\s*\n", text) if block.strip()]


def _sentence_blocks(text: str) -> list[str]:
    compact = " ".join(text.split())
    return [sentence.strip() for sentence in SENTENCE_PATTERN.split(compact) if sentence.strip()]


def _normalize_newlines(text: str) -> str:
    lines = [line.rstrip() for line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n")]
    return "\n".join(lines).strip()


def _compact_lines(text: str) -> str:
    lines = [line.strip() for line in text.splitlines()]
    compacted: list[str] = []
    previous_blank = False
    for line in lines:
        is_blank = not line
        if is_blank and previous_blank:
            continue
        compacted.append(line)
        previous_blank = is_blank
    return "\n".join(compacted).strip()
