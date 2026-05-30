"""Content-aware question suggestions for indexed documents."""

from __future__ import annotations

import json
import re
from collections import Counter

from src.config import Settings
from src.models import Chunk
from src.prompt_injection import UNTRUSTED_EVIDENCE_INSTRUCTION


FALLBACK_QUESTIONS = [
    "What are the main points in these documents?",
    "What deadlines, requirements, or action items are mentioned?",
    "What exceptions, limitations, or unclear details should I pay attention to?",
]


def suggest_questions(
    chunks: list[Chunk],
    settings: Settings | None = None,
    limit: int = 5,
    use_llm: bool = True,
) -> list[str]:
    if not chunks:
        return FALLBACK_QUESTIONS[:limit]
    if use_llm and settings and settings.google_api_key:
        try:
            questions = _suggest_with_gemini(chunks, settings, limit)
            if questions:
                return questions[:limit]
        except Exception:
            pass
    return _suggest_locally(chunks, limit)


def _suggest_with_gemini(chunks: list[Chunk], settings: Settings, limit: int) -> list[str]:
    from google import genai

    client = genai.Client(api_key=settings.google_api_key)
    response = client.models.generate_content(
        model=settings.gemini_reasoning_model,
        contents=_build_prompt(chunks, limit),
    )
    return _parse_questions(response.text or "")


def _build_prompt(chunks: list[Chunk], limit: int) -> str:
    evidence = "\n\n".join(
        f"[{index}] {chunk.citation}\n{_shorten(chunk.text, 700)}"
        for index, chunk in enumerate(chunks[:10], start=1)
    )
    return f"""
You generate high-quality suggested questions for a document Q&A app.

{UNTRUSTED_EVIDENCE_INSTRUCTION}

Documents/chunks:
{evidence}

Create {limit} specific, useful questions that a user would naturally ask about these documents.
Rules:
- Questions must be directly grounded in the provided document content.
- Avoid generic placeholders like "this topic" or single-word topics.
- Prefer questions about purpose, requirements, deadlines, exceptions, responsibilities, risks, comparison, or preparation.
- Return only JSON in this shape:
{{"questions": ["question 1", "question 2"]}}
""".strip()


def _suggest_locally(chunks: list[Chunk], limit: int) -> list[str]:
    headings = _extract_headings(chunks)
    source_names = _source_names(chunks)
    key_phrases = _key_phrases(chunks)

    questions: list[str] = []
    if source_names:
        questions.append(f"What is the purpose of {source_names[0]}?")
    if headings:
        questions.append(f"What should I know about {headings[0]}?")
    if len(headings) > 1:
        questions.append(f"How are {headings[0]} and {headings[1]} related?")
    if key_phrases:
        questions.append(f"What requirements or steps are mentioned for {key_phrases[0]}?")
    questions.append("What deadlines, requirements, or action items are mentioned?")
    questions.append("What exceptions, limitations, or missing details should I pay attention to?")

    return _dedupe_questions(questions)[:limit] or FALLBACK_QUESTIONS[:limit]


def _extract_headings(chunks: list[Chunk]) -> list[str]:
    headings: list[str] = []
    for chunk in chunks:
        for line in chunk.text.splitlines():
            clean = line.strip("# -\t ")
            if _looks_like_heading(clean):
                headings.append(clean)
    return _dedupe_terms(headings, limit=8)


def _source_names(chunks: list[Chunk]) -> list[str]:
    names = []
    for chunk in chunks:
        name = re.sub(r"\.(pdf|md|txt|docx)$", "", chunk.source, flags=re.IGNORECASE)
        name = name.replace("_", " ").replace("-", " ").strip()
        if name:
            names.append(name)
    return _dedupe_terms(names, limit=4)


def _key_phrases(chunks: list[Chunk]) -> list[str]:
    counter: Counter[str] = Counter()
    stop = {
        "about",
        "after",
        "before",
        "could",
        "document",
        "section",
        "their",
        "there",
        "these",
        "those",
        "which",
        "would",
    }
    for chunk in chunks:
        words = [
            word
            for word in re.findall(r"[a-zA-Z0-9]+", chunk.text.lower())
            if len(word) > 3 and word not in stop
        ]
        for index in range(len(words) - 1):
            phrase = f"{words[index]} {words[index + 1]}"
            counter[phrase] += 1
    return [phrase for phrase, _ in counter.most_common(6)]


def _parse_questions(raw_text: str) -> list[str]:
    text = raw_text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text, flags=re.IGNORECASE).strip()
        text = re.sub(r"```$", "", text).strip()
    data = json.loads(text)
    questions = data.get("questions", [])
    return _dedupe_questions([str(question).strip() for question in questions if str(question).strip()])


def _dedupe_questions(questions: list[str]) -> list[str]:
    cleaned: list[str] = []
    seen: set[str] = set()
    for question in questions:
        question = question.strip()
        if not question:
            continue
        if not question.endswith("?"):
            question += "?"
        key = question.lower()
        if key not in seen:
            cleaned.append(question)
            seen.add(key)
    return cleaned


def _dedupe_terms(terms: list[str], limit: int) -> list[str]:
    cleaned: list[str] = []
    seen: set[str] = set()
    for term in terms:
        term = " ".join(term.split())
        key = term.lower()
        if term and key not in seen:
            cleaned.append(term)
            seen.add(key)
        if len(cleaned) >= limit:
            break
    return cleaned


def _looks_like_heading(text: str) -> bool:
    if len(text) < 4 or len(text) > 90:
        return False
    words = text.split()
    if len(words) > 10:
        return False
    title_words = sum(1 for word in words if word[:1].isupper())
    return text.isupper() or title_words >= max(1, len(words) // 2)


def _shorten(text: str, limit: int) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3].rstrip() + "..."
