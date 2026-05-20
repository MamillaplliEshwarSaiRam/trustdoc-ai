"""LLM-backed conflict analysis for retrieved evidence."""

from __future__ import annotations

import json
import re

from src.config import Settings
from src.context_compressor import compress_retrieved_context
from src.models import RetrievedChunk
from src.trust import detect_conflicts


def analyze_conflicts(
    question: str,
    retrieved: list[RetrievedChunk],
    settings: Settings,
    mode: str,
) -> list[str]:
    if not retrieved:
        return []
    if mode == "Gemini reasoning" and settings.google_api_key:
        try:
            compressed = compress_retrieved_context(question, retrieved, "Conflict Analysis")
            return _analyze_with_gemini(question, compressed, settings)
        except Exception:
            return detect_conflicts(retrieved)
    if mode == "OpenAI reasoning" and settings.openai_api_key:
        try:
            compressed = compress_retrieved_context(question, retrieved, "Conflict Analysis")
            return _analyze_with_openai(question, compressed, settings)
        except Exception:
            return detect_conflicts(retrieved)
    return detect_conflicts(retrieved)


def _analyze_with_gemini(question: str, retrieved: list[RetrievedChunk], settings: Settings) -> list[str]:
    from google import genai

    client = genai.Client(api_key=settings.google_api_key)
    response = client.models.generate_content(
        model=settings.gemini_reasoning_model,
        contents=_build_prompt(question, retrieved),
    )
    return _parse_conflict_json(response.text or "")


def _analyze_with_openai(question: str, retrieved: list[RetrievedChunk], settings: Settings) -> list[str]:
    from openai import OpenAI

    client = OpenAI(api_key=settings.openai_api_key)
    response = client.chat.completions.create(
        model=settings.openai_chat_model,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": "Return only valid JSON. Do not use markdown.",
            },
            {"role": "user", "content": _build_prompt(question, retrieved)},
        ],
    )
    return _parse_conflict_json(response.choices[0].message.content or "")


def _build_prompt(question: str, retrieved: list[RetrievedChunk]) -> str:
    evidence = "\n\n".join(
        f"[{number}] Citation: {item.chunk.citation}\nText: {item.chunk.text}"
        for number, item in enumerate(retrieved[:6], start=1)
    )
    return f"""
You are an evidence conflict analyst for a retrieval-augmented chatbot.

User question:
{question}

Retrieved evidence:
{evidence}

Identify only real or likely contradictions between retrieved evidence chunks.
Do not flag differences that are explainable by conditions, exceptions, scope, role, eligibility, dates, or process stage.
Use only the retrieved evidence. Do not invent external facts.

Return JSON in this exact shape:
{{
  "conflicts": [
    {{
      "claim_a": "short claim from one source",
      "source_a": "citation",
      "claim_b": "short conflicting claim from another source",
      "source_b": "citation",
      "why_conflict": "why these claims are difficult to reconcile",
      "severity": "low|medium|high"
    }}
  ],
  "no_conflict_reason": "short reason when conflicts is empty"
}}
""".strip()


def _parse_conflict_json(raw_text: str) -> list[str]:
    data = json.loads(_extract_json(raw_text))
    conflicts = data.get("conflicts", [])
    warnings: list[str] = []
    for conflict in conflicts:
        source_a = str(conflict.get("source_a", "source A")).strip()
        source_b = str(conflict.get("source_b", "source B")).strip()
        claim_a = str(conflict.get("claim_a", "")).strip()
        claim_b = str(conflict.get("claim_b", "")).strip()
        why = str(conflict.get("why_conflict", "")).strip()
        severity = str(conflict.get("severity", "medium")).strip().lower()
        if not why:
            continue
        warnings.append(
            f"{severity.title()} conflict: {source_a} says '{claim_a}', while {source_b} says '{claim_b}'. {why}"
        )
    return warnings


def _extract_json(raw_text: str) -> str:
    text = raw_text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text, flags=re.IGNORECASE).strip()
        text = re.sub(r"```$", "", text).strip()
    if text.startswith("{"):
        return text
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        raise ValueError("LLM did not return JSON.")
    return match.group(0)
