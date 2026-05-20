"""Trust scoring, knowledge gaps, and conflict signals."""

from __future__ import annotations

import re

from src.models import RetrievedChunk, TrustReport


NEGATION_PATTERNS = [
    ("allowed", "not allowed"),
    ("required", "not required"),
    ("eligible", "not eligible"),
    ("refundable", "non refundable"),
    ("refund", "no refund"),
    ("must", "optional"),
    ("may", "may not"),
]


def evaluate_trust(
    question: str,
    retrieved: list[RetrievedChunk],
    answer: str,
    conflict_warnings: list[str] | None = None,
) -> TrustReport:
    reasons: list[str] = []
    gaps: list[str] = []
    conflict_warnings = conflict_warnings if conflict_warnings is not None else detect_conflicts(retrieved)

    if not retrieved:
        return TrustReport(
            score=0,
            label="No support",
            reasons=["No relevant document chunks were retrieved."],
            gaps=["Upload more relevant documents or ask a narrower question."],
            conflict_warnings=[],
            should_refuse=True,
        )

    top_score = retrieved[0].score
    average_score = sum(item.score for item in retrieved[:3]) / min(len(retrieved), 3)
    source_count = len({item.chunk.source for item in retrieved})
    question_terms = _important_terms(question)
    direct_overlap = _keyword_overlap(question_terms, " ".join(item.chunk.text for item in retrieved[:4]))

    top_signal = min(top_score / 0.35, 1.0)
    average_signal = min(average_score / 0.22, 1.0)
    score = int(round(15 + (top_signal * 35) + (average_signal * 20) + (direct_overlap * 25)))
    if len(retrieved) >= 3:
        score += 6
        reasons.append("Multiple supporting chunks were found.")
    if source_count > 1:
        score += 4
        reasons.append("Evidence comes from more than one document.")
    if conflict_warnings:
        score -= min(25, 10 * len(conflict_warnings))
        reasons.append("Potentially conflicting evidence was detected.")

    if top_score < 0.08:
        gaps.append("The closest source match is weak.")
    if question_terms and direct_overlap < 0.18:
        gaps.append("The question terms have limited overlap with the retrieved evidence.")
    if "not found" in answer.lower() or "could not find" in answer.lower():
        score = min(score, 35)
        gaps.append("The answer says the documents do not fully contain the requested detail.")

    score = max(0, min(100, score))
    should_refuse = score < 35 or top_score < 0.05
    label = "High trust" if score >= 75 else "Medium trust" if score >= 50 else "Low trust"
    if should_refuse:
        label = "Insufficient support"

    if not reasons:
        reasons.append("Score is based on retrieval relevance and evidence coverage.")

    return TrustReport(
        score=score,
        label=label,
        reasons=reasons,
        gaps=sorted(set(gaps)),
        conflict_warnings=conflict_warnings,
        should_refuse=should_refuse,
    )


def detect_conflicts(retrieved: list[RetrievedChunk]) -> list[str]:
    warnings: list[str] = []
    by_source = {item.chunk.citation: _normalize(item.chunk.text) for item in retrieved[:6]}

    for positive, negative in NEGATION_PATTERNS:
        positive_locations = [
            citation
            for citation, text in by_source.items()
            if positive in text and negative not in text
        ]
        negative_locations = [citation for citation, text in by_source.items() if negative in text]
        if positive_locations and negative_locations:
            locations = (positive_locations[:2] + negative_locations[:2])[:3]
            warnings.append(f"Possible disagreement around '{positive}' language in: {', '.join(locations)}")
    return sorted(set(warnings))


def _keyword_overlap(question_terms: list[str], evidence: str) -> float:
    question_terms = set(question_terms)
    if not question_terms:
        return 1.0
    evidence_terms = set(_important_terms(evidence))
    return len(question_terms & evidence_terms) / len(question_terms)


def _important_terms(text: str) -> list[str]:
    words = re.findall(r"[a-zA-Z0-9]+", text.lower())
    stop = {
        "what",
        "when",
        "where",
        "which",
        "does",
        "about",
        "from",
        "with",
        "that",
        "this",
        "there",
        "their",
        "would",
        "could",
        "should",
        "have",
        "been",
        "into",
        "your",
        "given",
        "document",
        "documents",
        "uploaded",
    }
    return [word for word in words if len(word) > 3 and word not in stop]


def _normalize(text: str) -> str:
    return text.lower().replace("-", " ")
