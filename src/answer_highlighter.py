"""Markdown highlighting for important answer facts."""

from __future__ import annotations

import re


FACT_PATTERN = re.compile(
    r"(?<![\w*])("
    r"\$?\d+(?:,\d{3})*(?:\.\d+)?\s*(?:%|percent|minutes?|mins?|hours?|hrs?|days?|weeks?|months?|years?|business days?)"
    r"|(?:within|after|before|by|no later than|at least|up to|maximum of|max(?:imum)?|minimum of|min(?:imum)?)\s+"
    r"\$?\d+(?:,\d{3})*(?:\.\d+)?\s*(?:%|percent|minutes?|mins?|hours?|hrs?|days?|weeks?|months?|years?|business days?)"
    r"|(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}(?:,\s*\d{4})?"
    r"|\d{1,2}/\d{1,2}/\d{2,4}"
    r")",
    flags=re.IGNORECASE,
)


def highlight_key_facts(answer: str) -> str:
    """Bold likely key facts without disturbing existing Markdown."""
    if not answer:
        return answer

    parts = re.split(r"(```.*?```|`[^`]*`|\*\*[^*]+\*\*)", answer, flags=re.DOTALL)
    highlighted: list[str] = []
    for part in parts:
        if not part:
            continue
        if part.startswith("```") or part.startswith("`") or (part.startswith("**") and part.endswith("**")):
            highlighted.append(part)
            continue
        highlighted.append(FACT_PATTERN.sub(_bold_match, part))
    return "".join(highlighted)


def _bold_match(match: re.Match[str]) -> str:
    value = match.group(1)
    if value.startswith("**") and value.endswith("**"):
        return value
    return f"**{value}**"
