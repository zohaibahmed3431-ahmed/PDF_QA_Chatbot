from __future__ import annotations

import re


INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?previous\s+instructions",
    r"ignore\s+(the\s+)?system\s+prompt",
    r"reveal\s+(your|the)\s+(system|developer)\s+prompt",
    r"show\s+me\s+your\s+hidden\s+instructions",
    r"disregard\s+the\s+instructions",
]


def sanitize_question(question: str) -> str:
    question = (question or "").strip()
    if not question:
        return ""

    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, question, flags=re.I):
            return (
                "Please ask a normal question about the uploaded documents or "
                "a general concept. I cannot expose hidden system instructions."
            )
    return question
