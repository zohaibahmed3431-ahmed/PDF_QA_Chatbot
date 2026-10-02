from __future__ import annotations

import re


def normalize_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunk_text(
    text: str,
    chunk_size: int = 850,
    overlap: int = 140,
) -> list[str]:
    text = normalize_text(text)
    if not text:
        return []

    paragraphs = re.split(r"\n\s*\n", text)
    chunks = []
    current = ""

    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        if not paragraph:
            continue

        candidate = f"{current}\n\n{paragraph}".strip()
        if len(candidate) <= chunk_size:
            current = candidate
            continue

        if current:
            chunks.append(current)

        # Preserve long paragraphs by slicing on character boundaries.
        if len(paragraph) > chunk_size:
            start = 0
            while start < len(paragraph):
                end = min(start + chunk_size, len(paragraph))
                piece = paragraph[start:end].strip()
                if piece:
                    chunks.append(piece)
                if end >= len(paragraph):
                    break
                start = max(0, end - overlap)
            current = ""
        else:
            tail = current[-overlap:] if overlap and current else ""
            current = f"{tail}\n\n{paragraph}".strip()

    if current:
        chunks.append(current)

    return chunks


def build_chunks(units: list[dict], source: str) -> list[dict]:
    output = []
    for unit in units:
        text = unit.get("text", "")
        for chunk_id, chunk in enumerate(chunk_text(text), start=1):
            output.append({
                "id": f"{source}:{unit.get('page')}:{chunk_id}",
                "source": source,
                "page": unit.get("page"),
                "sheet": unit.get("sheet"),
                "text": chunk,
            })
    return output
