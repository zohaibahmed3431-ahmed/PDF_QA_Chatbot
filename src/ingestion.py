from __future__ import annotations

from .chunking import build_chunks
from .extraction import extract_file


def ingest_uploaded_file(uploaded_file) -> list[dict]:
    units = extract_file(uploaded_file, uploaded_file.name)
    return build_chunks(units, uploaded_file.name)
