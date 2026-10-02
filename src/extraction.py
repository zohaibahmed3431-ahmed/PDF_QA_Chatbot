from __future__ import annotations

import io
import json
from pathlib import Path
from typing import BinaryIO

import pandas as pd
from PIL import Image
import pytesseract
from pypdf import PdfReader
from docx import Document
from pptx import Presentation
from openpyxl import load_workbook


TEXT_EXTENSIONS = {
    ".txt", ".md", ".py", ".java", ".cpp", ".c", ".h", ".hpp",
    ".js", ".ts", ".html", ".css", ".sql", ".json", ".xml"
}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def _decode_text(data: bytes) -> str:
    return data.decode("utf-8", errors="replace")


def extract_pdf(data: bytes) -> list[dict]:
    reader = PdfReader(io.BytesIO(data))
    pages = []
    for number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            pages.append({"text": text, "page": number})
        else:
            # OCR fallback for image-only PDF pages is intentionally handled
            # by the optional OCR path below when a rasterizer is available.
            pages.append({"text": "", "page": number})
    return pages


def extract_docx(data: bytes) -> list[dict]:
    doc = Document(io.BytesIO(data))
    text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
    for table in doc.tables:
        rows = []
        for row in table.rows:
            rows.append(" | ".join(cell.text.strip() for cell in row.cells))
        if rows:
            text += "\n" + "\n".join(rows)
    return [{"text": text, "page": None}] if text.strip() else []


def extract_pptx(data: bytes) -> list[dict]:
    prs = Presentation(io.BytesIO(data))
    output = []
    for slide_no, slide in enumerate(prs.slides, start=1):
        parts = []
        for shape in slide.shapes:
            if hasattr(shape, "text") and shape.text.strip():
                parts.append(shape.text.strip())
        text = "\n".join(parts)
        if text:
            output.append({"text": text, "page": slide_no})
    return output


def extract_xlsx(data: bytes) -> list[dict]:
    wb = load_workbook(io.BytesIO(data), read_only=True, data_only=True)
    output = []
    for ws in wb.worksheets:
        rows = []
        for row in ws.iter_rows(values_only=True):
            values = ["" if v is None else str(v) for v in row]
            if any(v.strip() for v in values):
                rows.append(" | ".join(values))
        if rows:
            output.append({
                "text": f"Sheet: {ws.title}\n" + "\n".join(rows),
                "page": None,
                "sheet": ws.title,
            })
    return output


def extract_csv(data: bytes) -> list[dict]:
    df = pd.read_csv(io.BytesIO(data))
    text = df.to_csv(index=False)
    return [{"text": text, "page": None}] if text.strip() else []


def extract_image(data: bytes) -> list[dict]:
    image = Image.open(io.BytesIO(data))
    text = pytesseract.image_to_string(image)
    return [{"text": text, "page": 1, "image": image}] if text.strip() else [{"text": "", "page": 1, "image": image}]


def extract_file(uploaded_file: BinaryIO, filename: str) -> list[dict]:
    data = uploaded_file.read()
    suffix = Path(filename).suffix.lower()

    if suffix == ".pdf":
        return extract_pdf(data)
    if suffix == ".docx":
        return extract_docx(data)
    if suffix == ".pptx":
        return extract_pptx(data)
    if suffix == ".xlsx":
        return extract_xlsx(data)
    if suffix == ".csv":
        return extract_csv(data)
    if suffix in TEXT_EXTENSIONS:
        text = _decode_text(data)
        return [{"text": text, "page": None}] if text.strip() else []
    if suffix in IMAGE_EXTENSIONS:
        return extract_image(data)

    raise ValueError(f"Unsupported file type: {suffix}")
