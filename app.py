import streamlit as st
import io
import re
import numpy as np
import faiss
import fitz
import pytesseract

from PIL import Image, ImageFilter
from pypdf import PdfReader
from docx import Document
from pptx import Presentation
from openpyxl import load_workbook

from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from google import genai


# =========================================================
# APP CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Document Q&A Assistant",
    page_icon="📚",
    layout="wide"
)

st.title("📚 Document Q&A Assistant")
st.caption(
    "Upload documents, ask questions, and get answers with source locations."
)


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.header("⚙️ Settings")

    response_language = st.selectbox(
        "Response Language",
        [
            "English",
            "Urdu",
            "Roman Urdu"
        ]
    )

    st.divider()

    st.subheader("Supported Files")

    st.write(
        "PDF, DOCX, TXT, PPTX, XLSX, CSV, "
        "JPG, JPEG, PNG, WEBP"
    )

    st.divider()

    st.caption(
        "Answers are generated from your uploaded files."
    )


# =========================================================
# EMBEDDING MODEL
# =========================================================

@st.cache_resource
def load_embedding_model():

    return SentenceTransformer(
        "all-MiniLM-L6-v2"
    )


embedding_model = load_embedding_model()


# =========================================================
# HELPER: IMAGE QUALITY
# =========================================================

def check_image_quality(image):

    gray = image.convert("L")

    # Blur detection using edge/detail variance
    edges = gray.filter(
        ImageFilter.FIND_EDGES
    )

    pixels = np.asarray(
        edges,
        dtype=np.float32
    )

    sharpness = float(
        pixels.var()
    )

    width, height = image.size

    if width < 500 or height < 500:

        return False, "low_resolution"

    # Very low edge variance usually means
    # extremely blurry/blank content.
    if sharpness < 20:

        return False, "blurry"

    return True, "ok"


# =========================================================
# OCR
# =========================================================

def perform_ocr(image):

    try:

        text = pytesseract.image_to_string(
            image,
            lang="eng+urd"
        )

        return text.strip()

    except Exception:

        # Fallback to English OCR
        try:

            text = pytesseract.image_to_string(
                image,
                lang="eng"
            )

            return text.strip()

        except Exception:

            return ""


# =========================================================
# PDF PROCESSING
# =========================================================

def process_pdf(
    file_bytes,
    file_name
):

    documents = []

    pdf = fitz.open(
        stream=file_bytes,
        filetype="pdf"
    )

    total_pages = len(pdf)

    for page_index in range(total_pages):

        page = pdf.load_page(
            page_index
        )

        page_number = page_index + 1

        text = page.get_text(
            "text"
        ).strip()

        # -------------------------------------------------
        # NORMAL TEXT PDF
        # -------------------------------------------------

        if len(text) >= 30:

            documents.append({
                "text": text,
                "source": file_name,
                "page": page_number,
                "location": (
                    f"{file_name} — "
                    f"Page {page_number}"
                ),
                "method": "text"
            })

            continue

        # -------------------------------------------------
        # SCANNED / IMAGE PDF
        # -------------------------------------------------

        pix = page.get_pixmap(
            matrix=fitz.Matrix(
                2,
                2
            ),
            alpha=False
        )

        image = Image.open(
            io.BytesIO(
                pix.tobytes("png")
            )
        )

        quality_ok, quality_status = (
            check_image_quality(
                image
            )
        )

        if not quality_ok:

            documents.append({
                "text": "",
                "source": file_name,
                "page": page_number,
                "location": (
                    f"{file_name} — "
                    f"Page {page_number}"
                ),
                "method": (
                    f"unreadable:{quality_status}"
                )
            })

            continue

        ocr_text = perform_ocr(
            image
        )

        if len(ocr_text.strip()) >= 10:

            documents.append({
                "text": ocr_text,
                "source": file_name,
                "page": page_number,
                "location": (
                    f"{file_name} — "
                    f"Page {page_number}"
                ),
                "method": "ocr"
            })

        else:

            documents.append({
                "text": "",
                "source": file_name,
                "page": page_number,
                "location": (
                    f"{file_name} — "
                    f"Page {page_number}"
                ),
                "method": "unreadable"
            })

    pdf.close()

    return documents


# =========================================================
# DOCX PROCESSING
# =========================================================

def process_docx(
    file_bytes,
    file_name
):

    documents = []

    document = Document(
        io.BytesIO(file_bytes)
    )

    paragraph_number = 0

    for paragraph in document.paragraphs:

        text = paragraph.text.strip()

        if text:

            paragraph_number += 1

            documents.append({
                "text": text,
                "source": file_name,
                "page": None,
                "location": (
                    f"{file_name} — "
                    f"Paragraph {paragraph_number}"
                ),
                "method": "text"
            })

    # Tables
    for table_index, table in enumerate(
        document.tables,
        start=1
    ):

        for row_index, row in enumerate(
            table.rows,
            start=1
        ):

            row_text = " | ".join(
                cell.text.strip()
                for cell in row.cells
                if cell.text.strip()
            )

            if row_text:

                documents.append({
                    "text": row_text,
                    "source": file_name,
                    "page": None,
                    "location": (
                        f"{file_name} — "
                        f"Table {table_index}, "
                        f"Row {row_index}"
                    ),
                    "method": "table"
                })

    return documents


# =========================================================
# TXT / CSV PROCESSING
# =========================================================

def process_text_file(
    file_bytes,
    file_name
):

    text = file_bytes.decode(
        "utf-8",
        errors="ignore"
    )

    if not text.strip():

        return []

    return [{
        "text": text,
        "source": file_name,
        "page": None,
        "location": file_name,
        "method": "text"
    }]


# =========================================================
# PPTX PROCESSING
# =========================================================

def process_pptx(
    file_bytes,
    file_name
):

    documents = []

    presentation = Presentation(
        io.BytesIO(file_bytes)
    )

    for slide_number, slide in enumerate(
        presentation.slides,
        start=1
    ):

        texts = []

        for shape in slide.shapes:

            if hasattr(
                shape,
                "text"
            ):

                value = shape.text.strip()

                if value:

                    texts.append(
                        value
                    )

        slide_text = "\n".join(
            texts
        )

        if slide_text.strip():

            documents.append({
                "text": slide_text,
                "source": file_name,
                "page": slide_number,
                "location": (
                    f"{file_name} — "
                    f"Slide {slide_number}"
                ),
                "method": "text"
            })

    return documents


# =========================================================
# XLSX PROCESSING
# =========================================================

def process_xlsx(
    file_bytes,
    file_name
):

    documents = []

    workbook = load_workbook(
        io.BytesIO(file_bytes),
        data_only=True
    )

    for sheet in workbook.worksheets:

        for row_number, row in enumerate(
            sheet.iter_rows(
                values_only=True
            ),
            start=1
        ):

            values = []

            for value in row:

                if value is not None:

                    values.append(
                        str(value)
                    )

            row_text = " | ".join(
                values
            )

            if row_text.strip():

                documents.append({
                    "text": row_text,
                    "source": file_name,
                    "page": None,
                    "location": (
                        f"{file_name} — "
                        f"Sheet '{sheet.title}', "
                        f"Row {row_number}"
                    ),
                    "method": "spreadsheet"
                })

    return documents


# =========================================================
# IMAGE PROCESSING
# =========================================================

def process_image(
    file_bytes,
    file_name
):

    image = Image.open(
        io.BytesIO(file_bytes)
    )

    quality_ok, quality_status = (
        check_image_quality(
            image
        )
    )

    if not quality_ok:

        return [{
            "text": "",
            "source": file_name,
            "page": None,
            "location": file_name,
            "method": (
                f"unreadable:{quality_status}"
            )
        }]

    text = perform_ocr(
        image
    )

    if not text.strip():

        return [{
            "text": "",
            "source": file_name,
            "page": None,
            "location": file_name,
            "method": "unreadable"
        }]

    return [{
        "text": text,
        "source": file_name,
        "page": None,
        "location": file_name,
        "method": "ocr"
    }]


# =========================================================
# UNIVERSAL FILE PROCESSOR
# =========================================================

def extract_documents(
    uploaded_files
):

    documents = []

    unreadable_files = []

    for uploaded_file in uploaded_files:

        file_name = uploaded_file.name

        file_bytes = uploaded_file.getvalue()

        extension = (
            file_name
            .lower()
            .split(".")[-1]
        )

        try:

            if extension == "pdf":

                new_documents = process_pdf(
                    file_bytes,
                    file_name
                )

            elif extension == "docx":

                new_documents = process_docx(
                    file_bytes,
                    file_name
                )

            elif extension in {
                "txt",
                "csv"
            }:

                new_documents = process_text_file(
                    file_bytes,
                    file_name
                )

            elif extension == "pptx":

                new_documents = process_pptx(
                    file_bytes,
                    file_name
                )

            elif extension == "xlsx":

                new_documents = process_xlsx(
                    file_bytes,
                    file_name
                )

            elif extension in {
                "jpg",
                "jpeg",
                "png",
                "webp"
            }:

                new_documents = process_image(
                    file_bytes,
                    file_name
                )

            else:

                new_documents = []

            for document in new_documents:

                if document["text"].strip():

                    documents.append(
                        document
                    )

                else:

                    unreadable_files.append(
                        document
                    )

        except Exception as error:

            unreadable_files.append({
                "source": file_name,
                "page": None,
                "location": file_name,
                "method": "error",
                "error": str(error)
            })

    return documents, unreadable_files


# =========================================================
# CHUNKING + VECTOR DATABASE
# =========================================================

def build_vector_database(
    documents
):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=100
    )

    chunks = []

    for document in documents:

        text_chunks = splitter.split_text(
            document["text"]
        )

        for chunk in text_chunks:

            if chunk.strip():

                chunks.append({
                    "text": chunk,
                    "source": document["source"],
                    "page": document["page"],
                    "location": document["location"],
                    "method": document["method"]
                })

    if not chunks:

        return None, []

    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    embeddings = embedding_model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=False
    )

    embeddings = np.asarray(
        embeddings,
        dtype="float32"
    )

    vector_db = faiss.IndexFlatIP(
        embeddings.shape[1]
    )

    vector_db.add(
        embeddings
    )

    return vector_db, chunks


# =========================================================
# SEMANTIC SEARCH
# =========================================================

def semantic_search(
    question,
    vector_db,
    chunks,
    top_k=8
):

    question_embedding = (
        embedding_model.encode(
            [question],
            normalize_embeddings=True,
            show_progress_bar=False
        )
    )

    question_embedding = np.asarray(
        question_embedding,
        dtype="float32"
    )

    k = min(
        top_k,
        len(chunks)
    )

    scores, indices = vector_db.search(
        question_embedding,
        k
    )

    results = []

    for score, index in zip(
        scores[0],
        indices[0]
    ):

        if index < 0:
            continue

        result = dict(
            chunks[index]
        )

        result["score"] = float(
            score
        )

        results.append(
            result
        )

    return results


# =========================================================
# LANGUAGE INSTRUCTIONS
# =========================================================

def get_language_instruction(
    language
):

    if language == "Urdu":

        return """
Answer entirely in Urdu script.
Do not answer in English or Roman Urdu.
Keep technical terms clear where necessary.
"""

    if language == "Roman Urdu":

        return """
Answer entirely in Roman Urdu.
Do not use Urdu script.
Keep technical terms clear where necessary.
"""

    return """
Answer entirely in English.
"""


# =========================================================
# ASK DOCUMENTS
# =========================================================

def ask_documents(
    question,
    vector_db,
    chunks,
    language
):

    results = semantic_search(
        question,
        vector_db,
        chunks,
        top_k=8
    )

    if not results:

        return (
            "No relevant information could "
            "be found in the uploaded files."
        )

    context = ""

    for index, result in enumerate(
        results,
        start=1
    ):

        context += f"""
SOURCE {index}

Location:
{result["location"]}

Content:
{result["text"]}

--------------------------------
"""

    language_instruction = (
        get_language_instruction(
            language
        )
    )

    prompt = f"""
You are a professional document Q&A assistant.

Your job is to answer ONLY from the
provided document context.

IMPORTANT RULES:

1. Do not invent information.
2. Do not use outside knowledge.
3. If the answer is not supported by
   the uploaded documents, clearly say
   that it could not be found.
4. Explain the answer clearly.
5. Use the user's selected response language.
6. Mention the relevant source locations.
7. If the user asks where a topic is
   discussed, identify the most relevant
   pages/slides/sections from the context.

Selected response language:
{language}

{language_instruction}

Question:
{question}

DOCUMENT CONTEXT:
{context}
"""

    try:

        api_key = st.secrets[
            "GEMINI_API_KEY"
        ]

    except Exception:

        return (
            "Gemini API key is not configured. "
            "Please add GEMINI_API_KEY in "
            "Streamlit Secrets."
        )

    try:

        client = genai.Client(
            api_key=api_key
        )

        response = client.models.generate_content(
            model="gemini-3.6-flash",
            contents=prompt
        )

        answer = response.text

    except Exception as error:

        return (
            "An error occurred while generating "
            "the answer.\n\n"
            f"Details: {str(error)}"
        )

    # -----------------------------------------------------
    # SOURCES
    # -----------------------------------------------------

    source_text = (
        "\n\n### 📌 Sources\n\n"
    )

    seen = set()

    for result in results:

        key = result["location"]

        if key in seen:
            continue

        seen.add(key)

        source_text += (
            f"- **{result['location']}**\n"
        )

    return answer + source_text


# =========================================================
# SESSION STATE
# =========================================================

if "vector_db" not in st.session_state:

    st.session_state.vector_db = None


if "chunks" not in st.session_state:

    st.session_state.chunks = []


if "unreadable_files" not in st.session_state:

    st.session_state.unreadable_files = []


# =========================================================
# FILE UPLOAD
# =========================================================

uploaded_files = st.file_uploader(
    "📁 Upload your files",
    type=[
        "pdf",
        "docx",
        "txt",
        "pptx",
        "xlsx",
        "csv",
        "jpg",
        "jpeg",
        "png",
        "webp"
    ],
    accept_multiple_files=True
)


# =========================================================
# PROCESS FILES
# =========================================================

if st.button(
    "⚙️ Process Files",
    use_container_width=True
):

    if not uploaded_files:

        st.warning(
            "Please upload at least one file."
        )

    else:

        with st.spinner(
            "Processing files..."
        ):

            documents, unreadable = (
                extract_documents(
                    uploaded_files
                )
            )

            vector_db, chunks = (
                build_vector_database(
                    documents
                )
            )

            st.session_state.vector_db = (
                vector_db
            )

            st.session_state.chunks = (
                chunks
            )

            st.session_state.unreadable_files = (
                unreadable
            )

        if vector_db is not None:

            st.success(
                f"Processed {len(uploaded_files)} "
                f"file(s) successfully. "
                f"{len(chunks)} searchable chunks created."
            )

        else:

            st.error(
                "No readable content could be "
                "extracted from the uploaded files."
            )


# =========================================================
# QUALITY WARNINGS
# =========================================================

if st.session_state.unreadable_files:

    st.warning(
        "Some files or pages could not be "
        "read reliably."
    )

    with st.expander(
        "View unreadable pages/files"
    ):

        for item in (
            st.session_state.unreadable_files
        ):

            location = item.get(
                "location",
                item.get(
                    "source",
                    "Unknown file"
                )
            )

            method = item.get(
                "method",
                "unreadable"
            )

            if "blurry" in method:

                st.error(
                    f"⚠️ {location}: "
                    "This page appears blurry. "
                    "We can't reliably read its content."
                )

            elif "low_resolution" in method:

                st.error(
                    f"⚠️ {location}: "
                    "This page has very low resolution. "
                    "We can't reliably read its content."
                )

            else:

                st.error(
                    f"⚠️ {location}: "
                    "We couldn't reliably extract "
                    "readable content."
                )


# =========================================================
# QUESTION AREA
# =========================================================

st.divider()

question = st.text_area(
    "💬 Ask a question",
    placeholder=(
        "Example: Where are loops discussed?"
    ),
    height=100
)


if st.button(
    "🔍 Ask",
    use_container_width=True
):

    if st.session_state.vector_db is None:

        st.warning(
            "Please process your files first."
        )

    elif not question.strip():

        st.warning(
            "Please enter a question."
        )

    else:

        with st.spinner(
            "Searching documents and generating answer..."
        ):

            answer = ask_documents(
                question,
                st.session_state.vector_db,
                st.session_state.chunks,
                response_language
            )

        st.markdown(answer)
