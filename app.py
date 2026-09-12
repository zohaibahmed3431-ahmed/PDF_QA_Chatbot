import streamlit as st
import numpy as np
import faiss
import re
import io

from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from google import genai

import fitz
import pytesseract
from PIL import Image


st.set_page_config(
    page_title="PDF Q&A Chatbot",
    page_icon="📚",
    layout="wide"
)

st.title("📚 PDF Q&A Chatbot")
st.write(
    "Upload text, scanned, or image-based PDFs and ask questions."
)


# ---------------------------------------------------------
# EMBEDDING MODEL
# ---------------------------------------------------------

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


embedding_model = load_embedding_model()


# ---------------------------------------------------------
# TEXT EXTRACTION + OCR
# ---------------------------------------------------------

def extract_page_text(file_bytes, page_number):
    """
    First try normal PDF text extraction.
    If little/no text is found, use OCR on the page image.
    """

    reader = PdfReader(io.BytesIO(file_bytes))

    text = ""

    if page_number < len(reader.pages):
        page = reader.pages[page_number]
        extracted = page.extract_text()

        if extracted:
            text = extracted.strip()

    # Normal text PDF
    if len(text) >= 30:
        return text, "text"

    # OCR fallback
    try:
        pdf_document = fitz.open(stream=file_bytes, filetype="pdf")

        page = pdf_document.load_page(page_number)

        pix = page.get_pixmap(
            matrix=fitz.Matrix(2, 2),
            alpha=False
        )

        image_bytes = pix.tobytes("png")

        image = Image.open(
            io.BytesIO(image_bytes)
        )

        ocr_text = pytesseract.image_to_string(
            image
        )

        pdf_document.close()

        if ocr_text and ocr_text.strip():
            return ocr_text.strip(), "ocr"

    except Exception as e:
        return "", f"ocr_error: {str(e)}"

    return "", "empty"


# ---------------------------------------------------------
# PROCESS PDFs
# ---------------------------------------------------------

def process_pdfs(uploaded_files):

    documents = []

    total_pages = 0
    ocr_pages = 0
    text_pages = 0

    progress = st.progress(0)

    for file_index, file in enumerate(uploaded_files):

        file_bytes = file.getvalue()

        reader = PdfReader(
            io.BytesIO(file_bytes)
        )

        page_count = len(reader.pages)
        total_pages += page_count

        for page_number in range(page_count):

            text, extraction_type = extract_page_text(
                file_bytes,
                page_number
            )

            if extraction_type == "text":
                text_pages += 1

            elif extraction_type == "ocr":
                ocr_pages += 1

            if text and text.strip():

                documents.append({
                    "text": text,
                    "source": file.name,
                    "page": page_number + 1,
                    "method": extraction_type
                })

            progress.progress(
                min(
                    1.0,
                    (
                        file_index * page_count
                        + page_number
                        + 1
                    )
                    / max(
                        1,
                        sum(
                            len(
                                PdfReader(
                                    io.BytesIO(
                                        f.getvalue()
                                    )
                                ).pages
                            )
                            for f in uploaded_files
                        )
                    )
                )
            )

    progress.empty()

    if not documents:
        return (
            None,
            None,
            "❌ PDF se readable text nahi mila."
        )

    # -----------------------------------------------------
    # CHUNKING
    # -----------------------------------------------------

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
                    "method": document["method"]
                })

    if not chunks:
        return (
            None,
            None,
            "❌ PDF ko chunks mein convert nahi kiya ja saka."
        )

    # -----------------------------------------------------
    # EMBEDDINGS
    # -----------------------------------------------------

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

    # -----------------------------------------------------
    # FAISS VECTOR DATABASE
    # -----------------------------------------------------

    vector_db = faiss.IndexFlatIP(
        embeddings.shape[1]
    )

    vector_db.add(embeddings)

    status = (
        f"✅ {len(uploaded_files)} PDF(s) processed\n\n"
        f"📄 Total pages: {total_pages}\n"
        f"📝 Text pages: {text_pages}\n"
        f"🔎 OCR pages: {ocr_pages}\n"
        f"🧩 Chunks: {len(chunks)}"
    )

    return vector_db, chunks, status


# ---------------------------------------------------------
# PAGE QUESTION DETECTION
# ---------------------------------------------------------

def is_page_question(question):

    words = re.findall(
        r"[a-zA-Z]+",
        question.lower()
    )

    page_words = {
        "page",
        "pages",
        "where",
        "discussed",
        "located",
        "appear",
        "appears",
        "find",
        "found",
        "kis",
        "kahan"
    }

    return any(
        word in page_words
        for word in words
    )


# ---------------------------------------------------------
# KEYWORD EXTRACTION
# ---------------------------------------------------------

def get_keywords(question):

    stop_words = {
        "what",
        "where",
        "when",
        "which",
        "who",
        "how",
        "are",
        "is",
        "the",
        "a",
        "an",
        "on",
        "in",
        "of",
        "to",
        "for",
        "and",
        "do",
        "does",
        "did",
        "this",
        "that",
        "these",
        "those",
        "page",
        "pages",
        "discussed",
        "located",
        "appear",
        "appears",
        "find",
        "found",
        "ka",
        "ki",
        "ke",
        "kis",
        "par",
        "batao",
        "hai",
        "hain",
        "kya",
        "kahan"
    }

    words = re.findall(
        r"[a-zA-Z0-9]+",
        question.lower()
    )

    keywords = []

    for word in words:

        if (
            len(word) >= 3
            and word not in stop_words
        ):
            keywords.append(word)

    return keywords


# ---------------------------------------------------------
# KEYWORD PAGE SEARCH
# ---------------------------------------------------------

def keyword_page_search(
    question,
    chunks
):

    keywords = get_keywords(
        question
    )

    if not keywords:
        return []

    page_scores = {}

    for chunk in chunks:

        text = chunk["text"].lower()

        score = 0

        for keyword in keywords:

            # Exact word
            pattern = (
                r"\b"
                + re.escape(keyword)
                + r"\b"
            )

            if re.search(
                pattern,
                text
            ):
                score += 3

            # Singular / plural
            if keyword.endswith("s"):

                singular = keyword[:-1]

                pattern = (
                    r"\b"
                    + re.escape(singular)
                    + r"\b"
                )

                if re.search(
                    pattern,
                    text
                ):
                    score += 2

            else:

                plural = keyword + "s"

                pattern = (
                    r"\b"
                    + re.escape(plural)
                    + r"\b"
                )

                if re.search(
                    pattern,
                    text
                ):
                    score += 2

        if score > 0:

            key = (
                chunk["source"],
                chunk["page"]
            )

            if key not in page_scores:
                page_scores[key] = 0

            page_scores[key] += score

    sorted_pages = sorted(
        page_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )

    return [
        page
        for page, score in sorted_pages[:20]
    ]


# ---------------------------------------------------------
# SEMANTIC SEARCH
# ---------------------------------------------------------

def semantic_search(
    question,
    vector_db,
    chunks,
    top_k=10
):

    question_embedding = embedding_model.encode(
        [question],
        normalize_embeddings=True,
        show_progress_bar=False
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

        results.append({
            "score": float(score),
            **chunks[index]
        })

    return results


# ---------------------------------------------------------
# ASK PDF
# ---------------------------------------------------------

def ask_pdf(
    question,
    vector_db,
    chunks
):

    # -----------------------------------------------------
    # PAGE-LOCATION QUESTIONS
    # -----------------------------------------------------

    if is_page_question(question):

        keyword_pages = keyword_page_search(
            question,
            chunks
        )

        # If keyword search found pages
        if keyword_pages:

            answer = (
                "### 📄 Relevant Pages\n\n"
            )

            for source, page in keyword_pages:

                answer += (
                    f"- **{source}** — "
                    f"Page **{page}**\n"
                )

            return answer

        # -------------------------------------------------
        # SEMANTIC FALLBACK
        # -------------------------------------------------

        semantic_results = semantic_search(
            question,
            vector_db,
            chunks,
            top_k=10
        )

        unique_pages = []
        seen = set()

        for result in semantic_results:

            key = (
                result["source"],
                result["page"]
            )

            if key not in seen:

                seen.add(key)
                unique_pages.append(key)

        if unique_pages:

            answer = (
                "### 📄 Relevant Pages\n\n"
            )

            for source, page in unique_pages:

                answer += (
                    f"- **{source}** — "
                    f"Page **{page}**\n"
                )

            return answer

    # -----------------------------------------------------
    # NORMAL QUESTION
    # -----------------------------------------------------

    results = semantic_search(
        question,
        vector_db,
        chunks,
        top_k=8
    )

    if not results:

        return (
            "❌ PDF mein relevant information nahi mili."
        )

    # -----------------------------------------------------
    # BUILD CONTEXT
    # -----------------------------------------------------

    context = ""

    for i, result in enumerate(
        results,
        start=1
    ):

        context += f"""
Source {i}

Document:
{result["source"]}

Page:
{result["page"]}

Content:
{result["text"]}

-------------------------
"""

    # -----------------------------------------------------
    # GEMINI
    # -----------------------------------------------------

    try:

        api_key = st.secrets[
            "GEMINI_API_KEY"
        ]

    except Exception:

        return (
            "❌ Gemini API key nahi mili. "
            "Streamlit Secrets mein "
            "`GEMINI_API_KEY` add karo."
        )

    client = genai.Client(
        api_key=api_key
    )

    prompt = f"""
You are a PDF Q&A assistant.

Answer the user's question ONLY from
the provided PDF context.

Do not use outside knowledge.

If the answer is not present in the
provided context, say:

"I could not find this information
in the uploaded PDFs."

Question:
{question}

PDF Context:
{context}

Give a clear and concise answer.

After the answer, provide the relevant
document and page numbers.
"""

    try:

        response = client.models.generate_content(
            model="gemini-3.6-flash",
            contents=prompt
        )

        answer = response.text

    except Exception as e:

        return (
            "❌ Gemini error:\n\n"
            f"{str(e)}"
        )

    # -----------------------------------------------------
    # SOURCES
    # -----------------------------------------------------

    sources = (
        "\n\n### 📌 Sources\n\n"
    )

    seen_sources = set()

    for result in results:

        key = (
            result["source"],
            result["page"]
        )

        if key not in seen_sources:

            seen_sources.add(key)

            sources += (
                f"- **{result['source']}** — "
                f"Page **{result['page']}**\n"
            )

    return answer + sources


# ---------------------------------------------------------
# SESSION STATE
# ---------------------------------------------------------

if "vector_db" not in st.session_state:
    st.session_state.vector_db = None

if "chunks" not in st.session_state:
    st.session_state.chunks = None


# ---------------------------------------------------------
# PDF UPLOAD
# ---------------------------------------------------------

uploaded_files = st.file_uploader(
    "📄 Upload PDF files",
    type=["pdf"],
    accept_multiple_files=True
)


# ---------------------------------------------------------
# PROCESS BUTTON
# ---------------------------------------------------------

if st.button(
    "⚙️ Process PDFs"
):

    if not uploaded_files:

        st.warning(
            "Pehle PDF upload karo."
        )

    else:

        with st.spinner(
            "PDFs process ho rahi hain..."
        ):

            vector_db, chunks, status = (
                process_pdfs(
                    uploaded_files
                )
            )

            st.session_state.vector_db = (
                vector_db
            )

            st.session_state.chunks = (
                chunks
            )

        if vector_db is not None:

            st.success(status)

        else:

            st.error(status)


# ---------------------------------------------------------
# QUESTION
# ---------------------------------------------------------

question = st.text_input(
    "❓ Ask a question",
    placeholder=(
        "e.g. Where are loops discussed?"
    )
)


# ---------------------------------------------------------
# ASK BUTTON
# ---------------------------------------------------------

if st.button(
    "🔍 Ask"
):

    if (
        st.session_state.vector_db
        is None
    ):

        st.warning(
            "Pehle PDFs process karo."
        )

    elif not question.strip():

        st.warning(
            "Question likho."
        )

    else:

        with st.spinner(
            "Finding answer..."
        ):

            answer = ask_pdf(
                question,
                st.session_state.vector_db,
                st.session_state.chunks
            )

        st.markdown(answer)
