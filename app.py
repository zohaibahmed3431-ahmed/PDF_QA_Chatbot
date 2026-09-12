import streamlit as st
import os
import numpy as np
import faiss

from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from google import genai


st.set_page_config(
    page_title="PDF Q&A Chatbot",
    page_icon="📚"
)

st.title("📚 PDF Q&A Chatbot")
st.write("Upload your PDFs and ask questions about them.")


@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


embedding_model = load_embedding_model()


def process_pdfs(uploaded_files):

    documents = []

    for file in uploaded_files:
        reader = PdfReader(file)

        for page_number, page in enumerate(reader.pages, start=1):
            text = page.extract_text()

            if text and text.strip():
                documents.append({
                    "text": text,
                    "source": file.name,
                    "page": page_number
                })

    if not documents:
        return None, None, "❌ PDF se text nahi mila."

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    chunks = []

    for doc in documents:
        text_chunks = splitter.split_text(doc["text"])

        for chunk in text_chunks:
            chunks.append({
                "text": chunk,
                "source": doc["source"],
                "page": doc["page"]
            })

    texts = [chunk["text"] for chunk in chunks]

    embeddings = embedding_model.encode(
        texts,
        normalize_embeddings=True
    )

    embeddings = np.array(
        embeddings,
        dtype="float32"
    )

    vector_db = faiss.IndexFlatIP(
        embeddings.shape[1]
    )

    vector_db.add(embeddings)

    return vector_db, chunks, (
        f"✅ {len(uploaded_files)} PDFs processed\n"
        f"📄 Pages: {len(documents)}\n"
        f"🧩 Chunks: {len(chunks)}"
    )


def ask_pdf(question, vector_db, chunks):

    question_embedding = embedding_model.encode(
        [question],
        normalize_embeddings=True
    )

    question_embedding = np.array(
        question_embedding,
        dtype="float32"
    )

    scores, indices = vector_db.search(
        question_embedding,
        min(3, len(chunks))
    )

    results = []

    for index in indices[0]:
        results.append(chunks[index])

    context = ""

    for i, result in enumerate(results, start=1):
        context += f"""
Source {i}
Document: {result["source"]}
Page: {result["page"]}

Content:
{result["text"]}

-------------------------
"""

    api_key = st.secrets["GEMINI_API_KEY"]

    client = genai.Client(
        api_key=api_key
    )

    prompt = f"""
You are a PDF Q&A assistant.

Answer the user's question ONLY using the provided PDF context.

If the answer is not present in the context, say:

"I could not find this information in the uploaded PDFs."

Question:
{question}

PDF Context:
{context}

Give a clear and concise answer.
"""

    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=prompt
    )

    answer = response.text

    sources = "\n\n### 📌 Sources\n"

    for result in results:
        sources += (
            f"- {result['source']} | "
            f"Page {result['page']}\n"
        )

    return answer + sources


# Session state
if "vector_db" not in st.session_state:
    st.session_state.vector_db = None

if "chunks" not in st.session_state:
    st.session_state.chunks = None


uploaded_files = st.file_uploader(
    "📄 Upload PDF files",
    type=["pdf"],
    accept_multiple_files=True
)


if st.button("⚙️ Process PDFs"):

    if not uploaded_files:
        st.warning("Pehle PDF upload karo.")

    else:

        with st.spinner("Processing PDFs..."):

            vector_db, chunks, status = process_pdfs(
                uploaded_files
            )

            st.session_state.vector_db = vector_db
            st.session_state.chunks = chunks

        st.success(status)


question = st.text_input(
    "❓ Ask a question",
    placeholder="e.g. Where are loops discussed?"
)


if st.button("🔍 Ask"):

    if st.session_state.vector_db is None:
        st.warning("Pehle PDFs process karo.")

    elif not question.strip():
        st.warning("Question likho.")

    else:

        with st.spinner("Finding answer..."):

            answer = ask_pdf(
                question,
                st.session_state.vector_db,
                st.session_state.chunks
            )

        st.markdown(answer)
