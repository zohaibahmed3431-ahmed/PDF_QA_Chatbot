import os
import gradio as gr
import numpy as np
import faiss

from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from google import genai


# Gemini API
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

# Embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

chunks = []
vector_db = None


def process_pdfs(files):
    global chunks, vector_db

    documents = []

    for file in files:
        reader = PdfReader(file.name)

        for page_number, page in enumerate(reader.pages, start=1):
            text = page.extract_text()

            if text and text.strip():
                documents.append({
                    "text": text,
                    "source": os.path.basename(file.name),
                    "page": page_number
                })

    if not documents:
        return "❌ PDF se text nahi mila."

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

    embeddings = np.array(embeddings, dtype="float32")

    vector_db = faiss.IndexFlatIP(embeddings.shape[1])
    vector_db.add(embeddings)

    return (
        f"✅ {len(files)} PDFs processed successfully\n"
        f"📄 Pages: {len(documents)}\n"
        f"🧩 Chunks: {len(chunks)}"
    )


def ask_pdf(question):

    if vector_db is None:
        return "Pehle PDFs upload karke Process PDFs button press karo."

    if not question.strip():
        return "Question likho."

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

    sources = "\n\n### Sources\n"

    for result in results:
        sources += f"- {result['source']} | Page {result['page']}\n"

    return response.text + sources


with gr.Blocks(title="PDF Q&A Chatbot") as app:

    gr.Markdown(
        "# 📚 PDF Q&A Chatbot\n"
        "Upload your PDFs and ask questions about them."
    )

    pdfs = gr.File(
        label="Upload PDF files",
        file_count="multiple",
        file_types=[".pdf"]
    )

    process_button = gr.Button("Process PDFs")

    status = gr.Textbox(
        label="Status",
        interactive=False
    )

    process_button.click(
        process_pdfs,
        inputs=pdfs,
        outputs=status
    )

    question = gr.Textbox(
        label="Ask a question",
        placeholder="e.g. Where are loops discussed?"
    )

    ask_button = gr.Button("Ask")

    answer = gr.Markdown()

    ask_button.click(
        ask_pdf,
        inputs=question,
        outputs=answer
    )


app.launch()