import os
import streamlit as st
from dotenv import load_dotenv

from src.ingestion import ingest_uploaded_file
from src.retrieval import HybridRetriever
from src.llm import GeminiAssistant
from src.security import sanitize_question
from src.citations import format_sources

load_dotenv()

st.set_page_config(
    page_title="DocuSphere AI",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.block-container {padding-top: 1.2rem; max-width: 1400px;}
[data-testid="stSidebar"] {min-width: 290px; max-width: 330px;}
.source-card {
    padding: 0.7rem 0.9rem; border: 1px solid rgba(128,128,128,.25);
    border-radius: 10px; margin: .35rem 0;
}
.small-muted {opacity: .72; font-size: .86rem;}
</style>
""", unsafe_allow_html=True)

st.title("📚 DocuSphere AI")
st.caption("Intelligent Document Assistant • Multimodal ingestion • Hybrid RAG • Gemini")

SUPPORTED = [
    "pdf", "docx", "txt", "md", "pptx", "xlsx", "csv",
    "py", "java", "cpp", "c", "h", "hpp", "js", "ts",
    "html", "css", "sql", "json", "xml",
    "jpg", "jpeg", "png", "webp",
]

if "records" not in st.session_state:
    st.session_state.records = []
if "chat" not in st.session_state:
    st.session_state.chat = []
if "retriever" not in st.session_state:
    st.session_state.retriever = HybridRetriever()
if "assistant" not in st.session_state:
    st.session_state.assistant = GeminiAssistant()

with st.sidebar:
    st.header("📁 Documents")
    uploads = st.file_uploader(
        "Upload one or more files",
        type=SUPPORTED,
        accept_multiple_files=True,
        help="Upload documents, spreadsheets, images, or source-code files.",
    )

    if st.button("➕ Process uploads", use_container_width=True, disabled=not uploads):
        new_records = []
        progress = st.progress(0)
        status = st.empty()

        for i, uploaded in enumerate(uploads):
            status.write(f"Processing `{uploaded.name}`…")
            try:
                records = ingest_uploaded_file(uploaded)
                new_records.extend(records)
            except Exception as exc:
                st.error(f"{uploaded.name}: {exc}")
            progress.progress((i + 1) / len(uploads))

        if new_records:
            # Replace same-name files rather than duplicating them.
            names = {r["source"] for r in new_records}
            st.session_state.records = [
                r for r in st.session_state.records if r["source"] not in names
            ] + new_records
            st.session_state.retriever.build(st.session_state.records)
            st.success(f"Ready: {len(new_records)} content units.")
            st.rerun()

    if st.session_state.records:
        st.divider()
        st.subheader("Ready documents")
        counts = {}
        for r in st.session_state.records:
            counts[r["source"]] = counts.get(r["source"], 0) + 1
        for name, count in counts.items():
            st.write(f"📄 **{name}**  \n`{count} chunks`")

        if st.button("🗑️ Clear documents", use_container_width=True):
            st.session_state.records = []
            st.session_state.retriever.clear()
            st.session_state.chat = []
            st.rerun()

    st.divider()
    st.subheader("Pipeline")
    st.write("📤 Upload → Detect → Extract/OCR → Normalize → Chunk")
    st.write("🧩 Embed → Vector index → Hybrid retrieval → Rerank → Gemini")
    st.write("📌 Answer → Citation")

    st.divider()
    if st.button("🧹 Clear chat", use_container_width=True):
        st.session_state.chat = []
        st.rerun()

# Main chat
if not st.session_state.records:
    st.info("Upload documents from the left to start. You can also ask general AI questions.")
else:
    st.success(f"{len({r['source'] for r in st.session_state.records})} document(s) ready.")

for item in st.session_state.chat:
    with st.chat_message(item["role"]):
        st.markdown(item["content"])
        if item.get("sources"):
            st.markdown(format_sources(item["sources"]))

question = st.chat_input(
    "Ask about your files, ask for an explanation, or modify uploaded code…"
)

if question:
    clean_question = sanitize_question(question)
    if not clean_question:
        st.warning("Please enter a question.")
        st.stop()

    st.session_state.chat.append({"role": "user", "content": question})

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            results = st.session_state.retriever.search(clean_question, top_k=8)
            history = st.session_state.chat[-12:]
            answer = st.session_state.assistant.answer(
                question=clean_question,
                retrieved=results,
                all_records=st.session_state.records,
                conversation=history,
            )

        st.markdown(answer)
        st.markdown(format_sources(results))

    st.session_state.chat.append({
        "role": "assistant",
        "content": answer,
        "sources": results,
    })
