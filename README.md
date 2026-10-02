# DocuSphere AI

**Intelligent Document Assistant** built with Streamlit and the Google GenAI SDK.

DocuSphere AI combines document extraction, OCR, embeddings, hybrid retrieval,
reranking, Gemini generation, citations, conversation context, and code-aware
assistance.

## Architecture

```text
Upload
  ↓
File type detection
  ↓
Extract / OCR
  ↓
Normalize
  ↓
Chunk
  ↓
Embedding
  ↓
Vector index
  ↓
READY

Question
  ↓
Query understanding
  ↓
Hybrid retrieval
  ↓
Reranking
  ↓
Gemini
  ↓
Answer + source locations
```

## Supported files

PDF, scanned/image-based content, DOCX, TXT, Markdown, PPTX, XLSX, CSV,
JPG/JPEG/PNG/WEBP, Python, Java, C/C++, JavaScript/TypeScript, HTML, CSS,
SQL, JSON, and XML.

## Knowledge model

DocuSphere distinguishes:

- **Document facts** — information actually present in uploaded content.
- **General AI knowledge** — explanations of general concepts.
- **Inference/reasoning** — conclusions derived from evidence, clearly marked
  as inference when they are not explicitly confirmed.

For source-code files, the assistant can use fuller file context when the user
asks for explanations, transformations, debugging, or modifications.

## Local setup

```bash
pip install -r requirements.txt
```

Set:

```text
GEMINI_API_KEY=your_key
```

Then:

```bash
streamlit run app.py
```

## Streamlit Cloud

Add `GEMINI_API_KEY` under the app's Secrets settings.

Do not commit a real API key to GitHub.

## Gemini models

The project uses Google's current GenAI Python SDK. The default generation
model is `gemini-3.8-flash`, with fallbacks. The default embedding model is
`gemini-embedding-2`, with a text-embedding fallback.

## Project status

This repository is the clean professional rebuild of DocuSphere AI. Evaluation
and experimental benchmarking are intentionally kept out of the main product
surface until the core assistant is stable.
