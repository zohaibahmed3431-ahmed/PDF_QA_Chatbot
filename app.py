import io
import re
import time
import numpy as np
import faiss
import fitz
import pytesseract
import streamlit as st
from PIL import Image, ImageFilter
from docx import Document
from pptx import Presentation
from openpyxl import load_workbook
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from google import genai

st.set_page_config(page_title='Document Q&A Assistant', page_icon='📚', layout='wide')
st.title('📚 Document Q&A Assistant')
st.caption('Upload documents, ask questions, and get answers with source locations.')

with st.sidebar:
    st.header('⚙️ Settings')
    response_language = st.selectbox('Response Language', ['English', 'Urdu', 'Roman Urdu'])
    st.divider()
    st.subheader('Supported Files')
    st.write('PDF, DOCX, TXT, PPTX, XLSX, CSV, JPG, JPEG, PNG, WEBP')
    st.divider()
    st.caption('The assistant answers from the uploaded files only.')

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

embedding_model = load_embedding_model()


def check_image_quality(image):
    gray = image.convert('L')
    width, height = image.size
    if width < 500 or height < 500:
        return False, 'low_resolution'
    edges = gray.filter(ImageFilter.FIND_EDGES)
    if float(np.asarray(edges, dtype=np.float32).var()) < 20:
        return False, 'blurry'
    return True, 'ok'


def perform_ocr(image):
    for lang in ('eng+urd', 'eng'):
        try:
            text = pytesseract.image_to_string(image, lang=lang).strip()
            if text:
                return text
        except Exception:
            pass
    return ''


def process_pdf(data, name):
    docs = []
    pdf = fitz.open(stream=data, filetype='pdf')
    for i in range(len(pdf)):
        page = pdf.load_page(i)
        number = i + 1
        text = page.get_text('text').strip()
        if len(text) >= 30:
            docs.append({'text': text, 'source': name, 'page': number,
                         'location': f'{name} — Page {number}', 'method': 'text'})
            continue
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
        image = Image.open(io.BytesIO(pix.tobytes('png')))
        ok, status = check_image_quality(image)
        if not ok:
            docs.append({'text': '', 'source': name, 'page': number,
                         'location': f'{name} — Page {number}', 'method': f'unreadable:{status}'})
            continue
        text = perform_ocr(image)
        docs.append({'text': text if len(text) >= 10 else '', 'source': name,
                     'page': number, 'location': f'{name} — Page {number}',
                     'method': 'ocr' if len(text) >= 10 else 'unreadable'})
    pdf.close()
    return docs


def process_docx(data, name):
    docs = []
    doc = Document(io.BytesIO(data))
    pno = 0
    for p in doc.paragraphs:
        text = p.text.strip()
        if text:
            pno += 1
            docs.append({'text': text, 'source': name, 'page': None,
                         'location': f'{name} — Paragraph {pno}', 'method': 'text'})
    for ti, table in enumerate(doc.tables, 1):
        for ri, row in enumerate(table.rows, 1):
            text = ' | '.join(c.text.strip() for c in row.cells if c.text.strip())
            if text:
                docs.append({'text': text, 'source': name, 'page': None,
                             'location': f'{name} — Table {ti}, Row {ri}', 'method': 'table'})
    return docs


def process_text(data, name):
    text = data.decode('utf-8', errors='ignore').strip()
    return [{'text': text, 'source': name, 'page': None, 'location': name, 'method': 'text'}] if text else []


def process_pptx(data, name):
    docs = []
    pres = Presentation(io.BytesIO(data))
    for sn, slide in enumerate(pres.slides, 1):
        texts = [shape.text.strip() for shape in slide.shapes if hasattr(shape, 'text') and shape.text.strip()]
        text = '\n'.join(texts)
        if text:
            docs.append({'text': text, 'source': name, 'page': sn,
                         'location': f'{name} — Slide {sn}', 'method': 'text'})
    return docs


def process_xlsx(data, name):
    docs = []
    wb = load_workbook(io.BytesIO(data), data_only=True)
    for sheet in wb.worksheets:
        for rn, row in enumerate(sheet.iter_rows(values_only=True), 1):
            text = ' | '.join(str(v) for v in row if v is not None)
            if text.strip():
                docs.append({'text': text, 'source': name, 'page': None,
                             'location': f"{name} — Sheet '{sheet.title}', Row {rn}", 'method': 'spreadsheet'})
    return docs


def process_image(data, name):
    image = Image.open(io.BytesIO(data))
    ok, status = check_image_quality(image)
    if not ok:
        return [{'text': '', 'source': name, 'page': None, 'location': name, 'method': f'unreadable:{status}'}]
    text = perform_ocr(image)
    return [{'text': text, 'source': name, 'page': None, 'location': name, 'method': 'ocr'}] if text else \
           [{'text': '', 'source': name, 'page': None, 'location': name, 'method': 'unreadable'}]


def extract_documents(files):
    docs, unreadable = [], []
    for f in files:
        name, data = f.name, f.getvalue()
        ext = name.lower().rsplit('.', 1)[-1]
        try:
            if ext == 'pdf': new = process_pdf(data, name)
            elif ext == 'docx': new = process_docx(data, name)
            elif ext in {'txt', 'csv'}: new = process_text(data, name)
            elif ext == 'pptx': new = process_pptx(data, name)
            elif ext == 'xlsx': new = process_xlsx(data, name)
            elif ext in {'jpg', 'jpeg', 'png', 'webp'}: new = process_image(data, name)
            else: new = []
            for d in new:
                (docs if d['text'].strip() else unreadable).append(d)
        except Exception as e:
            unreadable.append({'source': name, 'page': None, 'location': name, 'method': 'error', 'error': str(e), 'text': ''})
    return docs, unreadable


def build_vector_database(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=100)
    chunks = []
    for d in docs:
        for text in splitter.split_text(d['text']):
            if text.strip():
                chunks.append({**{k: d[k] for k in ('source','page','location','method')}, 'text': text})
    if not chunks:
        return None, []
    emb = embedding_model.encode([c['text'] for c in chunks], normalize_embeddings=True, show_progress_bar=False)
    emb = np.asarray(emb, dtype='float32')
    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(emb)
    return index, chunks

STOP = {'what','is','are','the','a','an','of','to','in','on','for','and','or','how','where','when','why','which','who','does','do','did','can','could','would','please','tell','me','about','discussed','discuss','explain','define','definition','give','show'}
SYN = {'arrays': {'array'}, 'array': {'arrays'}, 'loops': {'loop','iteration','iterations'}, 'loop': {'loops','iteration','iterations'},
       'functions': {'function','method','methods'}, 'function': {'functions','method','methods'}, 'classes': {'class'}, 'class': {'classes'},
       'lists': {'list'}, 'list': {'lists'}, 'dictionaries': {'dictionary','dict'}, 'dictionary': {'dictionaries','dict'}}


def terms(question):
    out = set()
    for w in re.findall(r'[A-Za-z0-9_]+', question.lower()):
        if w not in STOP:
            out.add(w)
            out.update(SYN.get(w, set()))
    return out


def lexical_score(question, text):
    q = terms(question)
    if not q: return 0.0
    t = set(re.findall(r'[A-Za-z0-9_]+', text.lower()))
    return len(q & t) / len(q)


def search_documents(question, index, chunks, k=12):
    qemb = embedding_model.encode([question], normalize_embeddings=True, show_progress_bar=False)
    qemb = np.asarray(qemb, dtype='float32')
    scores, ids = index.search(qemb, min(k, len(chunks)))
    out = []
    for score, idx in zip(scores[0], ids[0]):
        if idx < 0: continue
        r = dict(chunks[idx])
        r['semantic_score'] = float(score)
        r['lexical_score'] = lexical_score(question, r['text'])
        r['combined_score'] = 0.80 * r['semantic_score'] + 0.20 * r['lexical_score']
        out.append(r)
    return sorted(out, key=lambda x: x['combined_score'], reverse=True)


def relevant_results(question, candidates):
    if not candidates: return []
    top = candidates[0]
    if top['lexical_score'] < 0.50 and top['semantic_score'] < 0.48:
        return []
    return [r for r in candidates if r['combined_score'] >= 0.40 and (r['semantic_score'] >= 0.40 or r['lexical_score'] >= 0.50)][:6]


def language_instruction(language):
    if language == 'Urdu': return 'Answer entirely in Urdu script. Do not use Roman Urdu. Use English only for necessary technical terms.'
    if language == 'Roman Urdu': return 'Answer entirely in Roman Urdu. Do not use Urdu script. Use English only for necessary technical terms.'
    return 'Answer entirely in English.'


def generate_answer(prompt):
    try:
        key = st.secrets['GEMINI_API_KEY']
    except Exception:
        return None, 'Gemini API key is not configured. Please add GEMINI_API_KEY in Streamlit Secrets.'
    client = genai.Client(api_key=key)
    last = ''
    for model in ('gemini-3.8-flash', 'gemini-3.7-flash', 'gemini-3.6-flash'):
        for attempt in range(2):
            try:
                response = client.models.generate_content(model=model, contents=prompt)
                if response.text and response.text.strip(): return response.text.strip(), None
                last = f'{model} returned an empty response.'
            except Exception as e:
                last = str(e)
                transient = any(x in last for x in ('429','500','502','503','504','UNAVAILABLE','RESOURCE_EXHAUSTED'))
                if transient and attempt == 0:
                    time.sleep(2)
                    continue
                break
    return None, 'The AI service is temporarily unavailable. Please try again in a moment.\n\nTechnical detail: ' + last


def ask_documents(question, index, chunks, language):
    candidates = search_documents(question, index, chunks)
    results = relevant_results(question, candidates)
    if not results:
        if language == 'Urdu': return 'مجھے اپ لوڈ کی گئی فائلوں میں اس سوال کا قابلِ اعتماد جواب نہیں ملا۔'
        if language == 'Roman Urdu': return 'Mujhe upload ki gayi files mein is sawal ka reliable jawab nahi mila.'
        return "I couldn't find reliable information about this question in the uploaded files."

    context = '\n'.join(f"SOURCE {i}\nLocation: {r['location']}\nContent:\n{r['text']}\n----------------" for i, r in enumerate(results, 1))
    prompt = f'''You are a professional document Q&A assistant.\n\nAnswer ONLY from the supplied document context.\n1. Never use outside knowledge to fill a missing answer.\n2. Never invent facts, definitions, examples, page numbers, or sources.\n3. If the context does not support the answer, say that it could not be found in the uploaded files.\n4. Answer the user's actual question directly.\n5. Explain clearly when the document provides enough information.\n6. Use ONLY the selected response language.\n7. Do not cite a source unless its content actually supports the answer.\n8. If asked where a topic is discussed, identify relevant source locations from the context.\n\nSelected language: {language}\n{language_instruction(language)}\n\nUSER QUESTION:\n{question}\n\nDOCUMENT CONTEXT:\n{context}'''
    answer, error = generate_answer(prompt)
    if error: return error
    seen, sources = set(), []
    for r in results:
        if r['location'] not in seen:
            seen.add(r['location']); sources.append(f"- **{r['location']}**")
    return answer + '\n\n### 📌 Sources\n\n' + '\n'.join(sources)

if 'vector_db' not in st.session_state: st.session_state.vector_db = None
if 'chunks' not in st.session_state: st.session_state.chunks = []
if 'unreadable_files' not in st.session_state: st.session_state.unreadable_files = []

uploaded_files = st.file_uploader('📁 Upload your files', type=['pdf','docx','txt','pptx','xlsx','csv','jpg','jpeg','png','webp'], accept_multiple_files=True)

if st.button('⚙️ Process Files', use_container_width=True):
    if not uploaded_files:
        st.warning('Please upload at least one file.')
    else:
        with st.spinner('Processing files...'):
            docs, unreadable = extract_documents(uploaded_files)
            index, chunks = build_vector_database(docs)
            st.session_state.vector_db = index
            st.session_state.chunks = chunks
            st.session_state.unreadable_files = unreadable
        if index is not None:
            st.success(f'Processed {len(uploaded_files)} file(s) successfully. {len(chunks)} searchable chunks created.')
        else:
            st.error('No readable content could be extracted from the uploaded files.')

if st.session_state.unreadable_files:
    st.warning('Some files or pages could not be read reliably.')
    with st.expander('View unreadable pages/files'):
        for item in st.session_state.unreadable_files:
            loc = item.get('location', item.get('source', 'Unknown file'))
            method = item.get('method', 'unreadable')
            if 'blurry' in method:
                st.error(f"⚠️ {loc}: This page appears blurry. We can't reliably read its content.")
            elif 'low_resolution' in method:
                st.error(f"⚠️ {loc}: This page has very low resolution. We can't reliably read its content.")
            else:
                st.error(f"⚠️ {loc}: We couldn't reliably extract readable content.")

st.divider()
question = st.text_area('💬 Ask a question', placeholder='Example: What is a loop?', height=100)

if st.button('🔍 Ask', use_container_width=True):
    if st.session_state.vector_db is None:
        st.warning('Please process your files first.')
    elif not question.strip():
        st.warning('Please enter a question.')
    else:
        with st.spinner('Searching documents and generating answer...'):
            answer = ask_documents(question, st.session_state.vector_db, st.session_state.chunks, response_language)
        st.markdown(answer)
