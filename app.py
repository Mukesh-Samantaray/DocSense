import os
import tempfile
from pathlib import Path
from typing import List, Dict

import streamlit as st
from dotenv import load_dotenv
load_dotenv()

# PDF/Image libraries
try:
    import fitz  # PyMuPDF
except Exception:
    fitz = None

try:
    import pdfplumber
except Exception:
    pdfplumber = None

# LangChain imports
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import UnstructuredPDFLoader

# Gemini LLM
try:
    from langchain.chat_models import ChatGoogleGenerativeAI
except Exception:
    ChatGoogleGenerativeAI = None

# Optional OCR
try:
    import pytesseract
    from PIL import Image
except Exception:
    pytesseract = None
    Image = None

# ---------------- App Config ----------------
st.set_page_config(page_title="Gemini PDF Multimodal RAG", layout="wide")
st.title("📚 DocSense – AI that “understands” documents")

STORAGE_DIR = Path("./chroma_store")
IMAGES_DIR = Path("./page_images")
STORAGE_DIR.mkdir(exist_ok=True)
IMAGES_DIR.mkdir(exist_ok=True)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("Settings")
    st.write("Gemini API only. Ensure GEMINI_API_KEY is set in .env")
    st.text(f"Embedding model: {EMBEDDING_MODEL}")
    st.write("OCR available: " + ("✅" if pytesseract else "❌"))

# ---------------- Helpers ----------------

def extract_text_with_unstructured(pdf_path: str) -> str:
    try:
        loader = UnstructuredPDFLoader(pdf_path, mode="elements")
        docs = loader.load()
        return "\n\n".join([d.page_content for d in docs if d.page_content])
    except Exception:
        if pdfplumber:
            text_chunks = []
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    t = page.extract_text() or ""
                    text_chunks.append(t)
            return "\n\n".join(text_chunks)
        else:
            st.warning("No PDF text extraction library available.")
            return ""

def extract_page_images(pdf_path: str, output_dir: Path) -> List[Dict]:
    imgs = []
    if fitz is None:
        st.warning("PyMuPDF not installed — cannot extract page images.")
        return imgs

    doc = fitz.open(pdf_path)
    for page_index in range(len(doc)):
        page = doc[page_index]
        mat = fitz.Matrix(1.5, 1.5)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        out_path = output_dir / f"page_{page_index+1}.png"
        pix.save(str(out_path))
        imgs.append({"page": page_index + 1, "image_path": str(out_path)})
    return imgs

def chunk_and_persist(text: str, persist_dir: Path, embedding_model_name: str):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = [Document(page_content=t) for t in splitter.split_text(text)]
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
    db = Chroma.from_documents(documents=docs, embedding=embeddings, persist_directory=str(persist_dir))
    db.persist()
    return db

def load_or_build_index(pdf_path: str, persist_dir: Path, embedding_model_name: str):
    if persist_dir.exists() and any(persist_dir.iterdir()):
        try:
            embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
            db = Chroma(persist_directory=str(persist_dir), embedding_function=embeddings)
            return db
        except Exception:
            pass
    txt = extract_text_with_unstructured(pdf_path)
    if not txt.strip():
        st.error("No text extracted — cannot build index.")
        return None
    return chunk_and_persist(txt, persist_dir, embedding_model_name)

def is_image_question(q: str) -> bool:
    keywords = ["chart", "graph", "figure", "image", "plot", "table", "diagram", "visual"]
    return any(k in q.lower() for k in keywords)

def run_gemini_answer(retriever, question: str, include_images: List[Dict]):
    docs = retriever.get_relevant_documents(question)
    context = "\n\n".join([d.page_content for d in docs])

    if not GEMINI_API_KEY or ChatGoogleGenerativeAI is None:
        st.error("Gemini API not configured or ChatGoogleGenerativeAI not available.")
        return "Cannot generate answer — Gemini not available.", docs

    image_text = ""
    for im in include_images:
        image_text += f"Page {im['page']}: {im['image_path']}\n"

    prompt = f"""
You are an expert assistant. Use the context and the images (paths provided) to answer.
Context:
{context}

Images:
{image_text}

Question:
{question}

Answer:
"""
    model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", api_key=GEMINI_API_KEY)
    try:
        resp = model.generate([{"role": "user", "content": prompt}])
        text = ""
        for g in resp.generations:
            for gen in g:
                text += gen.text if hasattr(gen, "text") else str(gen)
        if not text:
            text = str(resp)
        return text, docs
    except Exception as e:
        st.error(f"Gemini call failed: {e}")
        return f"Gemini call failed: {e}", docs

# ---------------- UI ----------------

uploaded = st.file_uploader("Upload PDF (text + images/graphs)", type=["pdf"])
build_index_btn = st.button("Build / Rebuild Index (text chunks -> Chroma)")

if uploaded:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded.read())
        tmp.flush()
        pdf_path = tmp.name

    st.success("PDF received. Extracting text and images...")
    image_list = extract_page_images(pdf_path, IMAGES_DIR)

    if build_index_btn or not (STORAGE_DIR.exists() and any(STORAGE_DIR.iterdir())):
        with st.spinner("Building Chroma index..."):
            db = load_or_build_index(pdf_path, STORAGE_DIR, EMBEDDING_MODEL)
            if db:
                st.success("Index built successfully.")
    else:
        db = load_or_build_index(pdf_path, STORAGE_DIR, EMBEDDING_MODEL)
        st.info("Using existing Chroma index.")

    if db:
        retriever = db.as_retriever(search_kwargs={"k": 4})

        st.subheader("Ask a question about the PDF")
        user_q = st.text_input("Question (try: 'What does the chart on page 2 show?')")

        if user_q:
            include_images = image_list[:5] if is_image_question(user_q) else []
            with st.spinner("Generating answer via Gemini..."):
                answer, retrieved_docs = run_gemini_answer(retriever, user_q, include_images)
            st.markdown("### ✅ Answer")
            st.write(answer)

            with st.expander("Retrieved text chunks"):
                for i, d in enumerate(retrieved_docs, 1):
                    st.markdown(f"**Chunk {i}:**")
                    st.write(d.page_content[:1000] + ("..." if len(d.page_content) > 1000 else ""))

            if include_images:
                with st.expander("Images used for reasoning"):
                    for im in include_images:
                        st.markdown(f"**Page {im['page']}**")
                        st.image(im["image_path"], use_column_width=True)
                        if pytesseract and Image:
                            try:
                                text_from_image = pytesseract.image_to_string(Image.open(im["image_path"]))
                                if text_from_image.strip():
                                    st.markdown("**OCR text from image:**")
                                    st.write(text_from_image.strip()[:2000])
                            except:
                                pass

        if st.button("Show all images/graphs from PDF"):
            if image_list:
                st.subheader("All PDF Images / Graphs")
                for im in image_list:
                    st.markdown(f"**Page {im['page']}**")
                    st.image(im["image_path"], use_column_width=True)
                    if pytesseract and Image:
                        try:
                            text_from_image = pytesseract.image_to_string(Image.open(im["image_path"]))
                            if text_from_image.strip():
                                st.markdown("**OCR text from image:**")
                                st.write(text_from_image.strip()[:2000])
                        except:
                            pass
            else:
                st.info("No images/graphs found in this PDF.")

else:
    st.info("Upload a PDF to get started (supports text + images/graphs).")
