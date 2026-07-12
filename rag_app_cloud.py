"""
DocChat AI — Cloud Version (multi-file + memory)
====================================================
Stack: Streamlit · LangChain · HuggingFace embeddings · Groq (Llama 3) · ChromaDB

Adapted from the local Ollama version to run on free hosting (Streamlit
Community Cloud), where a local Ollama server isn't available.

Supports:
    - Multiple files at once, of different types: PDF, DOCX, TXT, MD, CSV
    - Friendly conversational memory (remembers the whole chat)

Get a free Groq API key: https://console.groq.com/keys

Local run:
    export GROQ_API_KEY="your_key_here"     # Windows: set GROQ_API_KEY=your_key_here
    streamlit run rag_app_cloud.py

Streamlit Community Cloud:
    Add GROQ_API_KEY under your app's Settings -> Secrets, formatted as:
    GROQ_API_KEY = "your_key_here"
"""

import os
import tempfile

import streamlit as st
from langchain_community.document_loaders import (
    PyPDFLoader, Docx2txtLoader, TextLoader, CSVLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_chroma import Chroma

# ─────────────────────────────────────────────
#  Page config
# ─────────────────────────────────────────────
st.set_page_config(page_title="DocChat AI", page_icon="📚", layout="wide")

try:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
except Exception:
    GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
    
# ─────────────────────────────────────────────
#  File type -> loader mapping
# ─────────────────────────────────────────────
LOADER_MAP = {
    ".pdf": lambda path: PyPDFLoader(path),
    ".docx": lambda path: Docx2txtLoader(path),
    ".txt": lambda path: TextLoader(path, encoding="utf-8"),
    ".md": lambda path: TextLoader(path, encoding="utf-8"),
    ".csv": lambda path: CSVLoader(path, encoding="utf-8"),
}
SUPPORTED_TYPES = ["pdf", "docx", "txt", "md", "csv"]

# ─────────────────────────────────────────────
#  Session state
# ─────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "indexed_files" not in st.session_state:
    st.session_state.indexed_files = []
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = True

# ─────────────────────────────────────────────
#  Theme palettes — professional, muted, slate-based
# ─────────────────────────────────────────────
DARK = {
    "bg": "#0f1115", "bg_alt": "#161920", "sidebar_bg": "#13151b", "border": "#2a2e38",
    "text": "#e4e6eb", "text_muted": "#9096a3", "accent": "#5b7fdb",
    "user_bubble": "#1c2333", "bot_bubble": "#181b22",
    "success_bg": "#152420", "success_text": "#7bc9a0", "success_border": "#2c5c47",
    "wait_bg": "#1d2130", "wait_text": "#8b95ab", "wait_border": "#3a4258",
    "source_bg": "#151a24", "source_border": "#2a3548",
}
LIGHT = {
    "bg": "#fafafa", "bg_alt": "#ffffff", "sidebar_bg": "#f2f3f5", "border": "#e0e1e6",
    "text": "#1f2328", "text_muted": "#5c6370", "accent": "#3a5bb8",
    "user_bubble": "#eaeefb", "bot_bubble": "#ffffff",
    "success_bg": "#e9f6ef", "success_text": "#227a53", "success_border": "#a8dcc0",
    "wait_bg": "#eef0f4", "wait_text": "#5c6370", "wait_border": "#c9cdd6",
    "source_bg": "#f4f6fa", "source_border": "#d7deea",
}
T = DARK if st.session_state.dark_mode else LIGHT

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"] {{ font-family: 'Inter', sans-serif; }}
.stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"], .main {{ background: {T['bg']}; }}
[data-testid="stHeader"] {{ background: {T['bg']}; }}
[data-testid="stBottomBlockContainer"] {{ background: {T['bg']}; }}
.block-container {{ background: transparent; }}
[data-testid="stSidebar"] {{ background: {T['sidebar_bg']}; border-right: 1px solid {T['border']}; }}
[data-testid="stSidebar"] * {{ color: {T['text']} !important; }}
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {{ color: {T['text']} !important; }}
[data-testid="stMarkdownContainer"] p, [data-testid="stMarkdownContainer"] li {{ color: {T['text']}; }}
[data-testid="stCaptionContainer"] {{ color: {T['text_muted']} !important; }}
[data-testid="stAlertContentSuccess"], [data-testid="stAlertContentInfo"] {{ background: {T['success_bg']} !important; color: {T['success_text']} !important; }}
[data-testid="stAlertContentWarning"], [data-testid="stAlertContentError"] {{ background: {T['wait_bg']} !important; color: {T['text']} !important; }}
div[data-testid="stAlert"] {{ border: 1px solid {T['border']}; border-radius: 8px; }}
div[data-testid="stAlert"] p {{ color: inherit !important; }}
[data-testid="stExpander"] {{ background: {T['bg_alt']}; border: 1px solid {T['border']} !important; border-radius: 8px; }}
[data-testid="stExpander"] summary {{ color: {T['text']} !important; }}
[data-testid="stExpander"] p {{ color: {T['text_muted']} !important; }}
[data-testid="stSpinner"] p {{ color: {T['text']} !important; }}
.app-header {{ background: {T['bg_alt']}; border: 1px solid {T['border']}; border-left: 3px solid {T['accent']};
    border-radius: 10px; padding: 18px 24px; margin-bottom: 22px; }}
.app-header h1 {{ color: {T['text']}; margin: 0; font-size: 1.4rem; font-weight: 700; }}
.app-header p  {{ color: {T['text_muted']}; margin: 4px 0 0; font-size: 0.85rem; }}
.badge {{ display: inline-block; padding: 3px 10px; border-radius: 6px; font-size: 0.75rem; font-weight: 600; margin: 3px 4px 0 0; }}
.badge-ready {{ background: {T['success_bg']}; color: {T['success_text']}; border: 1px solid {T['success_border']}; }}
.badge-wait  {{ background: {T['wait_bg']}; color: {T['wait_text']}; border: 1px solid {T['wait_border']}; }}
.msg-user {{ background: {T['user_bubble']}; border: 1px solid {T['border']}; border-radius: 10px 10px 2px 10px;
    padding: 12px 16px; margin: 8px 0 8px 50px; color: {T['text']}; line-height: 1.6; }}
.msg-bot {{ background: {T['bot_bubble']}; border: 1px solid {T['border']}; border-radius: 10px 10px 10px 2px;
    padding: 12px 16px; margin: 8px 50px 8px 0; color: {T['text']}; line-height: 1.6; }}
.msg-label {{ font-size: 0.7rem; font-weight: 600; letter-spacing: 0.04em; text-transform: uppercase; margin-bottom: 5px; }}
.label-user {{ color: {T['accent']}; }}
.label-bot  {{ color: {T['text_muted']}; }}
.source-card {{ background: {T['source_bg']}; border: 1px solid {T['source_border']}; border-left: 3px solid {T['accent']};
    border-radius: 6px; padding: 8px 12px; margin-top: 6px; font-size: 0.8rem; color: {T['text_muted']}; }}
.source-card strong {{ color: {T['text']}; }}
[data-testid="stFileUploader"] {{ background: {T['bg_alt']}; border: 1.5px dashed {T['border']}; border-radius: 8px; padding: 8px; }}
[data-testid="stChatInput"] textarea {{ background: {T['bg_alt']} !important; border: 1px solid {T['border']} !important;
    border-radius: 8px !important; color: {T['text']} !important; }}
.stButton > button {{ background: {T['accent']}; color: white; border: none; border-radius: 6px; font-weight: 600; padding: 8px 18px; }}
.stButton > button:hover {{ opacity: 0.88; }}
hr {{ border-color: {T['border']}; }}
.file-chip {{ background: {T['bg_alt']}; border: 1px solid {T['border']}; border-radius: 6px; padding: 5px 10px;
    font-size: 0.78rem; color: {T['text']}; margin: 3px 4px 3px 0; display: inline-block; }}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_llm():
    return ChatGroq(model="llama3-8b-8192", temperature=0.4, api_key=GROQ_API_KEY)


@st.cache_resource(show_spinner=False)
def load_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


def load_one_file(uploaded_file):
    ext = os.path.splitext(uploaded_file.name)[1].lower()
    if ext not in LOADER_MAP:
        return None, f"Unsupported file type: {ext}"

    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    try:
        loader = LOADER_MAP[ext](tmp_path)
        docs = loader.load()
        for doc in docs:
            doc.metadata["source_file"] = uploaded_file.name
        return docs, None
    except Exception as e:
        return None, str(e)
    finally:
        os.unlink(tmp_path)


def index_files(uploaded_files):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    embeddings = load_embeddings()

    all_chunks = []
    file_summaries = []
    errors = []

    for uploaded_file in uploaded_files:
        docs, error = load_one_file(uploaded_file)
        if error:
            errors.append(f"{uploaded_file.name}: {error}")
            continue
        chunks = splitter.split_documents(docs)
        all_chunks.extend(chunks)
        file_summaries.append({"name": uploaded_file.name, "chunks": len(chunks)})

    if not all_chunks:
        return errors

    if st.session_state.vector_store is None:
        st.session_state.vector_store = Chroma.from_documents(
            documents=all_chunks, embedding=embeddings, collection_name="rag_chat",
        )
    else:
        st.session_state.vector_store.add_documents(all_chunks)

    st.session_state.indexed_files.extend(file_summaries)
    return errors


def build_prompt(history: list, context: str, question: str) -> str:
    """Friendly system prompt with full conversation memory."""
    memory_text = ""
    if history:
        lines = []
        for msg in history:
            role = "User" if msg["role"] == "user" else "You"
            lines.append(f"{role}: {msg['content']}")
        memory_text = "\n".join(lines)

    prompt = f"""You are a friendly, helpful study buddy who has read the user's uploaded documents.
Talk like a knowledgeable friend explaining things clearly — warm and conversational, not stiff or robotic.
Use the conversation history to stay consistent and remember what's already been discussed.
Base your answer on the document context below. If the answer truly isn't in the documents, say so honestly
and offer to help in another way, rather than making something up.

Conversation so far:
{memory_text if memory_text else "(this is the first message)"}

Relevant document context:
{context}

User's new message: {question}

Your reply (friendly, clear, and grounded in the documents):"""
    return prompt


def ask(question: str) -> dict:
    retriever = st.session_state.vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 5, "fetch_k": 20, "lambda_mult": 0.5},
    )
    docs = retriever.invoke(question)
    context = "\n\n".join(f"[{d.metadata.get('source_file','?')}] {d.page_content}" for d in docs)
    prompt = build_prompt(st.session_state.messages, context, question)

    llm = load_llm()
    response = llm.invoke(prompt)

    sources = [
        {
            "file": d.metadata.get("source_file", "unknown"),
            "page": d.metadata.get("page", "-"),
            "snippet": d.page_content[:120].replace("\n", " "),
        }
        for d in docs
    ]
    return {"answer": response.content, "sources": sources}


# ─────────────────────────────────────────────
#  Sidebar
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📚 DocChat AI")
    st.markdown("Upload files and chat. It remembers your whole conversation.")

    if not GROQ_API_KEY:
        st.error("No GROQ_API_KEY found. Add it in Settings → Secrets (cloud) or as an environment variable (local).")

    st.toggle("Dark mode", key="dark_mode")

    st.markdown("---")
    st.markdown("### Upload files")
    st.caption("PDF · DOCX · TXT · MD · CSV — upload as many as you like")

    uploaded_files = st.file_uploader(
        "Choose files", type=SUPPORTED_TYPES, accept_multiple_files=True,
        label_visibility="collapsed", key="uploader",
    )

    already_indexed = {f["name"] for f in st.session_state.indexed_files}
    new_files = [f for f in (uploaded_files or []) if f.name not in already_indexed]

    if new_files:
        with st.spinner(f"Reading and indexing {len(new_files)} file(s)…"):
            errors = index_files(new_files)
        if errors:
            for e in errors:
                st.warning(e)
        st.success(f"Indexed {len(new_files)} new file(s)")

    if st.session_state.indexed_files:
        st.markdown("**Knowledge base:**")
        for f in st.session_state.indexed_files:
            st.markdown(f'<span class="file-chip">📄 {f["name"]} ({f["chunks"]} chunks)</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="badge badge-wait">No files yet</span>', unsafe_allow_html=True)

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Clear chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
    with col2:
        if st.button("Reset all", use_container_width=True):
            st.session_state.messages = []
            st.session_state.vector_store = None
            st.session_state.indexed_files = []
            st.rerun()

    st.markdown("---")
    st.markdown("### How it works")
    for name, desc in [
        ("Multi-loader", "reads PDF/DOCX/TXT/MD/CSV"),
        ("Text splitter", "chunks into 1000-char pieces"),
        ("MiniLM embeddings", "creates vector embeddings"),
        ("ChromaDB", "stores & searches all files together"),
        ("MMR retriever", "finds relevant + diverse chunks"),
        ("Full memory", "remembers the whole conversation"),
        ("Llama 3 (Groq)", "replies in a friendly, conversational tone"),
    ]:
        st.markdown(f"**{name}** — {desc}")


# ─────────────────────────────────────────────
#  Main header
# ─────────────────────────────────────────────
st.markdown(f"""
<div class="app-header">
    <h1>DocChat AI</h1>
    <p>Your friendly AI that's read all your uploaded documents — ask it anything, and it'll remember the whole conversation.</p>
</div>
""", unsafe_allow_html=True)

if not st.session_state.messages:
    st.markdown(f"""
    <div style="text-align:center; padding: 60px 20px; color: {T['text_muted']};">
        <div style="font-size:1.05rem; font-weight:600; color:{T['text']};">Upload some files to get started</div>
        <div style="font-size:0.85rem; margin-top:8px;">
            Mix and match PDFs, Word docs, text files, and CSVs — I'll read them all and chat with you about them.
        </div>
    </div>
    """, unsafe_allow_html=True)

for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f"""<div class="msg-user"><div class="msg-label label-user">You</div>{msg["content"]}</div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""<div class="msg-bot"><div class="msg-label label-bot">DocChat AI</div>{msg["content"]}</div>""", unsafe_allow_html=True)
        if msg.get("sources"):
            with st.expander(f"{len(msg['sources'])} source chunks used", expanded=False):
                for i, src in enumerate(msg["sources"], 1):
                    st.markdown(f"""
                    <div class="source-card"><strong>Chunk {i} · {src['file']} (page {src['page']})</strong><br>"{src['snippet']}…"</div>
                    """, unsafe_allow_html=True)

if st.session_state.vector_store is None:
    st.chat_input("Upload a file first…", disabled=True)
elif not GROQ_API_KEY:
    st.chat_input("Add your GROQ_API_KEY to enable chat…", disabled=True)
else:
    question = st.chat_input("Ask me anything about your files…")
    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.spinner("Thinking…"):
            result = ask(question)
        st.session_state.messages.append({"role": "assistant", "content": result["answer"], "sources": result["sources"]})
        st.rerun()
