
import hashlib
from pathlib import Path

import streamlit as st
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ---------------- CONFIG ----------------

st.set_page_config(
    page_title="RAG Assistant",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE = Path(__file__).resolve().parent
DB = BASE / "Chroma-Db"
UPLOADS = BASE / "uploaded_pdfs"
UPLOADS.mkdir(exist_ok=True)

# Matches your original Chroma default collection
COLLECTION = "langchain"

# ---------------- CUSTOM CSS ----------------

st.markdown("""
<style>
html, body, [data-testid="stAppViewContainer"] {
    background: #0e0d19;
    color: #f5f1ff;
}

[data-testid="stAppViewContainer"] {
    background:
        radial-gradient(
            ellipse at 70% 85%,
            #22123e 0%,
            transparent 45%
        ),
        #0e0d19;
}

[data-testid="stSidebar"] {
    background: #11101f;
    border-right: 1px solid #302742;
}

[data-testid="stSidebar"] * {
    color: #eae4f7;
}

.block-container {
    max-width: 1100px;
    padding-top: 1.5rem;
    padding-bottom: 5rem;
}

h1, h2, h3 {
    color: #f6f3ff !important;
}

.brand {
    font-size: 25px;
    font-weight: 800;
    letter-spacing: -0.7px;
}

.brand span {
    color: #ad79ff;
}

.subtitle {
    color: #a9a0bc;
    font-size: 14px;
}

.hero {
    text-align: center;
    margin: 12vh auto 2rem;
}

.hero .symbol {
    font-size: 65px;
    color: #ad79ff;
}

.hero h1 {
    font-size: clamp(32px, 4vw, 53px);
    margin: 10px 0;
    background: linear-gradient(90deg, #fff, #c58aff);
    color: transparent !important;
    background-clip: text;
}

.hero p {
    font-size: 18px;
    color: #b6aec9;
}

[data-testid="stChatMessage"] {
    background: #1a182a;
    border: 1px solid #342b49;
    border-radius: 18px;
    margin: 14px 0;
    padding: 12px;
}

[data-testid="stChatInput"] {
    border: 1px solid #7854a6;
    border-radius: 20px;
    background: #1d192c;
}

[data-testid="stChatInput"] textarea {
    color: white !important;
}

.stButton > button {
    border: 1px solid #46345f;
    background: #211b34;
    color: #eee6ff;
    border-radius: 13px;
    min-height: 44px;
}

.stButton > button:hover {
    border-color: #ad79ff;
    background: #38244e;
    color: #fff;
}

[data-testid="stFileUploader"] {
    border: 1px dashed #7854a6;
    border-radius: 15px;
    padding: 8px;
}

[data-testid="stExpander"] {
    background: #171426;
    border: 1px solid #332a46;
    border-radius: 12px;
}

hr {
    border-color: #302743;
}
</style>
""", unsafe_allow_html=True)

# ---------------- MODELS ----------------

@st.cache_resource
def embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


@st.cache_resource
def llm():
    return ChatOllama(
        model="gemma3:4b",
        temperature=0.7
    )


@st.cache_resource
def store():
    return Chroma(
        collection_name=COLLECTION,
        persist_directory=str(DB),
        embedding_function=embeddings()
    )


# ---------------- PROMPT ----------------


prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
You are an expert AI assistant specializing in
document-based question answering.

Follow these instructions strictly:

1. Answer using ONLY the provided context.
2. Provide LONG, DETAILED, and COMPREHENSIVE answers.
3. Explain every relevant concept thoroughly.
4. Use well-structured paragraphs, headings, subheadings,
   and bullet points wherever appropriate.
5. Cover all important information available in the context.
6. Explain concepts step by step when necessary.
7. Include definitions, explanations, examples, advantages,
   disadvantages, and applications ONLY if they are
   available in the provided context.
8. Do not give short or one-line answers when sufficient
   information is available.
9. Do not summarize unnecessarily. Prioritize depth,
   clarity, and completeness.
10. Do not use external knowledge or make assumptions.
11. Avoid repeating the same information unnecessarily.
12. If the context contains limited information,
    explain only what is supported by the context.
13. If the answer is not present in the context,
    respond exactly:
    "I could not find the answer in the document."

Your goal is to generate the most detailed and
informative answer possible using the given context.
"""
    ),
    (
        "human",
        """
Context:
{context}

Question:
{question}

Provide a detailed, comprehensive, and well-structured
answer based strictly on the context.
"""
    )
])


# ---------------- SESSION STATE ----------------

if "chats" not in st.session_state:
    st.session_state.chats = {"Welcome Chat": []}

if "active" not in st.session_state:
    st.session_state.active = "Welcome Chat"

if "indexed_files" not in st.session_state:
    st.session_state.indexed_files = {}

if "question" not in st.session_state:
    st.session_state.question = None

# ---------------- SIDEBAR ----------------

with st.sidebar:

    st.markdown(
        '<div class="brand">◈ RAG <span>Assistant</span></div>'
        '<div class="subtitle">PDF · LangChain · ChromaDB</div>',
        unsafe_allow_html=True
    )

    if st.button("＋ New Chat", use_container_width=True):
        name = f"New Chat {len(st.session_state.chats) + 1}"
        st.session_state.chats[name] = []
        st.session_state.active = name
        st.rerun()

    search = st.text_input(
        "Search Chats",
        placeholder="Search conversations..."
    )

    st.markdown("#### Recent Chats")

    for name in list(st.session_state.chats.keys())[::-1]:
        if search.lower() in name.lower():
            if st.button(
                ("● " if name == st.session_state.active else "◌ ")
                + name[:35],
                key="chat_" + name,
                use_container_width=True
            ):
                st.session_state.active = name
                st.rerun()

    st.divider()
    st.markdown("#### My Documents")

    files = st.file_uploader(
        "Upload PDF",
        type=["pdf"],
        accept_multiple_files=True
    )

    if files and st.button(
        "Index uploaded PDFs",
        use_container_width=True
    ):

        for f in files:
            raw = f.getvalue()
            digest = hashlib.sha256(raw).hexdigest()

            if digest in st.session_state.indexed_files:
                st.info(f"{f.name} already indexed")
                continue

            file_path = UPLOADS / f"{digest}.pdf"
            file_path.write_bytes(raw)

            try:
                with st.spinner(f"Indexing {f.name}..."):

                    pages = PyPDFLoader(str(file_path)).load()

                    for page in pages:
                        page.metadata["source"] = f.name
                        page.metadata["file_hash"] = digest

                    splitter = RecursiveCharacterTextSplitter(
                        chunk_size=1000,
                        chunk_overlap=200
                    )

                    chunks = splitter.split_documents(pages)

                    ids = [
                        f"{digest}-{i}"
                        for i in range(len(chunks))
                    ]

                    store().add_documents(
                        chunks,
                        ids=ids
                    )

                    st.session_state.indexed_files[digest] = {
                        "name": f.name,
                        "pages": len(pages),
                        "chunks": len(chunks)
                    }

                st.success(f"Indexed {f.name}")

            except Exception as exc:
                st.error(f"Indexing failed: {exc}")

    for info in st.session_state.indexed_files.values():
        st.caption(
            f"📄 {info['name']} · "
            f"{info['pages']} pages · "
            f"{info['chunks']} chunks"
        )

    if not st.session_state.indexed_files and DB.exists():
        st.caption("✓ Existing Chroma-Db will be reused")

    with st.expander("⚙ Retrieval Settings"):
        k = st.slider("Results (k)", 1, 10, 4)

        fetch_k = st.slider(
            "Candidates (fetch_k)",
            k, 30, max(10, k)
        )

        diversity = st.slider(
            "lambda_mult",
            0.0, 1.0, 0.5, 0.05
        )

    st.caption("🤖 Ollama · gemma3:4b")

# ---------------- MAIN UI ----------------

st.markdown(
    '<div class="brand">◈ RAG Assistant <span>⌄</span></div>'
    '<div class="subtitle">'
    'Powered by LangChain + Chroma + Gemma 3:4b'
    '</div>',
    unsafe_allow_html=True
)

messages = st.session_state.chats[
    st.session_state.active
]

# ---------------- WELCOME SCREEN ----------------

if not messages:

    st.markdown("""
    <div class="hero">
        <div class="symbol">◈</div>
        <h1>How can I help you today?</h1>
        <p>
            Ask any question about your PDF documents using RAG.
        </p>
    </div>
    """, unsafe_allow_html=True)

    choices = [
        "Summarize the document",
        "Explain a concept",
        "List key applications",
        "Give important takeaways"
    ]

    cols = st.columns(4)

    for col, choice in zip(cols, choices):
        with col:
            if st.button(choice, use_container_width=True):
                st.session_state.question = choice
                st.rerun()

# ---------------- CHAT HISTORY ----------------

for message in messages:

    with st.chat_message(
        message["role"],
        avatar="🟣" if message["role"] == "assistant" else "👤"
    ):

        st.markdown(message["content"])

        if message.get("sources"):
            with st.expander("📚 Retrieved PDF sources"):
                for src in message["sources"]:
                    st.caption(src)

# ---------------- CHAT INPUT ----------------

question = st.chat_input(
    "Ask a question about your documents..."
)

if st.session_state.question:
    question = st.session_state.question
    st.session_state.question = None

# ---------------- RAG PIPELINE ----------------

if question and question.strip():

    messages.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("user", avatar="👤"):
        st.markdown(question)

    with st.chat_message("assistant", avatar="🟣"):

        try:
            with st.spinner("Searching your documents..."):

                db = store()

                if db._collection.count() == 0:
                    answer = "Please upload and index a PDF first."
                    sources = []

                else:
                    retriever = db.as_retriever(
                        search_type="mmr",
                        search_kwargs={
                            "k": k,
                            "fetch_k": fetch_k,
                            "lambda_mult": diversity
                        }
                    )

                    retrieved = retriever.invoke(question)

                    context = "\n\n".join(
                        doc.page_content
                        for doc in retrieved
                    )

                    final_prompt = prompt.invoke({
                        "context": context,
                        "question": question
                    })

                    response = llm().invoke(final_prompt)
                    answer = response.content

                    sources = list(dict.fromkeys(
                        f"{Path(str(d.metadata.get('source', 'Document'))).name}"
                        f" · page {int(d.metadata.get('page', 0)) + 1}"
                        for d in retrieved
                    ))

            st.markdown(answer)

            if sources:
                with st.expander("📚 Retrieved PDF sources"):
                    for src in sources:
                        st.caption(src)

            messages.append({
                "role": "assistant",
                "content": answer,
                "sources": sources
            })

        except Exception as exc:
            st.error(
                f"RAG error: {exc}. "
                "Check Ollama is running and gemma3:4b is installed."
            )

    st.rerun()
