# ================================
# 📦 IMPORTS & CONFIGURATION
# ================================
import os
import tempfile
from dotenv import load_dotenv

# LangChain - Loaders & Processing
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Embeddings & Vector Store
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS

# LangChain Core
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

# UI
import streamlit as st

# ================================
# 🔐 ENVIRONMENT SETUP
# ================================
load_dotenv()

if not os.getenv("GROQ_API_KEY"):
    st.warning("GROQ_API_KEY missing. Add it to your .env file.")

# ================================
# ⚙️ PAGE CONFIG
# ================================
st.set_page_config(page_title="My RAG Knowledge Assistant", page_icon="📚", layout="wide")
st.title("📘 My RAG-Based Knowledge Assistant")
st.caption("Upload a PDF and ask questions about it — powered by free local embeddings + Groq LLM")

# ================================
# 🧠 CACHE THE EMBEDDING MODEL (loads once, not on every rerun)
# ================================
@st.cache_resource
def get_embedding_model():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

embedding_model = get_embedding_model()

# ================================
# 📥 SIDEBAR: FILE UPLOAD
# ================================
with st.sidebar:
    st.markdown("### 👋 Welcome")
    st.markdown("Upload any PDF and ask questions — answers come only from your document.")
    st.markdown("---")
    st.header("📄 Upload Document")
    uploaded_file = st.file_uploader("Choose a PDF", type="pdf")
    process_btn = st.button("Process Document")

# Keep vector store in session so it persists across reruns
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "messages" not in st.session_state:
    st.session_state.messages = []

if process_btn and uploaded_file is not None:
    with st.spinner("Reading and indexing your PDF..."):
        # Save uploaded file to a temp path so PyMuPDFLoader can read it
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        loader = PyMuPDFLoader(tmp_path)
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=250)
        chunks = splitter.split_documents(docs)

        st.session_state.vector_store = FAISS.from_documents(chunks, embedding_model)
        st.session_state.messages = []  # reset chat for new document

    st.success(f"Processed '{uploaded_file.name}' — {len(chunks)} chunks indexed.")

# ================================
# 🔍 RETRIEVAL + GENERATION PIPELINE
# ================================
def build_chain(vector_store):
    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 5, "lambda_mult": 0.7}
    )

    def context_creator(retrieved_docs):
        return "\n\n".join(doc.page_content for doc in retrieved_docs), retrieved_docs

    parser = StrOutputParser()
    model = ChatGroq(model="llama-3.1-8b-instant")

    prompt = PromptTemplate(
        template="""
Answer the question strictly using the provided context.
If the answer is not present, say "I don't know".

Keep the answer concise and accurate.

Context:
{context}

Question:
{query}
""",
        input_variables=["context", "query"],
    )

    return retriever, context_creator, prompt, model, parser

# ================================
# 💬 CHAT UI
# ================================
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander("Sources"):
                for i, s in enumerate(msg["sources"], 1):
                    st.markdown(f"**Chunk {i}:** {s[:400]}...")

if st.session_state.vector_store is None:
    st.info("👈 Upload a PDF and click 'Process Document' to get started.")
else:
    user_input = st.chat_input("Ask something about your document...")
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.write(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                retriever, context_creator, prompt, model, parser = build_chain(
                    st.session_state.vector_store
                )
                retrieved_docs = retriever.invoke(user_input)
                context_text, docs = context_creator(retrieved_docs)

                final_prompt = prompt.format(context=context_text, query=user_input)
                result = parser.invoke(model.invoke(final_prompt))

                st.write(result)
                sources = [d.page_content for d in docs]
                with st.expander("Sources"):
                    for i, s in enumerate(sources, 1):
                        st.markdown(f"**Chunk {i}:** {s[:400]}...")

        st.session_state.messages.append(
            {"role": "assistant", "content": result, "sources": sources}
        )

st.markdown("---")
st.caption("Built by Mahi • Powered by Groq + HuggingFace • Free & Open Source")