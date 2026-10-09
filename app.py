import streamlit as st
import os
import sys
import tempfile
import gc
import base64
import time
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Add repository root and src directory to sys.path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))
if str(BASE_DIR / "src") not in sys.path:
    sys.path.insert(0, str(BASE_DIR / "src"))

from crewai import LLM
from src.agentic_rag.crew import AgenticRag
from src.agentic_rag.tools.custom_tool import DocumentSearchTool, get_web_search_tool

# Supported document types for MarkItDown
SUPPORTED_EXTENSIONS = ["pdf", "docx", "pptx", "xlsx", "html", "md", "txt", "csv"]

# ===========================
#   Streamlit Page Config
# ===========================
st.set_page_config(
    page_title="ContextForge - Agentic RAG",
    page_icon="🤖",
    layout="wide"
)

# ===========================
#   Session State Setup
# ===========================
if "messages" not in st.session_state:
    st.session_state.messages = []

if "doc_tool" not in st.session_state:
    st.session_state.doc_tool = None

if "indexed_files" not in st.session_state:
    st.session_state.indexed_files = []

def reset_chat():
    st.session_state.messages = []
    gc.collect()

def clear_vector_index():
    st.session_state.doc_tool = None
    st.session_state.indexed_files = []
    st.session_state.messages = []
    gc.collect()

def get_llm_instance(model_choice: str, custom_model_name: str, ollama_url: str):
    """Factory to get the selected LLM instance."""
    if model_choice == "OpenAI / Default (API Key)":
        return None  # CrewAI uses default OpenAI configuration from environment
    elif model_choice == "Ollama: DeepSeek-R1 (7B)":
        return LLM(model="ollama/deepseek-r1:7b", base_url=ollama_url)
    elif model_choice == "Ollama: Llama 3.2":
        return LLM(model="ollama/llama3.2", base_url=ollama_url)
    elif model_choice == "Ollama: Custom":
        clean_name = custom_model_name.strip()
        model_tag = f"ollama/{clean_name}" if not clean_name.startswith("ollama/") else clean_name
        return LLM(model=model_tag, base_url=ollama_url)
    return None

# ===========================
#   Sidebar Controls
# ===========================
with st.sidebar:
    st.header("⚙️ Configuration")

    st.subheader("1. Model Selection")
    model_choice = st.selectbox(
        "Choose LLM Provider / Model",
        [
            "OpenAI / Default (API Key)",
            "Ollama: DeepSeek-R1 (7B)",
            "Ollama: Llama 3.2",
            "Ollama: Custom"
        ],
        index=0
    )

    ollama_url = "http://localhost:11434"
    custom_model_name = ""
    if "Ollama" in model_choice:
        ollama_url = st.text_input("Ollama Host URL", value="http://localhost:11434")
        if model_choice == "Ollama: Custom":
            custom_model_name = st.text_input("Custom Model Name (e.g. mistral, qwen2.5)", value="")

    st.subheader("2. Web Search Provider")
    search_provider = st.selectbox(
        "Search Tool Fallback",
        ["Auto-detect", "Serper", "Firecrawl"],
        index=0
    )

    st.divider()

    st.subheader("3. RAG & Retrieval Settings")
    chunking_strategy = st.selectbox(
        "Chunking Strategy",
        ["semantic", "recursive", "sentence"],
        index=0,
        help="Semantic chunking uses embedding similarity to divide topics; Recursive splits on paragraphs/tokens."
    )
    chunk_size = st.slider("Chunk Size", min_value=128, max_value=1024, value=512, step=64)
    top_k = st.slider("Top Chunks (k)", min_value=1, max_value=10, value=5)
    enable_rerank = st.checkbox("Cross-Encoder Re-ranking", value=True, help="Re-ranks candidate chunks using FastEmbed ms-marco-MiniLM cross-encoder for sharper relevance.")
    storage_mode = st.radio("Vector Store Mode", ["Persistent Disk", "In-Memory"], index=0, horizontal=True)

    st.divider()

    st.subheader("4. Knowledge Base")
    uploaded_files = st.file_uploader(
        "Upload Documents (Multi-file & Multi-format)",
        type=SUPPORTED_EXTENSIONS,
        accept_multiple_files=True,
        help="Supports .pdf, .docx, .pptx, .xlsx, .html, .md, .txt, .csv"
    )

    if uploaded_files:
        new_files = [f for f in uploaded_files if f.name not in st.session_state.indexed_files]
        if new_files:
            with tempfile.TemporaryDirectory() as temp_dir:
                saved_paths = []
                for uf in uploaded_files:
                    target_path = os.path.join(temp_dir, uf.name)
                    with open(target_path, "wb") as f:
                        f.write(uf.getvalue())
                    saved_paths.append(target_path)

                with st.spinner(f"Indexing {len(uploaded_files)} document(s) with {chunking_strategy} chunker..."):
                    use_mem = (storage_mode == "In-Memory")
                    st.session_state.doc_tool = DocumentSearchTool(
                        file_paths=saved_paths,
                        use_memory=use_mem,
                        chunk_size=chunk_size,
                        chunking_strategy=chunking_strategy,
                        top_k=top_k,
                        enable_rerank=enable_rerank
                    )
                    st.session_state.indexed_files = [f.name for f in uploaded_files]

            st.success(f"✓ Indexed: {', '.join(st.session_state.indexed_files)}")
    else:
        # Load sample knowledge file if available
        sample_doc = BASE_DIR / "knowledge" / "dspy.pdf"
        if sample_doc.exists() and not st.session_state.indexed_files:
            if st.button("Load sample knowledge (dspy.pdf)"):
                with st.spinner("Indexing sample document..."):
                    use_mem = (storage_mode == "In-Memory")
                    st.session_state.doc_tool = DocumentSearchTool(
                        file_path=str(sample_doc),
                        use_memory=use_mem,
                        chunk_size=chunk_size,
                        chunking_strategy=chunking_strategy,
                        top_k=top_k,
                        enable_rerank=enable_rerank
                    )
                    st.session_state.indexed_files = ["dspy.pdf"]
                st.success("Loaded and indexed dspy.pdf sample!")

    if st.session_state.indexed_files:
        st.write("**Currently Indexed Files:**")
        for f in st.session_state.indexed_files:
            st.caption(f"📄 {f}")
        if st.button("Reset Knowledge Base", use_container_width=True):
            clear_vector_index()
            st.rerun()

    st.divider()
    st.button("Clear Chat History", on_click=reset_chat, use_container_width=True)

# ===========================
#   Main Interface
# ===========================
crewai_logo_path = BASE_DIR / "assets" / "crewai.png"
if crewai_logo_path.exists():
    encoded_logo = base64.b64encode(open(crewai_logo_path, "rb").read()).decode()
    st.markdown(f"""
        # ContextForge <img src="data:image/png;base64,{encoded_logo}" width="110" style="vertical-align: -3px;">
    """, unsafe_allow_html=True)
else:
    st.markdown("# ContextForge - Agentic RAG")

active_docs_label = ", ".join(st.session_state.indexed_files) if st.session_state.indexed_files else "None (Web Search fallback only)"
st.caption(f"Active Model: **{model_choice}** | Search: **{search_provider}** | Chunks: **{chunking_strategy} ({chunk_size})** | Rerank: **{'On' if enable_rerank else 'Off'}** | Docs: **{active_docs_label}**")

# Render conversation history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
prompt = st.chat_input("Ask a question about your indexed documents or query the web...")

if prompt:
    # 1. Show user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Build Crew dynamically with selected LLM, Search tool, and Document tool
    with st.chat_message("assistant"):
        message_placeholder = st.empty()

        with st.spinner("Analyzing document and web sources..."):
            try:
                selected_llm = get_llm_instance(model_choice, custom_model_name, ollama_url)
                provider_key = "auto" if search_provider == "Auto-detect" else search_provider.lower()
                web_tool = get_web_search_tool(provider=provider_key)

                # Initialize unified CrewBase crew
                agentic_crew = AgenticRag(
                    pdf_tool=st.session_state.doc_tool,
                    web_search_tool=web_tool,
                    llm=selected_llm
                ).crew()

                inputs = {"query": prompt}
                result = agentic_crew.kickoff(inputs=inputs).raw

                # Render with streaming simulation effect
                full_response = ""
                lines = str(result).split('\n')
                for i, line in enumerate(lines):
                    full_response += line
                    if i < len(lines) - 1:
                        full_response += '\n'
                    message_placeholder.markdown(full_response + "▌")
                    time.sleep(0.03)

                message_placeholder.markdown(full_response)
                st.session_state.messages.append({"role": "assistant", "content": full_response})

            except Exception as e:
                err_msg = f"**Error executing query:** {str(e)}"
                message_placeholder.error(err_msg)
                st.session_state.messages.append({"role": "assistant", "content": err_msg})
