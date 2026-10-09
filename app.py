import streamlit as st
import os
import sys
import tempfile
import gc
import json
import base64
import time
from datetime import datetime
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

SUPPORTED_EXTENSIONS = ["pdf", "docx", "pptx", "xlsx", "html", "md", "txt", "csv"]

# ===========================
#   Streamlit Page Config
# ===========================
st.set_page_config(
    page_title="ContextForge - Agentic RAG",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for polished aesthetics
st.markdown("""
<style>
    .stChatFloatingInputContainer { bottom: 20px; }
    .status-badge {
        display: inline-block;
        padding: 0.25rem 0.5rem;
        border-radius: 4px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-right: 0.5rem;
    }
    .badge-doc { background-color: #1e3a8a; color: #bfdbfe; }
    .badge-web { background-color: #065f46; color: #a7f3d0; }
    .citation-box {
        background-color: rgba(255, 255, 255, 0.05);
        border-left: 3px solid #3b82f6;
        padding: 0.6rem 0.8rem;
        margin-top: 0.5rem;
        border-radius: 0 4px 4px 0;
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)

# ===========================
#   Session State Setup
# ===========================
if "sessions" not in st.session_state:
    st.session_state.sessions = {
        "Default Session": []
    }

if "active_session_name" not in st.session_state:
    st.session_state.active_session_name = "Default Session"

if "doc_tool" not in st.session_state:
    st.session_state.doc_tool = None

if "indexed_files" not in st.session_state:
    st.session_state.indexed_files = []

# Helper getters
def get_current_messages():
    name = st.session_state.active_session_name
    if name not in st.session_state.sessions:
        st.session_state.sessions[name] = []
    return st.session_state.sessions[name]

def clear_active_chat():
    st.session_state.sessions[st.session_state.active_session_name] = []
    gc.collect()

def clear_vector_index():
    st.session_state.doc_tool = None
    st.session_state.indexed_files = []
    gc.collect()

def get_llm_instance(model_choice: str, custom_model_name: str, ollama_url: str):
    """Factory to get the selected LLM instance."""
    if model_choice == "OpenAI / Default (API Key)":
        return None
    elif model_choice == "Ollama: DeepSeek-R1 (7B)":
        return LLM(model="ollama/deepseek-r1:7b", base_url=ollama_url)
    elif model_choice == "Ollama: Llama 3.2":
        return LLM(model="ollama/llama3.2", base_url=ollama_url)
    elif model_choice == "Ollama: Custom":
        clean_name = custom_model_name.strip()
        model_tag = f"ollama/{clean_name}" if not clean_name.startswith("ollama/") else clean_name
        return LLM(model=model_tag, base_url=ollama_url)
    return None

def extract_citations_from_text(text: str) -> list:
    """Extract sources and references from output text."""
    lines = text.split("\n")
    citations = []
    in_source_block = False
    for line in lines:
        line_clean = line.strip()
        if "### Sources" in line or "#### Verification Badge" in line:
            in_source_block = True
        if in_source_block and line_clean:
            citations.append(line_clean)
        elif "[Source:" in line_clean or "[Web:" in line_clean:
            citations.append(line_clean)
    return citations

# ===========================
#   Sidebar Controls
# ===========================
with st.sidebar:
    st.title("⚡ ContextForge")

    # 1. Chat Sessions Management
    st.subheader("💬 Chat Sessions")
    session_names = list(st.session_state.sessions.keys())
    selected_sess = st.selectbox(
        "Current Session",
        session_names,
        index=session_names.index(st.session_state.active_session_name)
    )
    st.session_state.active_session_name = selected_sess

    col1, col2 = st.columns([3, 1])
    with col1:
        new_sess_input = st.text_input("New Session Name", placeholder="e.g. Research DSPy", label_visibility="collapsed")
    with col2:
        if st.button("➕", help="Create new session") and new_sess_input.strip():
            clean_new = new_sess_input.strip()
            if clean_new not in st.session_state.sessions:
                st.session_state.sessions[clean_new] = []
                st.session_state.active_session_name = clean_new
                st.rerun()

    # Export chat history options
    curr_messages = get_current_messages()
    if curr_messages:
        md_export = f"# ContextForge Chat Export - {st.session_state.active_session_name}\n*Exported: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n"
        for m in curr_messages:
            md_export += f"### {m['role'].capitalize()}\n{m['content']}\n\n"

        json_export = json.dumps(curr_messages, indent=2)

        exp_col1, exp_col2 = st.columns(2)
        with exp_col1:
            st.download_button(
                "📥 Markdown",
                data=md_export,
                file_name=f"chat_{st.session_state.active_session_name.lower().replace(' ', '_')}.md",
                mime="text/markdown",
                use_container_width=True
            )
        with exp_col2:
            st.download_button(
                "📥 JSON",
                data=json_export,
                file_name=f"chat_{st.session_state.active_session_name.lower().replace(' ', '_')}.json",
                mime="application/json",
                use_container_width=True
            )

    st.divider()

    # 2. Model & Search
    st.subheader("⚙️ Model & Search")
    model_choice = st.selectbox(
        "LLM Model",
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
            custom_model_name = st.text_input("Custom Model Name", value="")

    search_provider = st.selectbox(
        "Web Search Tool",
        ["Auto-detect", "Serper", "Firecrawl"],
        index=0
    )

    st.divider()

    # 3. RAG Settings
    st.subheader("📚 Retrieval Pipeline")
    chunking_strategy = st.selectbox(
        "Chunking Strategy",
        ["semantic", "recursive", "sentence"],
        index=0
    )
    chunk_size = st.slider("Chunk Size", min_value=128, max_value=1024, value=512, step=64)
    top_k = st.slider("Top Chunks (k)", min_value=1, max_value=10, value=5)
    enable_rerank = st.checkbox("Cross-Encoder Re-ranking", value=True)
    enable_verification = st.checkbox("Hallucination Grader Badge", value=True)
    storage_mode = st.radio("Storage Mode", ["Persistent Disk", "In-Memory"], index=0, horizontal=True)

    st.divider()

    # 4. Knowledge Documents
    st.subheader("📁 Documents")
    uploaded_files = st.file_uploader(
        "Upload files",
        type=SUPPORTED_EXTENSIONS,
        accept_multiple_files=True,
        help="Upload .pdf, .docx, .pptx, .xlsx, .html, .md, .txt, .csv"
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

                with st.spinner(f"Indexing {len(uploaded_files)} document(s)..."):
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

            st.success(f"✓ Indexed {len(st.session_state.indexed_files)} files")
    else:
        sample_doc = BASE_DIR / "knowledge" / "dspy.pdf"
        if sample_doc.exists() and not st.session_state.indexed_files:
            if st.button("Load sample (dspy.pdf)"):
                with st.spinner("Indexing sample knowledge..."):
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
                st.success("Loaded sample doc!")

    if st.session_state.indexed_files:
        st.write("**Indexed:**")
        for f in st.session_state.indexed_files:
            st.caption(f"📄 {f}")
        if st.button("Clear Documents", use_container_width=True):
            clear_vector_index()
            st.rerun()

    st.divider()
    if st.button("Clear Chat", on_click=clear_active_chat, use_container_width=True):
        st.rerun()

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

active_docs_label = ", ".join(st.session_state.indexed_files) if st.session_state.indexed_files else "None (Web Search fallback)"
st.caption(f"Session: **{st.session_state.active_session_name}** | Model: **{model_choice}** | Chunker: **{chunking_strategy}** | Knowledge: **{active_docs_label}**")

# Display current chat conversation with expandable source citations
for message in get_current_messages():
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("citations"):
            with st.expander("🔍 Sources & Grounding Details", expanded=False):
                for cit in message["citations"]:
                    st.markdown(f"- {cit}")

# Chat input
prompt = st.chat_input("Ask a question about your documents or search the web...")

if prompt:
    # 1. Record & render user query
    current_chat = get_current_messages()
    current_chat.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Execute with live agent thought progression in st.status
    with st.chat_message("assistant"):
        message_placeholder = st.empty()

        with st.status("🧠 Agents collaborating on query...", expanded=True) as status_box:
            status_box.write("📍 **Router Agent**: Analyzing query intent & formulating retrieval strategy...")

            # Callback hooks for agent task updates
            def on_step_callback(step_output):
                try:
                    tool_name = getattr(step_output, 'tool', 'Agent')
                    status_box.write(f"⚙️ **{tool_name}** executed: analyzing findings...")
                except Exception:
                    pass

            def on_task_callback(task_output):
                try:
                    desc = getattr(task_output, 'description', '')[:60]
                    status_box.write(f"✓ Completed stage: *{desc}...*")
                except Exception:
                    pass

            try:
                selected_llm = get_llm_instance(model_choice, custom_model_name, ollama_url)
                provider_key = "auto" if search_provider == "Auto-detect" else search_provider.lower()
                web_tool = get_web_search_tool(provider=provider_key)

                status_box.write("🔍 **Retriever Agent**: Inspecting vector documents & web search fallback...")

                agentic_crew = AgenticRag(
                    pdf_tool=st.session_state.doc_tool,
                    web_search_tool=web_tool,
                    llm=selected_llm,
                    enable_verification=enable_verification,
                    step_callback=on_step_callback,
                    task_callback=on_task_callback
                ).crew()

                inputs = {"query": prompt}
                status_box.write("✍️ **Synthesizer Agent**: Crafting grounded response with source attribution...")

                result = agentic_crew.kickoff(inputs=inputs).raw

                status_box.update(label="✅ Query completed and verified!", state="complete", expanded=False)

                # Simulated real-time typing effect
                full_response = ""
                lines = str(result).split('\n')
                for i, line in enumerate(lines):
                    full_response += line
                    if i < len(lines) - 1:
                        full_response += '\n'
                    message_placeholder.markdown(full_response + "▌")
                    time.sleep(0.02)

                message_placeholder.markdown(full_response)

                # Extract citations for dedicated drawer
                citations = extract_citations_from_text(full_response)
                if citations:
                    with st.expander("🔍 Sources & Grounding Details", expanded=False):
                        for cit in citations:
                            st.markdown(f"- {cit}")

                current_chat.append({
                    "role": "assistant",
                    "content": full_response,
                    "citations": citations
                })

            except Exception as e:
                status_box.update(label="❌ Error executing query", state="error", expanded=True)
                err_msg = f"**Error executing query:** {str(e)}"
                message_placeholder.error(err_msg)
                current_chat.append({"role": "assistant", "content": err_msg, "citations": []})
