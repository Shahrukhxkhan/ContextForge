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

if "pdf_tool" not in st.session_state:
    st.session_state.pdf_tool = None

if "current_file_name" not in st.session_state:
    st.session_state.current_file_name = None

def reset_chat():
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

def display_pdf(file_bytes: bytes, file_name: str):
    """Displays the uploaded PDF in an iframe."""
    base64_pdf = base64.b64encode(file_bytes).decode("utf-8")
    pdf_display = f"""
    <iframe 
        src="data:application/pdf;base64,{base64_pdf}" 
        width="100%" 
        height="500px" 
        type="application/pdf"
    >
    </iframe>
    """
    st.markdown(f"**Preview: {file_name}**")
    st.markdown(pdf_display, unsafe_allow_html=True)

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
            custom_model_name = st.text_input("Custom Ollama Model Name (e.g. mistral, qwen2.5)", value="")

    st.subheader("2. Web Search Provider")
    search_provider = st.selectbox(
        "Search Tool Fallback",
        ["Auto-detect", "Serper", "Firecrawl"],
        index=0
    )

    st.divider()

    st.subheader("3. Knowledge Base")
    uploaded_file = st.file_uploader("Upload a PDF Document", type=["pdf"])

    if uploaded_file is not None:
        if st.session_state.current_file_name != uploaded_file.name:
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_file_path = os.path.join(temp_dir, uploaded_file.name)
                with open(temp_file_path, "wb") as f:
                    f.write(uploaded_file.getvalue())

                with st.spinner(f"Indexing '{uploaded_file.name}' into vector memory..."):
                    st.session_state.pdf_tool = DocumentSearchTool(file_path=temp_file_path)
                    st.session_state.current_file_name = uploaded_file.name

            st.success(f"✓ '{uploaded_file.name}' indexed!")

        with st.expander("Preview Document", expanded=False):
            display_pdf(uploaded_file.getvalue(), uploaded_file.name)
    else:
        # Default bundled knowledge doc fallback if available
        default_doc = BASE_DIR / "knowledge" / "dspy.pdf"
        if default_doc.exists() and st.session_state.pdf_tool is None:
            if st.button("Load default sample doc (dspy.pdf)"):
                with st.spinner("Indexing default knowledge document..."):
                    st.session_state.pdf_tool = DocumentSearchTool(file_path=str(default_doc))
                    st.session_state.current_file_name = "dspy.pdf"
                st.success("Loaded dspy.pdf sample!")

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

st.caption(f"Active Model: **{model_choice}** | Active Search: **{search_provider}** | Knowledge: **{st.session_state.current_file_name or 'No PDF loaded (Web Search only)'}**")

# Render conversation history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
prompt = st.chat_input("Ask a question about your document or query the web...")

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
                    pdf_tool=st.session_state.pdf_tool,
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
                    time.sleep(0.04)

                message_placeholder.markdown(full_response)
                st.session_state.messages.append({"role": "assistant", "content": full_response})

            except Exception as e:
                err_msg = f"**Error executing query:** {str(e)}"
                message_placeholder.error(err_msg)
                st.session_state.messages.append({"role": "assistant", "content": err_msg})
