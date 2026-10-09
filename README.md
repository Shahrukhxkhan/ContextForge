# ContextForge

An agentic Retrieval-Augmented Generation (RAG) system powered by **CrewAI**, **Qdrant**, and **MarkItDown**. ContextForge intelligently analyzes user queries, searches indexed multi-format documents first, and dynamically falls back to web search when needed. It supports cloud models (OpenAI) as well as 100% local models (Ollama: DeepSeek-R1, Llama 3.2, etc.).

---

## ✨ Features

- **Agentic Multi-Stage Pipeline:** Router $\rightarrow$ Precision Retriever $\rightarrow$ Synthesizer $\rightarrow$ Hallucination Grader / Groundedness Verifier.
- **Multi-Format Ingestion:** Process `.pdf`, `.docx`, `.pptx`, `.xlsx`, `.html`, `.md`, `.txt`, and `.csv` files using MarkItDown.
- **Configurable Chunking:** Semantic chunking via Chonkie, plus recursive and sentence splitters.
- **Persistent Vector Storage:** Local vector search powered by Qdrant (saved to disk, no re-embedding on restarts).
- **Cross-Encoder Re-ranking:** Re-ranks candidate passages with FastEmbed (`ms-marco-MiniLM-L-6-v2`) before synthesis.
- **Web Search Fallback:** Configurable live web retrieval via Serper or FireCrawl.
- **Chat Management:** Multiple conversation sessions with one-click export to Markdown or JSON.

---

## 🚀 Installation and Setup

### 1. Clone the Repository
```bash
git clone https://github.com/Shahrukhxkhan/ContextForge.git
cd ContextForge
```

### 2. Configure Environment Variables
Copy the example configuration:
```bash
cp .env.example .env
```
Fill in your API keys in `.env`:
- [Serper API Key](https://serper.dev) *(Optional, for Google search fallback)*
- [FireCrawl API Key](https://www.firecrawl.dev) *(Optional, for web search fallback)*
- `OPENAI_API_KEY` *(Only required if using OpenAI cloud models)*

### 3. Install Dependencies
Ensure you have Python 3.10 to 3.13 installed:
```bash
pip install -e .
```
Or install directly:
```bash
pip install "crewai[tools]" streamlit markitdown "chonkie[semantic]" qdrant-client fastembed python-dotenv
```

---

## 💻 Running the App

### Web Application (Streamlit)
Launch the unified Streamlit interface:
```bash
streamlit run app.py
```

From the app interface:
1. **Model Selection:** Choose OpenAI (default) or Local Ollama (`deepseek-r1:7b`, `llama3.2`, or custom model).
2. **Web Search Tool:** Choose between Serper, FireCrawl, or Auto-detect.
3. **Upload Knowledge:** Drag and drop documents (PDF, DOCX, TXT, etc.) into the sidebar.
4. **Chat & Verify:** Ask questions with real-time agent execution status and view cited sources in the grounding drawer.

### CLI Mode (Terminal)
Run queries directly from the command line:
```bash
python src/agentic_rag/main.py "What is DSPy?"
```

---

## 🤝 Contribution

Contributions are welcome! Please fork the repository, create a feature branch, and submit a pull request.

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
