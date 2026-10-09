# ContextForge

An agentic RAG system powered by CrewAI that searches through your docs first and falls back to web search when needed, with support for running local models like DeepSeek-R1 or Llama 3.2.

Before that, make sure you grab your FireCrawl API keys to search the web.

**Get API Keys**:
   - [FireCrawl](https://www.firecrawl.dev/i/api)



## Installation and setup

**Get API Keys**:
   - [FireCrawl](https://www.firecrawl.dev/i/api)


**Install Dependencies**:
   Ensure you have Python 3.11 or later installed.
   ```bash
   pip install crewai crewai-tools chonkie[semantic] markitdown qdrant-client fastembed
   ```

**Running the app**:

Launch the unified Streamlit interface:
```bash
streamlit run app.py
```

From the sidebar in the app, you can:
- **Select LLM:** Choose between OpenAI (default), Ollama local models (`deepseek-r1:7b`, `llama3.2`), or any custom Ollama model.
- **Select Search Tool:** Toggle between Serper and FireCrawl (with auto-detection based on configured keys).
- **Upload Knowledge:** Upload your PDF document to index it into local vector memory.

---

## 📬 Stay Updated with Our Newsletter!
**Get a FREE Data Science eBook** 📖 with 150+ essential lessons in Data Science when you subscribe to our newsletter! Stay in the loop with the latest tutorials, insights, and exclusive resources. [Subscribe now!](https://join.dailydoseofds.com)

[![Daily Dose of Data Science Newsletter](https://github.com/patchy631/ai-engineering/blob/main/resources/join_ddods.png)](https://join.dailydoseofds.com)

---

## Contribution

Contributions are welcome! Please fork the repository and submit a pull request with your improvements.
