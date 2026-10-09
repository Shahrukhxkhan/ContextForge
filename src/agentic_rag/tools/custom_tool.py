import os
import hashlib
from typing import Type, List, Optional, Union
from pathlib import Path
from pydantic import BaseModel, Field, ConfigDict
from crewai.tools import BaseTool
from markitdown import MarkItDown
from chonkie import SemanticChunker, RecursiveChunker, SentenceChunker
from qdrant_client import QdrantClient

# Default storage directory for persistent vector store
DEFAULT_STORAGE_PATH = str(Path(__file__).resolve().parent.parent.parent.parent / "data" / "qdrant_db")

class DocumentSearchToolInput(BaseModel):
    """Input schema for DocumentSearchTool."""
    query: str = Field(..., description="Query to search the documents.")

class DocumentSearchTool(BaseTool):
    name: str = "DocumentSearchTool"
    description: str = "Search indexed documents using hybrid semantic retrieval and reranking for the given query."
    args_schema: Type[BaseModel] = DocumentSearchToolInput

    model_config = ConfigDict(extra="allow")

    def __init__(
        self,
        file_paths: Optional[Union[str, List[str]]] = None,
        file_path: Optional[str] = None,
        pdf: Optional[str] = None,
        collection_name: str = "contextforge_knowledge",
        storage_path: Optional[str] = None,
        use_memory: bool = False,
        chunk_size: int = 512,
        chunking_strategy: str = "semantic",  # 'semantic', 'recursive', 'sentence'
        top_k: int = 5,
        enable_rerank: bool = True
    ):
        """
        Enhanced DocumentSearchTool:
        - Multi-file support (.pdf, .docx, .pptx, .xlsx, .html, .md, .txt, .csv) via MarkItDown
        - Persistent local disk storage or in-memory fallback
        - Configurable chunking (semantic, recursive, sentence)
        - FastEmbed cross-encoder reranking
        """
        super().__init__()

        # Resolve paths
        paths: List[str] = []
        if file_paths:
            if isinstance(file_paths, list):
                paths.extend(file_paths)
            else:
                paths.append(file_paths)
        if file_path:
            paths.append(file_path)
        if pdf:
            paths.append(pdf)

        self.file_paths = [str(p) for p in paths if p and os.path.exists(p)]
        self.collection_name = collection_name
        self.chunk_size = chunk_size
        self.chunking_strategy = chunking_strategy
        self.top_k = top_k
        self.enable_rerank = enable_rerank

        # Setup Qdrant Client (Persistent by default, or memory if requested)
        if use_memory:
            self.client = QdrantClient(":memory:")
        else:
            db_dir = storage_path or os.getenv("QDRANT_STORAGE_PATH", DEFAULT_STORAGE_PATH)
            os.makedirs(db_dir, exist_ok=True)
            self.client = QdrantClient(path=db_dir)

        # Initialize MarkItDown parser
        self.md = MarkItDown()

        # Initialize reranker lazily
        self._reranker = None

        # Process any provided files
        if self.file_paths:
            self.add_documents(self.file_paths)

    def _get_reranker(self):
        if self._reranker is None and self.enable_rerank:
            try:
                from fastembed.rerank.cross_encoder import TextCrossEncoder
                self._reranker = TextCrossEncoder(model_name="Xenova/ms-marco-MiniLM-L-6-v2")
            except Exception:
                self._reranker = None
        return self._reranker

    def _extract_text(self, file_path: str) -> str:
        """Extract markdown/text from supported document formats using MarkItDown."""
        try:
            result = self.md.convert(file_path)
            return result.text_content or ""
        except Exception as e:
            # Fallback for plain text/markdown if MarkItDown fails
            try:
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    return f.read()
            except Exception:
                return f"[Error extracting text from {file_path}: {str(e)}]"

    def _create_chunks(self, raw_text: str) -> list:
        """Create chunks using chosen chunking strategy."""
        if not raw_text.strip():
            return []

        strategy = self.chunking_strategy.lower()
        if strategy == "recursive":
            chunker = RecursiveChunker(chunk_size=self.chunk_size)
        elif strategy == "sentence":
            chunker = SentenceChunker(chunk_size=self.chunk_size)
        else:
            # Semantic chunking (default)
            chunker = SemanticChunker(
                embedding_model="minishlab/potion-base-8M",
                threshold=0.5,
                chunk_size=self.chunk_size,
                min_sentences=1
            )
        return chunker.chunk(raw_text)

    def add_documents(self, file_paths: Union[str, List[str]]):
        """Index multiple documents into the Qdrant collection with deduplication."""
        paths = [file_paths] if isinstance(file_paths, str) else file_paths

        all_docs = []
        all_metadata = []
        all_ids = []

        for fpath in paths:
            if not os.path.isfile(fpath):
                continue
            
            fname = os.path.basename(fpath)
            raw_text = self._extract_text(fpath)
            if not raw_text.strip():
                continue

            chunks = self._create_chunks(raw_text)
            for idx, chunk in enumerate(chunks):
                chunk_text = getattr(chunk, "text", str(chunk))
                # Deterministic ID using file name and chunk index to avoid duplicates on re-index
                hash_id = hashlib.md5(f"{fname}_{idx}_{chunk_text[:50]}".encode()).hexdigest()
                
                all_docs.append(chunk_text)
                all_metadata.append({
                    "source": fname,
                    "file_path": fpath,
                    "chunk_index": idx
                })
                all_ids.append(hash_id)

        if all_docs:
            self.client.add(
                collection_name=self.collection_name,
                documents=all_docs,
                metadata=all_metadata,
                ids=all_ids
            )

    def _run(self, query: str) -> str:
        """Retrieve most relevant chunks with optional cross-encoder reranking."""
        try:
            # Retrieve initial candidate pool
            fetch_k = max(self.top_k * 3, 10)
            candidate_chunks = self.client.query(
                collection_name=self.collection_name,
                query_text=query,
                limit=fetch_k
            )
        except Exception as e:
            return f"Error querying vector collection: {str(e)}"

        if not candidate_chunks:
            return "No relevant information found in the documents."

        docs = [chunk.document for chunk in candidate_chunks if getattr(chunk, "document", None)]
        sources = [
            chunk.metadata.get("source", "Unknown") if hasattr(chunk, "metadata") and chunk.metadata else "Unknown"
            for chunk in candidate_chunks
        ]

        if not docs:
            return "No matching document contents found."

        # Rerank if enabled
        reranker = self._get_reranker()
        if reranker and len(docs) > 1:
            try:
                scores = list(reranker.rerank(query, docs))
                # Sort indices descending by score
                ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:self.top_k]
                selected_docs = [docs[i] for i in ranked_indices]
                selected_sources = [sources[i] for i in ranked_indices]
            except Exception:
                selected_docs = docs[:self.top_k]
                selected_sources = sources[:self.top_k]
        else:
            selected_docs = docs[:self.top_k]
            selected_sources = sources[:self.top_k]

        # Format output with source metadata
        formatted_results = []
        for doc, src in zip(selected_docs, selected_sources):
            formatted_results.append(f"[Source: {src}]\n{doc}")

        return "\n\n---\n\n".join(formatted_results)


class FireCrawlWebSearchToolInput(BaseModel):
    """Input schema for FireCrawlWebSearchTool."""
    query: str = Field(..., description="Query to search the web using FireCrawl.")

class FireCrawlWebSearchTool(BaseTool):
    name: str = "FireCrawlWebSearchTool"
    description: str = "Search the web using FireCrawl for a query when information is not in the document."
    args_schema: Type[BaseModel] = FireCrawlWebSearchToolInput

    model_config = ConfigDict(extra="allow")

    def __init__(self, api_key: str = None):
        super().__init__()
        self.api_key = api_key or os.getenv("FIRECRAWL_API_KEY")

    def _run(self, query: str) -> str:
        """Execute web search using FirecrawlSearchTool if available or FirecrawlApp directly."""
        try:
            from crewai_tools import FirecrawlSearchTool
            tool = FirecrawlSearchTool(api_key=self.api_key) if self.api_key else FirecrawlSearchTool()
            return str(tool._run(query=query))
        except Exception as e:
            return f"Error executing FireCrawl web search: {str(e)}. Please verify your FIRECRAWL_API_KEY in .env."

def get_web_search_tool(provider: str = "auto"):
    """
    Factory helper to return an initialized web search tool.
    Supports 'serper', 'firecrawl', or 'auto' (detects based on available API keys).
    """
    provider_lower = (provider or "auto").lower()

    if provider_lower == "firecrawl":
        return FireCrawlWebSearchTool()
    elif provider_lower == "serper":
        from crewai_tools import SerperDevTool
        return SerperDevTool()
    else:
        if os.getenv("SERPER_API_KEY"):
            from crewai_tools import SerperDevTool
            return SerperDevTool()
        elif os.getenv("FIRECRAWL_API_KEY"):
            return FireCrawlWebSearchTool()
        else:
            from crewai_tools import SerperDevTool
            return SerperDevTool()
