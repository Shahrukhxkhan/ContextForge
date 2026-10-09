import os
from crewai.tools import BaseTool
from typing import Type
from pydantic import BaseModel, Field, ConfigDict
from markitdown import MarkItDown
from chonkie import SemanticChunker
from qdrant_client import QdrantClient

class DocumentSearchToolInput(BaseModel):
    """Input schema for DocumentSearchTool."""
    query: str = Field(..., description="Query to search the document.")

class DocumentSearchTool(BaseTool):
    name: str = "DocumentSearchTool"
    description: str = "Search the document for the given query."
    args_schema: Type[BaseModel] = DocumentSearchToolInput
    
    def __init__(self, file_path: str = None, pdf: str = None):
        """Initialize the searcher with a PDF file path and set up the Qdrant collection."""
        super().__init__()
        self.file_path = file_path or pdf
        if not self.file_path:
            raise ValueError("Either file_path or pdf must be provided to DocumentSearchTool.")
        self.client = QdrantClient(":memory:")  # For small experiments
        self._process_document()

    def _extract_text(self) -> str:
        """Extract raw text from PDF using MarkItDown."""
        md = MarkItDown()
        result = md.convert(self.file_path)
        return result.text_content

    def _create_chunks(self, raw_text: str) -> list:
        """Create semantic chunks from raw text."""
        chunker = SemanticChunker(
            embedding_model="minishlab/potion-base-8M",
            threshold=0.5,
            chunk_size=512,
            min_sentences=1
        )
        return chunker.chunk(raw_text)

    def _process_document(self):
        """Process the document and add chunks to Qdrant collection."""
        raw_text = self._extract_text()
        chunks = self._create_chunks(raw_text)
        
        docs = [chunk.text for chunk in chunks]
        metadata = [{"source": os.path.basename(self.file_path)} for _ in range(len(chunks))]
        ids = list(range(len(chunks)))

        self.client.add(
            collection_name="demo_collection",
            documents=docs,
            metadata=metadata,
            ids=ids
        )

    def _run(self, query: str) -> list:
        """Search the document with a query string."""
        relevant_chunks = self.client.query(
            collection_name="demo_collection",
            query_text=query
        )
        docs = [chunk.document for chunk in relevant_chunks]
        separator = "\n___\n"
        return separator.join(docs)

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
            # Fallback or informative error if FireCrawl API is unconfigured/fails
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
        # Auto-detect priority: Serper if key present, else Firecrawl if key present, else SerperDevTool
        if os.getenv("SERPER_API_KEY"):
            from crewai_tools import SerperDevTool
            return SerperDevTool()
        elif os.getenv("FIRECRAWL_API_KEY"):
            return FireCrawlWebSearchTool()
        else:
            from crewai_tools import SerperDevTool
            return SerperDevTool()


# Test the implementation
def test_document_searcher():
    # Resolve knowledge/dspy.pdf dynamically relative to repository root
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    pdf_path = os.path.join(base_dir, "knowledge", "dspy.pdf")
    
    # Create instance
    searcher = DocumentSearchTool(file_path=pdf_path)
    
    # Test search
    result = searcher._run("What is the purpose of DSpy?")
    print("Search Results:", result)

if __name__ == "__main__":
    test_document_searcher()

