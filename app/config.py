
"""
Central place for all settings. Everything else in the app imports from here.
"""
import os
from dotenv import load_dotenv

load_dotenv()  # reads .env in the project root

# ---- API Keys ----
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")

# ---- LangSmith tracing ----
# LangChain reads these directly from os.environ, so we set them here,
# as early as possible, before any LangChain objects are created elsewhere.
if LANGSMITH_API_KEY:
    tracing_value = os.getenv("LANGSMITH_TRACING", "true")

    os.environ["LANGSMITH_TRACING"] = tracing_value
    os.environ["LANGSMITH_API_KEY"] = LANGSMITH_API_KEY
    os.environ["LANGSMITH_PROJECT"] = os.getenv("LANGSMITH_PROJECT", "arxiv-rag-bot")

    # Older naming scheme, kept for compatibility
    os.environ["LANGCHAIN_TRACING_V2"] = tracing_value
    os.environ["LANGCHAIN_API_KEY"] = LANGSMITH_API_KEY
    os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGSMITH_PROJECT", "arxiv-rag-bot")

# ---- Pinecone ----
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "arxiv-rag-bot")

# ---- Embedding model ----
EMBEDDING_MODEL_NAME = "BAAI/bge-large-en-v1.5"
EMBEDDING_DIMENSIONS = 1024

# ---- LLM (via OpenRouter, OpenAI-compatible endpoint) ----
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
LLM_MODEL_NAME = "openai/gpt-4o-mini"
LLM_TEMPERATURE = 0

# ---- Chunking ----
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

# ---- Retrieval ----
RETRIEVAL_K = 25
FINAL_TOP_K = 5

# ---- Paths ----
PAPERS_DIR = "data/papers"