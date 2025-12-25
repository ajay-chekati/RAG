import os

EMBEDDING_API_BASE = "http://localhost:11434/v1"
EMBEDDING_API_KEY = "lm-studio"

EMBEDDING_MODEL_NAME = "qwen3-embedding:latest"
LLM_MODEL_NAME = "llama3.1:70b"

VECTOR_DB_BACKEND = "chroma"
VECTOR_STORE_BASE_DIR = os.path.join(os.path.dirname(__file__), "vector_stores")
COLLECTION_NAME = "rag_collection"


def get_vector_store_path(backend: str = None, embedding_model: str = None) -> str:
    """Generate vector store path based on backend and embedding model."""
    backend = backend or VECTOR_DB_BACKEND
    embedding_model = embedding_model or EMBEDDING_MODEL_NAME
    sanitized_model = embedding_model.replace(":", "_").replace("/", "_").replace("\\", "_")
    dir_name = f"{backend}_{sanitized_model}"
    return os.path.join(VECTOR_STORE_BASE_DIR, dir_name)


VECTOR_STORE_PATH = get_vector_store_path()

DATASET_NAME = "yixuantt/MultiHopRAG"
DATASET_SUBSET = "corpus"
