import os

# Using an absolute path or relative to where the script is run. 
VECTOR_STORE_PATH = os.path.join(os.path.dirname(__file__), "chroma_db")
COLLECTION_NAME = "rag_collection"

# Embedding Settings - Local LM Studio
# EMBEDDING_API_BASE = "http://localhost:1234/v1" #lmstudio
EMBEDDING_API_BASE = "http://localhost:11434/v1"
LLM_MODEL_NAME = "gemma3:27b"
EMBEDDING_API_KEY = "lm-studio"
# EMBEDDING_MODEL_NAME = "qwen3-embedding:latest" 
EMBEDDING_MODEL_NAME = "nomic-embed-text" 

# Source Data
DATASET_NAME = "yixuantt/MultiHopRAG"
DATASET_SUBSET = "corpus"

