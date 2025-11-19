# Local RAG Pipeline

This project implements a RAG (Retrieval-Augmented Generation) pipeline using local embeddings and LLMs via LM Studio.

## Setup

1.  **Dependencies**: Ensure you have the required packages installed.
    ```bash
    pip install langchain-community langchain-openai langchain-text-splitters chromadb datasets
    ```

2.  **LM Studio**: 
    - Start LM Studio.
    - Load an embedding model (and an LLM if using generation).
    - Start the Local Server on port `1234`.

## Usage

**1. Ingest Data**
Load data from Hugging Face dataset (`yixuantt/MultiHopRAG`), generate embeddings, and store them in the vector database.
```bash
python -m rag_pipeline.main ingest
```

**2. Query**
Search for relevant text in the vector store.
```bash
python -m rag_pipeline.main query "What are the best Cyber Monday deals?"
```

**3. Query & Generate**
Search and generate an answer using the local LLM.
```bash
python -m rag_pipeline.main query "What are the best Cyber Monday deals?" --generate
```

## Configuration
Edit `rag_pipeline/config.py` to change:
- Dataset Name/Subset
- Embedding API endpoint
- Vector store path
