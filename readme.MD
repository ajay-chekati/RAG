# Local RAG Pipeline

RAG pipeline with local embeddings/LLMs via Ollama. Includes evaluation framework with caching for benchmarking on MultiHop-RAG dataset.

## Setup

1.  **Dependencies**: Ensure you have the required packages installed.
    ```bash
    pip install langchain-community langchain-openai langchain-text-splitters chromadb datasets tqdm
    ```

2.  **LM Studio / Ollama**: 
    - Start LM Studio or Ollama.
    - Load an embedding model (and an LLM for generation/evaluation).
    - Ensure the API is running and update the API endpoint in `config.py` (default: `http://localhost:11434/v1` for Ollama or `http://localhost:1234/v1` for LM Studio).

3.  **Ingest Documents**: 
    Load the corpus, generate embeddings, and store in the vector database. **Important**: Use the same embedding model in `config.py` for both ingestion and querying—mismatched models cause dimension errors.
    ```bash
    python -m rag_pipeline.main ingest
    ```
    Vector stores are auto-created at `vector_stores/{backend}_{embedding_model}/`.

## Evaluation Framework

The evaluation framework allows you to automatically benchmark your RAG pipeline across different configurations and measure both retrieval quality and generation accuracy.

### Running Evaluations

**Full evaluation with caching (default):**
```bash
python -m rag_pipeline.run_experiments --limit 100
```

**Run without caching (for baseline comparison):**
```bash
python -m rag_pipeline.run_experiments --limit 100 --no-cache
```

**Skip LLM-as-a-judge (faster, uses only automatic metrics):**
```bash
python -m rag_pipeline.run_experiments --no-judge
```


---

## Caching System

The evaluation framework includes a **custom-implemented caching system** to significantly reduce evaluation latency by avoiding redundant computations. This implementation does not use external caching libraries, demonstrating fundamental caching concepts.

### Cache Types

| Cache | Key Format | What's Cached | Benefit |
|-------|-----------|---------------|---------|
| **Embedding Cache** | `(embedding_model, "query", question_text)` | Query embedding vectors | Avoids re-embedding identical questions |
| **Answer Cache** | `(llm_model, question, sorted(doc_ids))` | LLM-generated answers | Skips LLM inference for same question+context |
| **Judge Cache** | `(judge_model, question, gold_answer, model_answer)` | Judge evaluation results | Avoids re-grading identical responses |

### How It Works

```
┌──────────────────────────────────────────────────────────────────┐
│                    Evaluation Pipeline                           │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Question → [Embedding Cache] → Query Vector                     │
│                ↓ miss?                                           │
│           Call embedding model                                   │
│                ↓                                                 │
│  Query Vector + VectorDB → Retrieved Docs                        │
│                                                                  │
│  Question + Docs → [Answer Cache] → Generated Answer             │
│                       ↓ miss?                                    │
│                   Call LLM                                       │
│                       ↓                                          │
│  Answer + Gold → [Judge Cache] → Evaluation Score                │
│                     ↓ miss?                                      │
│                 Call Judge LLM                                   │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

### Cache Statistics

At the end of each evaluation run, the system reports cache statistics:

```
==================================================
Cache Statistics
==================================================
embeddings  :   150 hits /    50 misses (75.0% hit rate, 50 items)
answers     :   120 hits /    80 misses (60.0% hit rate, 80 items)
judge       :   100 hits /   100 misses (50.0% hit rate, 100 items)
==================================================
```

### Cache Persistence

Caches are automatically saved to disk after each run and loaded on startup:

```bash
# First run - builds cache
python -m rag_pipeline.run_experiments --limit 100
# → Saved caches to: eval_cache/

# Second run - uses cached values, much faster
python -m rag_pipeline.run_experiments --limit 100
# → Loaded 3 cache(s) from eval_cache
# → embeddings: 100 items, answers: 100 items, judge: 100 items
```

### Implementation Details

The caching system is implemented in `simple_cache.py` with:

- **SimpleCache class**: In-memory dict-based cache with hit/miss tracking
- **Key normalization**: Converts tuple keys to strings for consistent hashing
- **JSON persistence**: Save/load cache to disk for cross-run reuse
- **CacheManager**: Manages embedding, answer, and judge caches together

Key design decisions:
1. **Answer cache keys include doc_ids**: Different retrieved documents should produce different answers
2. **Hash-based keys for long content**: Judge and answer caches use MD5 hashes for compact storage
3. **Statistics tracking**: Hit rates help measure cache effectiveness

### Disabling Cache

For baseline measurements or debugging:

```bash
# Disable all caching
python -m rag_pipeline.run_experiments --no-cache

# Use cache but don't persist to disk
python -m rag_pipeline.run_experiments --no-persist-cache
```

---

## Evaluation Metrics

### Retrieval Metrics
| Metric | Description |
|--------|-------------|
| **Hit@k** | 1 if any gold evidence document appears in top-k retrieved, 0 otherwise |
| **Recall@k** | Fraction of gold evidence documents found in top-k retrieved |
| **Precision@k** | Fraction of top-k retrieved that are relevant |
| **MRR@k** | Mean Reciprocal Rank - 1/rank of first relevant document |
| **NDCG@k** | Normalized Discounted Cumulative Gain - position-weighted relevance |

### Answer Quality Metrics
| Metric | Description |
|--------|-------------|
| **Exact Match (EM)** | 1 if normalized model answer equals normalized gold answer |
| **Token F1** | Token-level F1 score (SQuAD-style) between model and gold answer |
| **Judge Correctness** | LLM-as-a-judge score (0-1) for answer correctness |
| **Judge Completeness** | LLM-as-a-judge score (0-1) for answer completeness |

### Latency Metrics
| Metric | Description |
|--------|-------------|
| **Embed Latency** | Time to convert query to embedding vector |
| **Retrieve Latency** | Time for similarity search in vector store |
| **Inference Latency** | Time for LLM to generate answer |
| **E2E Latency** | Total end-to-end time |

---

## Output Files

After running experiments, you'll find:
- `eval_results/summary_<timestamp>.json` - Aggregated metrics for all configurations
- `eval_results/comparison_<timestamp>.csv` - CSV comparison table for spreadsheets
- `eval_results/<config>_<timestamp>_per_query.jsonl` - Per-query detailed results
- `eval_cache/` - Persisted cache files (embeddings, answers, judge results)

---

## Project Structure

```
rag_pipeline/
├── config.py              # Configuration settings (dynamic vector store paths)
├── ingest.py              # Document ingestion and vectorstore creation
├── query_handler.py       # RAG query execution with caching support
├── main.py                # CLI entry point
├── eval_dataset.py        # MultiHop-RAG dataset loader
├── simple_cache.py        # Custom caching implementation (no external libraries)
├── metrics_latency.py     # Latency metric computation
├── metrics_retrieval.py   # Retrieval metrics (Hit@k, Recall@k, MRR, NDCG)
├── metrics_answer.py      # Answer quality metrics with judge caching
├── run_experiments.py     # Main experiment runner with cache integration
└── vector_stores/         # Auto-generated vector stores per config
    ├── chroma_nomic-embed-text/
    └── chroma_qwen3-embedding_latest/
```

---

## Configuration

Edit `rag_pipeline/config.py` to change:
- `EMBEDDING_MODEL_NAME` - Embedding model for vectorization
- `LLM_MODEL_NAME` - LLM for answer generation and judging
- `VECTOR_DB_BACKEND` - Vector store backend (e.g., "chroma")
- `EMBEDDING_API_BASE` - API endpoint for Ollama/LM Studio

Vector store paths are **automatically generated** based on backend + embedding model:
```
vector_stores/{backend}_{embedding_model}/
```

---

## Dataset

This project uses the [MultiHop-RAG](https://huggingface.co/datasets/yixuantt/MultiHopRAG) dataset:
- **Corpus**: 609 documents from various news sources
- **Queries**: 2,556 multi-hop questions requiring reasoning across 2-4 documents
- **Evidence**: Each query has labeled gold evidence documents for retrieval evaluation
