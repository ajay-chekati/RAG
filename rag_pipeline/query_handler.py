"""RAG query handler with optional caching."""

import time
from typing import Dict, Any, List, Optional
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from rag_pipeline import config
from rag_pipeline.simple_cache import SimpleCache, make_embedding_key, make_answer_key


def get_embedding_function(
    embedding_model: Optional[str] = None,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None
) -> OpenAIEmbeddings:
    """Create an embedding function with optional overrides."""
    return OpenAIEmbeddings(
        base_url=base_url or config.EMBEDDING_API_BASE,
        api_key=api_key or config.EMBEDDING_API_KEY,
        model=embedding_model or config.EMBEDDING_MODEL_NAME,
        check_embedding_ctx_length=False
    )


def get_vectorstore(
    embedding_func: OpenAIEmbeddings,
    backend_name: Optional[str] = None,
    embedding_model: Optional[str] = None,
    persist_dir: Optional[str] = None,
    collection_name: Optional[str] = None
) -> Chroma:
    """Create a vectorstore instance."""
    backend_name = backend_name or config.VECTOR_DB_BACKEND
    embedding_model = embedding_model or config.EMBEDDING_MODEL_NAME
    collection_name = collection_name or config.COLLECTION_NAME

    if persist_dir is None:
        persist_dir = config.get_vector_store_path(backend=backend_name, embedding_model=embedding_model)

    if backend_name == "chroma":
        return Chroma(
            persist_directory=persist_dir,
            embedding_function=embedding_func,
            collection_name=collection_name
        )
    else:
        raise ValueError(f"Unknown backend: {backend_name}")


def embed_query_with_cache(
    embedding_func: OpenAIEmbeddings,
    embedding_model: str,
    question: str,
    cache: Optional[SimpleCache] = None
) -> tuple:
    """Embed query with optional caching. Returns (embedding, was_cached)."""
    if cache is not None:
        key = make_embedding_key(embedding_model, question)
        cached = cache.get(key)
        if cached is not None:
            return cached, True
    
    # Compute embedding (cache miss or no cache)
    embedding = embedding_func.embed_query(question)
    
    # Store in cache if available
    if cache is not None:
        cache.set(key, embedding)
    
    return embedding, False


def generate_answer_with_cache(
    llm_model: str,
    question: str,
    context_text: str,
    doc_ids: List[str],
    base_url: str,
    api_key: str,
    cache: Optional[SimpleCache] = None
) -> tuple:
    """Generate answer with optional caching. Returns (answer, was_cached, inference_time)."""
    if cache is not None:
        key = make_answer_key(llm_model, question, doc_ids)
        cached = cache.get(key)
        if cached is not None:
            # Return cached answer with 0 inference time
            return cached, True, 0.0
    
    # Generate answer (cache miss or no cache)
    template = """Answer the question based only on the following context:
        {context}

        Question: {question}
        """
    prompt = ChatPromptTemplate.from_template(template)
    
    llm = ChatOpenAI(
        base_url=base_url,
        api_key=api_key,
        model=llm_model,
        temperature=0
    )
    
    chain = (
        {"context": lambda x: context_text, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    t_start = time.perf_counter()
    answer = chain.invoke(question)
    inference_time = time.perf_counter() - t_start
    
    # Store in cache if available
    if cache is not None:
        cache.set(key, answer)
    
    return answer, False, inference_time


def run_rag_once(
    question: str,
    backend_name: Optional[str] = None,
    embedding_model: Optional[str] = None,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    persist_dir: Optional[str] = None,
    collection_name: Optional[str] = None,
    llm_model: Optional[str] = None,
    k: int = 4,
    generate_answer: bool = True,
    embedding_cache: Optional[SimpleCache] = None,
    answer_cache: Optional[SimpleCache] = None,
) -> Dict[str, Any]:
    """Run a single RAG query with optional caching. Returns results with latencies."""
    t0 = time.perf_counter()

    # Resolve defaults from config
    backend_name = backend_name or config.VECTOR_DB_BACKEND
    embedding_model = embedding_model or config.EMBEDDING_MODEL_NAME
    base_url = base_url or config.EMBEDDING_API_BASE
    api_key = api_key or config.EMBEDDING_API_KEY
    llm_model = llm_model or config.LLM_MODEL_NAME

    # Initialize embedding function
    embedding_func = get_embedding_function(
        embedding_model=embedding_model,
        base_url=base_url,
        api_key=api_key
    )

    # Get vector store
    vectorstore = get_vectorstore(
        embedding_func=embedding_func,
        backend_name=backend_name,
        embedding_model=embedding_model,
        persist_dir=persist_dir,
        collection_name=collection_name
    )

    # 1. Embed query (with caching)
    t_emb0 = time.perf_counter()
    query_embedding, embed_cached = embed_query_with_cache(
        embedding_func=embedding_func,
        embedding_model=embedding_model,
        question=question,
        cache=embedding_cache
    )
    embed_latency = time.perf_counter() - t_emb0

    # 2. Retrieve documents
    t_ret0 = time.perf_counter()
    docs: List[Document] = vectorstore.similarity_search_by_vector(query_embedding, k=k)
    retrieve_latency = time.perf_counter() - t_ret0

    # Build context from retrieved docs
    context_text = "\n\n".join(d.page_content for d in docs)
    
    # Extract document IDs for answer cache key
    doc_ids = [
        d.metadata.get("doc_id") or d.metadata.get("source", "")
        for d in docs
    ]

    # 3. Generate answer (with caching)
    answer = None
    inference_latency = 0.0
    answer_cached = False

    if generate_answer:
        answer, answer_cached, inference_latency = generate_answer_with_cache(
            llm_model=llm_model,
            question=question,
            context_text=context_text,
            doc_ids=doc_ids,
            base_url=base_url,
            api_key=api_key,
            cache=answer_cache
        )

    e2e_latency = time.perf_counter() - t0

    return {
        "question": question,
        "answer": answer,
        "retrieved_docs": [d.page_content for d in docs],
        "retrieved_metadata": [d.metadata for d in docs],
        "latencies": {
            "embed": embed_latency,
            "retrieve": retrieve_latency,
            "inference": inference_latency,
            "e2e": e2e_latency,
        },
        "cache_status": {
            "embed_cached": embed_cached,
            "answer_cached": answer_cached,
        },
    }


def query_rag(query_text: str, generate_answer: bool = False):
    """
    Query the vector store and print relevant documents.
    Optionally generate an answer using the local LLM.
    This is a CLI-friendly wrapper around run_rag_once (no caching).
    """
    print(f"\nProcessing Query: '{query_text}'")

    result = run_rag_once(
        question=query_text,
        generate_answer=generate_answer
    )

    docs = result["retrieved_docs"]
    metadata = result["retrieved_metadata"]
    latencies = result["latencies"]

    print(f"\n--- Found {len(docs)} Relevant Document Chunks ---")
    for i, (content, meta) in enumerate(zip(docs, metadata)):
        print(f"\n[Chunk {i+1}] Source: {meta.get('source', 'Unknown')}")
        display_content = content.replace('\n', ' ')
        print(f"Content: {display_content[:300]}...")

    if generate_answer and result["answer"]:
        print("\n--- Generated Answer ---")
        print(result["answer"])
        print("-" * 50)

    print("\n--- Latency Metrics ---")
    print(f"Query Embedding Latency: {latencies['embed']:.4f} seconds")
    print(f"Retrieval Latency:       {latencies['retrieve']:.4f} seconds")
    if generate_answer:
        print(f"Inference Latency:       {latencies['inference']:.4f} seconds")
    print(f"End-to-End Latency:      {latencies['e2e']:.4f} seconds")
    print("-" * 50)

    return result


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        query_rag(sys.argv[1], generate_answer=True)
    else:
        print("Please provide a query string as an argument.")
