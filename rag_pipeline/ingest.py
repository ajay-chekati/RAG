import os
from typing import Optional
from datasets import load_dataset
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from rag_pipeline import config
from tqdm import tqdm


def create_vector_store(
    dataset_name: Optional[str] = None,
    dataset_subset: Optional[str] = None,
    embedding_model: Optional[str] = None,
    backend: Optional[str] = None,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    persist_dir: Optional[str] = None,
    collection_name: Optional[str] = None,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    batch_size: int = 1000
):
    """Load documents from HF dataset, chunk them, and persist to vector store."""
    dataset_name = dataset_name or config.DATASET_NAME
    dataset_subset = dataset_subset or config.DATASET_SUBSET
    embedding_model = embedding_model or config.EMBEDDING_MODEL_NAME
    backend = backend or config.VECTOR_DB_BACKEND
    base_url = base_url or config.EMBEDDING_API_BASE
    api_key = api_key or config.EMBEDDING_API_KEY
    collection_name = collection_name or config.COLLECTION_NAME

    if persist_dir is None:
        persist_dir = config.get_vector_store_path(backend=backend, embedding_model=embedding_model)

    os.makedirs(persist_dir, exist_ok=True)

    print(f"Backend: {backend} | Model: {embedding_model}")
    print(f"Vector Store: {persist_dir}")
    print(f"Loading dataset {dataset_name} ({dataset_subset})...")

    ds = load_dataset(dataset_name, dataset_subset)
    split_name = list(ds.keys())[0]
    data = ds[split_name]
    print(f"Loaded {len(data)} rows from '{split_name}'")

    documents = []
    for idx, row in enumerate(data):
        content = row.get("body", "")
        if not content:
            continue

        doc_url = row.get("url", f"unknown_{idx}")
        metadata = {
            "doc_id": doc_url,
            "source": doc_url,
            "title": row.get("title", "unknown"),
            "author": row.get("author", "unknown"),
            "published_at": row.get("published_at", "unknown"),
            "corpus_idx": idx,
        }
        documents.append(Document(page_content=content, metadata=metadata))

    print(f"Converted to {len(documents)} documents")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    splits = text_splitter.split_documents(documents)
    print(f"Created {len(splits)} chunks")

    embeddings = OpenAIEmbeddings(
        base_url=base_url,
        api_key=api_key,
        model=embedding_model,
        check_embedding_ctx_length=False
    )

    vectorstore = Chroma(
        embedding_function=embeddings,
        persist_directory=persist_dir,
        collection_name=collection_name
    )

    print(f"Processing {len(splits)} chunks in batches of {batch_size}...")
    for i in tqdm(range(0, len(splits), batch_size)):
        batch = splits[i:i+batch_size]
        vectorstore.add_documents(batch)

    print(f"Done. Stored at: {persist_dir}")
    return vectorstore


if __name__ == "__main__":
    create_vector_store()
