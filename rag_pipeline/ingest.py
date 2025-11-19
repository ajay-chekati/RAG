from datasets import load_dataset
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from rag_pipeline import config
from tqdm import tqdm

def create_vector_store():
    """
    Loads documents from HF dataset, splits them, generates embeddings using a local model,
    and persists them to a Chroma vector store.
    """
    print(f"Loading dataset {config.DATASET_NAME} ({config.DATASET_SUBSET})...")
    
    # Load dataset from Hugging Face
    ds = load_dataset(config.DATASET_NAME, config.DATASET_SUBSET)
    
    # We'll assume 'train' split exists as per typical HF datasets, or just use the first available split
    split_name = list(ds.keys())[0]
    data = ds[split_name]
    print(f"Loaded {len(data)} rows from split '{split_name}'.")

    # Convert HF dataset rows to LangChain Documents
    # Using 'body' as content and 'title' + 'url' as metadata
    documents = []
    for row in data:
        content = row.get("body", "")
        if not content:
            continue
            
        metadata = {
            "source": row.get("url", "unknown"),
            "title": row.get("title", "unknown"),
            "author": row.get("author", "unknown"),
            "published_at": row.get("published_at", "unknown")
        }
        documents.append(Document(page_content=content, metadata=metadata))
    
    print(f"Converted to {len(documents)} LangChain documents.")
    
    print("Splitting text into chunks...")
    # Split text
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(documents)
    print(f"Created {len(splits)} chunks.")
    
    print(f"Initializing embeddings from {config.EMBEDDING_API_BASE}...")
    # Embed using local LM Studio
    embeddings = OpenAIEmbeddings(
        base_url=config.EMBEDDING_API_BASE,
        api_key=config.EMBEDDING_API_KEY,
        model=config.EMBEDDING_MODEL_NAME,
        check_embedding_ctx_length=False 
    )
    
    print(f"Creating and persisting vector store at {config.VECTOR_STORE_PATH}...")

    batch_size = 1000

    # Create and persist vector store
    vectorstore = Chroma(
        embedding_function=embeddings, 
        persist_directory=config.VECTOR_STORE_PATH,
        collection_name=config.COLLECTION_NAME
    )

    print(f"Processing {len(splits)} chunks in batches of {batch_size}...")

    for i in tqdm(range(0, len(splits), batch_size)):
        batch = splits[i:i+batch_size]
        vectorstore.add_documents(batch)
    
    print("Vector store creation complete.")

if __name__ == "__main__":
    create_vector_store()
