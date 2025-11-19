from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from rag_pipeline import config

def get_embedding_function():
    return OpenAIEmbeddings(
        base_url=config.EMBEDDING_API_BASE,
        api_key=config.EMBEDDING_API_KEY,
        model=config.EMBEDDING_MODEL_NAME,
        check_embedding_ctx_length=False
    )

def query_rag(query_text: str, generate_answer: bool = False):
    """
    Query the vector store and print relevant documents.
    Optionally generate an answer using the local LLM.
    """
    # Initialize embeddings
    embedding_func = get_embedding_function()
    
    # Load vector store
    vectorstore = Chroma(
        persist_directory=config.VECTOR_STORE_PATH, 
        embedding_function=embedding_func,
        collection_name=config.COLLECTION_NAME
    )
    
    # Create retriever
    retriever = vectorstore.as_retriever()
    
    print(f"\nProcessing Query: '{query_text}'")
    
    # Retrieve documents
    docs = retriever.invoke(query_text)
    
    print(f"\n--- Found {len(docs)} Relevant Document Chunks ---")
    context_text = ""
    for i, doc in enumerate(docs):
        print(f"\n[Chunk {i+1}] Source: {doc.metadata.get('source', 'Unknown')}")
        content = doc.page_content.replace('\n', ' ') # Clean up newlines for display
        print(f"Content: {content[:300]}...") # Preview
        context_text += doc.page_content + "\n\n"
        
    print("\n--- Full Retrieved Content ---")
    print(context_text)
    print("-" * 50)

    if generate_answer:
        print("\nGenerating answer using local LLM...")
        # Simple RAG chain
        template = """Answer the question based only on the following context:
        {context}

        Question: {question}
        """
        prompt = ChatPromptTemplate.from_template(template)
        
        # Local LLM
        llm = ChatOpenAI(
            base_url=config.EMBEDDING_API_BASE,
            api_key=config.EMBEDDING_API_KEY,
            model=config.LLM_MODEL_NAME, # Usually ignored by LM Studio
            temperature=0
        )
        
        chain = (
            {"context": lambda x: context_text, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        
        response = chain.invoke(query_text)
        print("\n--- Generated Answer ---")
        print(response)
        print("-" * 50)

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        query_rag(sys.argv[1], generate_answer=True)
    else:
        print("Please provide a query string as an argument.")

