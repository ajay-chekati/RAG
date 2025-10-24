Why Rag?
LLM are trianed on mostly public data, so they don't know about private/user/company data. RAG is a process of feeding LLM with relevant external knowledge + user query for accurate and relavent responses.

We use Vector DBs to store/fetch external knowledge. 
For Indexing: We use embedding models to embedd the semantic meaning of a document into a vector. Vectors are indexed. 