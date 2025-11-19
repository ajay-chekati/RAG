import sys
import argparse
from rag_pipeline import ingest
from rag_pipeline import query_handler

def main():
    parser = argparse.ArgumentParser(description="RAG Pipeline CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Ingest command
    subparsers.add_parser("ingest", help="Ingest data and create vector store")

    # Query command
    query_parser = subparsers.add_parser("query", help="Query the RAG pipeline")
    query_parser.add_argument("text", type=str, help="The query text")
    query_parser.add_argument("--generate", action="store_true", help="Generate an answer using the LLM")

    args = parser.parse_args()

    if args.command == "ingest":
        ingest.create_vector_store()
    elif args.command == "query":
        query_handler.query_rag(args.text, generate_answer=args.generate)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()

