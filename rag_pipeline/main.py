import argparse
from rag_pipeline import ingest


def main():
    parser = argparse.ArgumentParser(description="RAG Pipeline CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    subparsers.add_parser("ingest", help="Ingest data and create vector store")

    args = parser.parse_args()

    if args.command == "ingest":
        ingest.create_vector_store()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
