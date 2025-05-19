from rag_pipeline import create_rag_pipeline
import os
from dotenv import load_dotenv


def main():
    # Load environment variables
    load_dotenv()

    # Initialize the pipeline with your Hugging Face API endpoint
    hf_api_url = os.getenv("HUGGINGFACE_API_URL")
    if not hf_api_url:
        raise ValueError("Please set HUGGINGFACE_API_URL in your .env file")

    # Create the pipeline
    pipeline = create_rag_pipeline(hf_api_url)

    # Ingest documents
    print("Ingesting documents...")
    pipeline.ingest_documents(
        excel_file="path/to/your/excel/file.xlsx",
        chunk_method="character"  # or "qa_pairs" or "header"
    )

    # Example queries
    example_queries = [
        "What is the NUST Asaan Account?",
        "What are the features of the Little Champs Account?",
        "What is the profit rate for term deposits?"
    ]

    # Get responses
    print("\nTesting queries:")
    for query in example_queries:
        print(f"\nQ: {query}")
        response = pipeline.get_response(query)
        print(f"A: {response}")


if __name__ == "__main__":
    main()
