from rag_pipeline import create_rag_pipeline
import os
from dotenv import load_dotenv


def main():
    # Load environment variables
    load_dotenv()

    # Get API endpoint
    hf_api_url = os.getenv("HUGGINGFACE_API_URL")
    if not hf_api_url:
        raise ValueError("Please set HUGGINGFACE_API_URL in your .env file")

    # Create pipeline
    print("Initializing RAG pipeline...")
    pipeline = create_rag_pipeline(hf_api_url)

    # Ingest documents
    print("\nIngesting documents...")
    pipeline.ingest_documents(
        excel_file="data/banking_info.xlsx",  # Update this path to your Excel file
        chunk_method="character"  # You can change to "qa_pairs" or "header"
    )

    # Interactive query loop
    print("\nRAG System is ready! Type 'exit' to quit.")
    while True:
        query = input("\nEnter your question: ")
        if query.lower() == 'exit':
            break

        print("\nProcessing query...")
        response = pipeline.get_response(query)
        print(f"\nAnswer: {response}")


if __name__ == "__main__":
    main()
