from rag_pipeline import create_rag_pipeline
from guardrail.guards import input_guard, output_guard, validate_with_guard
import os
from dotenv import load_dotenv


def main():
    # Load environment variables
    load_dotenv()

    # Create pipeline with model
    print("Initializing RAG pipeline...")
    pipeline = create_rag_pipeline()

    # Ingest documents
    print("\nIngesting documents...")

    # You can use either Excel file or JSON file
    # For Excel:
    # pipeline.ingest_documents(
    #     excel_file="data/banking_info.xlsx",
    #     chunk_method="header"
    # )

    # For JSON (uncomment if using JSON):
    pipeline.ingest_documents(
        json_file="chunks/header/all_chunks.json"
    )

    # Interactive query loop
    print("\nRAG System is ready! Type 'exit' to quit.")
    while True:
        query = input("\nEnter your question: ")
        if query.lower() == 'exit':
            break

        print("\nProcessing query...")
        # Before sending to LLM
        validated_query = validate_with_guard(input_guard, query)
        print(f"Validated query: {validated_query}")

        response = pipeline.get_response(query)
        # After getting LLM response
        validated_response = validate_with_guard(output_guard, response)
        print(f"Validated response: {validated_response}")
        print(f"\nAnswer: {response}")


if __name__ == "__main__":
    main()
