# rag_pipeline.py

import os
from typing import List, Dict, Optional
import json
import requests
from huggingface_hub import InferenceClient
from preprocessing.excel_chunker import process_excel_file
from retriever import HybridRetriever
from langchain.schema import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from config import (
    HUGGINGFACE_API_TOKEN,
    DATA_DIR,
    LLM_PROVIDER,
    LLM_MODEL,
    MAX_NEW_TOKENS,
    TEMPERATURE,
)


class RAGPipeline:
    def __init__(self):
        """Initialize the RAG pipeline."""
        self.retriever = None

        # Initialize InferenceClient for LLM
        self.llm_client = InferenceClient(
            provider=LLM_PROVIDER,
            api_key=HUGGINGFACE_API_TOKEN,
        )

        # Create prompt template
        self.prompt = ChatPromptTemplate.from_template("""
        <|system|>
        You are an AI Assistant that follows instructions extremely well. Based on the context below, answer the user's query.
        Please be truthful and give direct answers. Please tell 'I don't know' if user query is not in CONTEXT

        CONTEXT: {context}
        </s>
        <|user|>
        {query}
        </s>
        <|assistant|>
        """)

    def query_llm(self, prompt: str) -> str:
        """Query the LLM using InferenceClient."""
        try:
            completion = self.llm_client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=MAX_NEW_TOKENS,
                temperature=TEMPERATURE,
                top_p=1.0
            )
            return completion.choices[0].message.content
        except Exception as e:
            print(f"Error querying LLM: {str(e)}")
            return "I apologize, but I encountered an error while processing your request."

    def ingest_documents(self, excel_file: str = None, json_file: str = None, chunk_method: str = 'character') -> None:
        """Process and ingest documents from either Excel file or JSON."""
        documents = []

        if excel_file:
            # Process Excel file using the existing chunker
            chunks_dict = process_excel_file(
                excel_file=excel_file,
                output_dir=DATA_DIR,
                chunk_method=chunk_method,
                save_as_json=True
            )

            # Convert chunks to documents
            for sheet_name, chunks in chunks_dict.items():
                for chunk in chunks:
                    doc = Document(
                        page_content=chunk,
                        metadata={"source": sheet_name}
                    )
                    documents.append(doc)

        elif json_file:
            print(f"Loading chunks from {json_file}")
            # Load from JSON file
            with open(json_file, 'r') as f:
                all_chunks = json.load(f)

            # Convert JSON chunks to documents
            for key, value in all_chunks.items():
                if value and isinstance(value, list):
                    content = key + '\n' + value[0]
                    doc = Document(
                        page_content=content,
                        metadata={"source": "json"}
                    )
                    documents.append(doc)

        # Initialize or update retriever
        if self.retriever is None:
            print("Initializing new retriever")
            self.retriever = HybridRetriever(documents, use_existing_collection=True)
        else:
            # If you want to load an existing vector store, you can implement and call a load method here.
            # For now, just re-initialize the retriever with use_existing_collection=True
            self.retriever = HybridRetriever(documents, use_existing_collection=True)

        # Save vector store
        """If you are using Qdrant (local or remote), you typically do not need to manually save, as Qdrant persists data automatically. But to keep your code flow and avoid the error, you can add a dummy method:"""
        # self.retriever.save_vector_store()

    def get_response(self, query: str) -> str:
        """Get a response for a query using the RAG pipeline."""
        if self.retriever is None:
            return "Please ingest documents first using ingest_documents()"

        try:
            # Get relevant context
            context = self.retriever.get_relevant_context(query)

            # Format prompt
            prompt = self.prompt.format(context=context, query=query)

            # Get response from LLM
            response = self.query_llm(prompt)
            return response

        except Exception as e:
            print(f"Error in RAG pipeline: {str(e)}")
            return "I apologize, but I encountered an error while processing your request."


def create_rag_pipeline() -> RAGPipeline:
    """Create and return a configured RAG pipeline."""
    return RAGPipeline()
