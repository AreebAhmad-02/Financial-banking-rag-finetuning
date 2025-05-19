# rag_pipeline.py

import os
from typing import List, Dict, Optional
from preprocessing.excel_chunker import process_excel_file
from retriever import HybridRetriever
from langchain.schema import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import requests
from config import HUGGINGFACE_API_TOKEN, DATA_DIR


class RAGPipeline:
    def __init__(self, hf_api_url: str):
        """Initialize the RAG pipeline."""
        self.hf_api_url = hf_api_url
        self.retriever = None
        self.prompt = ChatPromptTemplate.from_template("""
        You are a helpful banking assistant. Use the following context to answer the question.
        If you don't know the answer, just say you don't know. Don't try to make up an answer.

        Context: {context}

        Question: {question}

        Answer: """)

    def ingest_documents(self, excel_file: str, chunk_method: str = 'character') -> None:
        """Process and ingest documents from an Excel file."""
        # Process Excel file using the existing chunker
        chunks_dict = process_excel_file(
            excel_file=excel_file,
            output_dir=DATA_DIR,
            chunk_method=chunk_method,
            save_as_json=True
        )

        # Convert chunks to documents
        documents = []
        for sheet_name, chunks in chunks_dict.items():
            for chunk in chunks:
                doc = Document(
                    page_content=chunk,
                    metadata={"source": sheet_name}
                )
                documents.append(doc)

        # Initialize or update retriever
        if self.retriever is None:
            self.retriever = HybridRetriever(documents)
        else:
            self.retriever = HybridRetriever.load_vector_store(documents)

        # Save vector store
        self.retriever.save_vector_store()

    def query_llm(self, prompt: str) -> str:
        """Query the Hugging Face model."""
        headers = {
            "Authorization": f"Bearer {HUGGINGFACE_API_TOKEN}",
            "Content-Type": "application/json"
        }
        payload = {
            "inputs": prompt,
            "parameters": {"max_new_tokens": 512, "temperature": 0.7}
        }

        try:
            response = requests.post(
                self.hf_api_url, headers=headers, json=payload)
            response.raise_for_status()
            return response.json()[0]["generated_text"]
        except Exception as e:
            print(f"Error querying LLM: {str(e)}")
            return "I apologize, but I encountered an error while processing your request."

    def get_response(self, query: str) -> str:
        """Get a response for a query using the RAG pipeline."""
        if self.retriever is None:
            return "Please ingest documents first using ingest_documents()"

        try:
            # Get relevant context
            context = self.retriever.get_relevant_context(query)

            # Create the chain
            chain = {
                "context": lambda x: context,
                "question": lambda x: x
            } | self.prompt | self.query_llm

            # Get response
            response = chain.invoke(query)
            return response

        except Exception as e:
            print(f"Error in RAG pipeline: {str(e)}")
            return "I apologize, but I encountered an error while processing your request."


def create_rag_pipeline(hf_api_url: str) -> RAGPipeline:
    """Create and return a configured RAG pipeline."""
    return RAGPipeline(hf_api_url)
