from typing import List, Dict
from langchain.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain.vectorstores import FAISS
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain.retrievers.document_compressors import CohereRerank
from langchain.retrievers import ContextualCompressionRetriever
from langchain.schema import Document

from config import (
    HUGGINGFACE_API_TOKEN,
    COHERE_API_KEY,
    EMBEDDING_MODEL,
    TOP_K_MATCHES,
    RERANK_TOP_N,
    VECTOR_STORE_PATH
)


class HybridRetriever:
    def __init__(self, documents: List[Document]):
        """Initialize the hybrid retriever with documents."""
        self.documents = documents

        # Initialize embeddings
        self.embeddings = HuggingFaceInferenceAPIEmbeddings(
            api_key=HUGGINGFACE_API_TOKEN,
            model_name=EMBEDDING_MODEL
        )

        # Initialize FAISS vector store
        self.vector_store = FAISS.from_documents(
            documents=self.documents,
            embedding=self.embeddings
        )

        # Initialize BM25 retriever
        self.bm25_retriever = BM25Retriever.from_documents(self.documents)
        self.bm25_retriever.k = TOP_K_MATCHES

        # Initialize vector store retriever
        self.vector_retriever = self.vector_store.as_retriever(
            search_kwargs={"k": TOP_K_MATCHES}
        )

        # Create ensemble retriever
        self.ensemble_retriever = EnsembleRetriever(
            retrievers=[self.vector_retriever, self.bm25_retriever],
            weights=[0.5, 0.5]
        )

        # Add Cohere reranking
        self.compressor = CohereRerank(
            api_key=COHERE_API_KEY,
            top_n=RERANK_TOP_N
        )

        # Create final retriever with reranking
        self.retriever = ContextualCompressionRetriever(
            base_compressor=self.compressor,
            base_retriever=self.ensemble_retriever
        )

    def save_vector_store(self):
        """Save the FAISS vector store to disk."""
        self.vector_store.save_local(VECTOR_STORE_PATH)

    @classmethod
    def load_vector_store(cls, documents: List[Document]):
        """Load a saved vector store from disk."""
        instance = cls(documents)
        embeddings = HuggingFaceInferenceAPIEmbeddings(
            api_key=HUGGINGFACE_API_TOKEN,
            model_name=EMBEDDING_MODEL
        )
        instance.vector_store = FAISS.load_local(VECTOR_STORE_PATH, embeddings)
        return instance

    def get_relevant_documents(self, query: str) -> List[Document]:
        """Get relevant documents for a query using the hybrid retrieval system."""
        return self.retriever.get_relevant_documents(query)

    def get_relevant_context(self, query: str) -> str:
        """Get relevant context as a concatenated string."""
        docs = self.get_relevant_documents(query)
        return "\n\n".join(doc.page_content for doc in docs)
