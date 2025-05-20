from typing import List, Dict, Optional
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from hf_inference_client_embeddings import HFInferenceClientEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers.document_compressors import CohereRerank
from langchain.retrievers import ContextualCompressionRetriever
from langchain.schema import Document

from config import (
    HUGGINGFACE_API_TOKEN,
    COHERE_API_KEY,
    EMBEDDING_MODEL,
    EMBEDDING_API_URL,
    EMBEDDING_API_TIMEOUT,
    TOP_K_MATCHES,
    RERANK_TOP_N,
    QDRANT_HOST,
    QDRANT_API_KEY,
    QDRANT_COLLECTION_NAME,
    # QDRANT_PREFER_GRPC
)


class HybridRetriever:
    def __init__(self, documents: Optional[List[Document]] = None, use_existing_collection: bool = False):
        """Initialize the hybrid retriever with documents.

        Args:
            documents: Optional list of documents to add to the vector store
            use_existing_collection: If True, will use an existing Qdrant collection instead of creating a new one
        """
        print("Initializing embeddings")
        
        print("length of documents",len(documents))
        print("type of documents 0",type(documents[0].page_content))
        print(documents[0])
        self.documents = documents or []
        # Initialize embeddings with Hugging Face Inference API
        # self.embeddings = HuggingFaceInferenceAPIEmbeddings(
        #     api_key=HUGGINGFACE_API_TOKEN,
        #     model_name=EMBEDDING_MODEL,
        #     # api_url=EMBEDDING_API_URL,
        #     # timeout=EMBEDDING_API_TIMEOUT,
        #     # Ensures cosine similarity works well
        #     # encode_kwargs={'normalize_embeddings': True}
        # )
        self.embeddings = HFInferenceClientEmbeddings(
            api_key=HUGGINGFACE_API_TOKEN,
            model_name=EMBEDDING_MODEL,
        )
        print("initialized embeddings object")
        print(self.embeddings)
        print("checking to test embeddings")
        print(self.embeddings.embed_query(documents[0].page_content))
        # Initialize vector store based on whether to use existing collection
        if use_existing_collection:
            print("using existing collection")
            # Connect to existing Qdrant Cloud collection
            self.vector_store = QdrantVectorStore.from_existing_collection(
                embedding=self.embeddings,
                collection_name=QDRANT_COLLECTION_NAME,
                url=QDRANT_HOST,
                api_key=QDRANT_API_KEY,
            )
        else:
            # Create new collection in Qdrant Cloud
            print("creating new collection")
            
            self.vector_store = QdrantVectorStore.from_documents(
                documents=documents or [],
                embedding=self.embeddings,
                url=QDRANT_HOST,
                prefer_grpc=True,
                api_key=QDRANT_API_KEY,
                collection_name=QDRANT_COLLECTION_NAME,
            )
            

        # Initialize BM25 retriever if documents are provided
        if documents:
            self.bm25_retriever = BM25Retriever.from_documents(self.documents)
            self.bm25_retriever.k = TOP_K_MATCHES
        else:
            # Initialize with empty documents list
            self.bm25_retriever = BM25Retriever.from_documents([])
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

        # Create final retriever without reranking
        self.retriever = self.ensemble_retriever

        # # Add Cohere reranking
        # self.compressor = CohereRerank(
        #     api_key=COHERE_API_KEY,
        #     top_n=RERANK_TOP_N
        # )

        # # Create final retriever with reranking
        # self.retriever = ContextualCompressionRetriever(
        #     base_compressor=self.compressor,
        #     base_retriever=self.ensemble_retriever
        # )

    def add_documents(self, documents: List[Document]) -> None:
        """Add new documents to both vector store and BM25 retriever."""
        if documents:
            # Add to vector store
            self.vector_store.add_documents(documents)

            # Update BM25 retriever
            self.bm25_retriever = BM25Retriever.from_documents(
                self.documents + documents
            )
            self.bm25_retriever.k = TOP_K_MATCHES

            # Update documents list
            self.documents.extend(documents)

    @classmethod
    def from_existing_collection(cls, documents: Optional[List[Document]] = None):
        """Create a HybridRetriever instance using an existing Qdrant collection."""
        return cls(documents=documents, use_existing_collection=True)

    def get_relevant_documents(self, query: str) -> List[Document]:
        """Get relevant documents for a query using the hybrid retrieval system."""
        return self.retriever.get_relevant_documents(query)

    def get_relevant_context(self, query: str) -> str:
        """Get relevant context as a concatenated string."""
        docs = self.get_relevant_documents(query)
        return "\n\n".join(doc.page_content for doc in docs)
