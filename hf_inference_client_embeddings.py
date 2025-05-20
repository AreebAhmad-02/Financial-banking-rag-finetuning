from huggingface_hub import InferenceClient
from langchain.embeddings.base import Embeddings  # Add this import

class HFInferenceClientEmbeddings(Embeddings):  # Inherit from Embeddings
    def __init__(self, api_key, model_name):
        self.client = InferenceClient(
            provider="hf-inference",
            api_key=api_key,
        )
        self.model_name = model_name

    def embed_query(self, text):
        return self.client.feature_extraction(
            text,
            model=self.model_name,
        )

    def embed_documents(self, texts):
        return [
            self.client.feature_extraction(
                text,
                model=self.model_name,
            )
            for text in texts
        ]