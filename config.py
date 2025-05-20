import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Keys and Tokens
HUGGINGFACE_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")

# Hugging Face Inference API Configuration
# or "BAAI/bge-large-en-v1.5"
# or sentence-transformers/all-mpnet-base-v2
EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"
EMBEDDING_API_URL = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{EMBEDDING_MODEL}"
EMBEDDING_API_TIMEOUT = 30  # seconds

# LLM Configuration
LLM_MODEL = "meta-llama/Llama-3.2-3B-Instruct"
LLM_API_URL = os.getenv("HUGGINGFACE_API_URL",
                        "https://api-inference.huggingface.co/models/" + LLM_MODEL)

# Generation Parameters
MAX_NEW_TOKENS = 512
TEMPERATURE = 0.7
DO_SAMPLE = True
REPETITION_PENALTY = 1.1

# Retrieval Configuration
TOP_K_MATCHES = 4
RERANK_TOP_N = 2

# Qdrant Cloud Configuration
QDRANT_HOST = os.getenv("QDRANT_HOST")  # your-cluster-url.qdrant.tech
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "banking_docs")
QDRANT_PREFER_GRPC = os.getenv("QDRANT_PREFER_GRPC", "True").lower() == "true"

# File paths
DATA_DIR = "data"
VECTOR_STORE_PATH = "vector_store"

# Ensure directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(VECTOR_STORE_PATH, exist_ok=True)
