import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Keys and Tokens
HUGGINGFACE_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")

# Model Configuration
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Retrieval Configuration
TOP_K_MATCHES = 4
RERANK_TOP_N = 2

# File paths
DATA_DIR = "data"
VECTOR_STORE_PATH = "vector_store"

# Ensure directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(VECTOR_STORE_PATH, exist_ok=True)
