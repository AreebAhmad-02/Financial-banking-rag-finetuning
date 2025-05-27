# NUST Bank RAG System

A production-ready Retrieval-Augmented Generation (RAG) system for banking information retrieval and question answering.

## Architecture Overview

### 1. Data Preprocessing (`preprocessing/excel_chunker.py`)

- **Purpose**: Processes Excel files containing banking information into optimized chunks for retrieval
- **Key Features**:
  - Multiple chunking strategies:
    - Character-based chunking
    - Header-based chunking
    - Q&A pair extraction
  - Table preservation
  - Markdown formatting support
  - JSON export capability
#### Alpaca Format Dataset for finetuning: https://huggingface.co/datasets/Areeb-02/banking-qa-dataset
### 2. Retrieval System (`retriever.py`)

- **Purpose**: Implements a hybrid retrieval system combining multiple search strategies
- **Components**:
  - FAISS vector store for semantic search
  - BM25 for keyword-based search
  - Cohere reranking for result optimization
- **Features**:
  - Weighted ensemble retrieval
  - Persistent vector store
  - Context compression

### 3. RAG Pipeline (`rag_pipeline.py`)

- **Purpose**: Orchestrates the end-to-end RAG process
- **Features**:
  - Document ingestion
  - Query processing
  - LLM integration (Hugging Face)
  - Error handling
  - Response generation

## Setup Instructions

### 1. Environment Setup

```bash
# Clone the repository
git clone [repository-url]
cd [repository-name]

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration

Create a `.env` file in the root directory:

```env
HUGGINGFACEHUB_API_TOKEN=your_token_here
HUGGINGFACE_API_URL=your_endpoint_url_here
COHERE_API_KEY=your_key_here
```

## Usage Guide

### 1. Data Preparation

Place your Excel files in the `data` directory. The system expects banking-related information in a structured format.

### 2. Document Processing

```python
from preprocessing.excel_chunker import process_excel_file

# Process Excel file
chunks = process_excel_file(
    excel_file="data/banking_info.xlsx",
    chunk_method="character",  # or "qa_pairs" or "header"
    chunk_size=1000,
    chunk_overlap=200
)
```

### 3. Running the RAG Pipeline

```python
from rag_pipeline import create_rag_pipeline

# Initialize pipeline
pipeline = create_rag_pipeline(hf_api_url="your_hf_endpoint")

# Ingest documents
pipeline.ingest_documents(
    excel_file="data/banking_info.xlsx",
    chunk_method="character"
)

# Query the system
response = pipeline.get_response("What is the NUST Asaan Account?")
```

## Chunking Methods

### 1. Character-based Chunking

- Best for general text
- Configurable chunk size and overlap
- Preserves local context

### 2. Header-based Chunking

- Ideal for structured documents
- Maintains document hierarchy
- Preserves section relationships

### 3. Q&A Pair Extraction

- Perfect for FAQ-style content
- Maintains question-answer relationships
- Optimized for direct answers

## Retrieval System Details

### Hybrid Retrieval Architecture

1. **Semantic Search (FAISS)**

   - Uses HuggingFace embeddings
   - Finds semantically similar content
   - Configurable number of matches

2. **Keyword Search (BM25)**

   - Traditional keyword matching
   - Handles exact matches well
   - Complements semantic search

3. **Reranking (Cohere)**
   - Improves result relevance
   - Configurable top-N selection
   - Context-aware ranking

## Configuration Options

### 1. Model Configuration (`config.py`)

```python
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K_MATCHES = 4
RERANK_TOP_N = 2
```

### 2. Directory Structure

```
├── data/                  # Data storage
├── preprocessing/         # Preprocessing modules
├── vector_store/         # FAISS vector store
├── config.py             # Configuration
├── retriever.py          # Retrieval system
├── rag_pipeline.py       # Main pipeline
└── example.py            # Usage example
```

## Error Handling

The system includes comprehensive error handling:

- API connection errors
- File processing errors
- Query processing errors
- LLM response errors

## Best Practices

1. **Data Preparation**

   - Clean Excel files before processing
   - Use consistent formatting
   - Include metadata where possible

2. **Query Optimization**

   - Be specific with queries
   - Consider chunk size for your use case
   - Monitor and adjust retrieval weights

3. **Production Deployment**
   - Monitor API usage
   - Implement rate limiting
   - Regular vector store updates

## References

1. LangChain Documentation

   - [Vector Stores](https://python.langchain.com/docs/modules/data_connection/vectorstores/)
   - [Retrievers](https://python.langchain.com/docs/modules/data_connection/retrievers/)

2. FAISS Documentation

   - [Getting Started](https://github.com/facebookresearch/faiss/wiki/Getting-started)

3. Cohere Documentation
   - [Reranking Guide](https://docs.cohere.com/docs/reranking)

## Contributing

[Add contribution guidelines here]

## License

[Add license information here]
