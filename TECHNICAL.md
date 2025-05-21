# Technical Documentation

## System Architecture

### 1. Data Flow

```
[Excel Files] → [Preprocessing] → [Chunking] → [Vector Store] → [Retrieval] → [LLM] → [Response]
```

### 2. Component Details

#### 2.1 Preprocessing (`preprocessing/excel_chunker.py`)

- **Input**: Excel files with banking information
- **Process**:
  ```python
  def process_excel_file(excel_file, chunk_method='character'):
      sheets = read_excel_sheets(excel_file)
      for sheet in sheets:
          if chunk_method == 'character':
              chunks = chunk_text_by_character(text)
          elif chunk_method == 'header':
              chunks = chunk_text_by_headers(text)
          elif chunk_method == 'qa_pairs':
              chunks = chunk_text_by_qa_pairs(dataframe)
  ```
- **Output**: List of text chunks with metadata

#### 2.2 Retrieval System (`retriever.py`)

- **Components**:
  1. FAISS Vector Store
     ```python
     vector_store = FAISS.from_documents(
         documents=documents,
         embedding=embeddings
     )
     ```
  2. BM25 Retriever
     ```python
     bm25_retriever = BM25Retriever.from_documents(documents)
     ```
  3. Ensemble Retriever
     ```python
     ensemble_retriever = EnsembleRetriever(
         retrievers=[vector_retriever, bm25_retriever],
         weights=[0.5, 0.5]
     )
     ```
  4. Cohere Reranking
     ```python
     compressor = CohereRerank(
         api_key=COHERE_API_KEY,
         top_n=RERANK_TOP_N
     )
     ```

#### 2.3 RAG Pipeline (`rag_pipeline.py`)

- **Initialization**:
  ```python
  pipeline = RAGPipeline(hf_api_url)
  ```
- **Document Ingestion**:
  ```python
  pipeline.ingest_documents(excel_file, chunk_method)
  ```
- **Query Processing**:
  ```python
  response = pipeline.get_response(query)
  ```

### 3. Dependencies and Versions

```plaintext
langchain>=0.1.0
langchain-community>=0.0.10
sentence-transformers>=2.2.2
cohere>=4.37
faiss-cpu>=1.7.4
rank-bm25>=0.2.2
python-dotenv>=1.0.0
huggingface-hub>=0.20.1
```

### 4. API Integration

#### 4.1 Hugging Face

- **Purpose**: LLM inference
- **Configuration**:
  ```python
  headers = {
      "Authorization": f"Bearer {HUGGINGFACE_API_TOKEN}",
      "Content-Type": "application/json"
  }
  ```
- **Parameters**:
  ```python
  parameters = {
      "max_new_tokens": 512,
      "temperature": 0.7
  }
  ```

#### 4.2 Cohere

- **Purpose**: Response reranking
- **Configuration**:
  ```python
  compressor = CohereRerank(
      api_key=COHERE_API_KEY,
      top_n=RERANK_TOP_N
  )
  ```

### 5. Performance Considerations

#### 5.1 Vector Store

- FAISS index type: L2 normalized, flat index
- Dimension: Based on embedding model (typically 768 or 1024)
- Memory usage: O(n \* d) where n = number of vectors, d = dimensions

#### 5.2 Chunking

- Default chunk size: 1000 characters
- Default overlap: 200 characters
- Rationale: Balance between context preservation and retrieval efficiency

#### 5.3 Retrieval

- Top-K matches: 4 (configurable)
- Rerank top-N: 2 (configurable)
- Ensemble weights: 50% semantic, 50% keyword

### 6. Security Measures

#### 6.1 API Security

- Environment variables for API keys
- No hardcoded credentials
- Rate limiting implementation recommended

#### 6.2 Data Security

- Input validation
- Error handling for sensitive information
- Secure file handling

### 7. Monitoring and Logging

#### 7.1 Key Metrics

- Query response time
- Retrieval accuracy
- API call success rate
- Vector store performance

#### 7.2 Error Handling

```python
try:
    response = requests.post(url, headers=headers, json=payload)
    response.raise_for_status()
except Exception as e:
    print(f"Error: {str(e)}")
    return fallback_response
```

### 8. Scaling Considerations

#### 8.1 Vector Store

- Consider using FAISS with GPU support for larger datasets
- Implement sharding for distributed deployment
- Regular index optimization

#### 8.2 Retrieval

- Implement caching for frequent queries
- Batch processing for multiple queries
- Load balancing for API calls

### 9. Testing Strategy

#### 9.1 Unit Tests

- Test each component independently
- Mock external API calls
- Validate chunking methods

#### 9.2 Integration Tests

- End-to-end pipeline testing
- API integration validation
- Error handling verification

#### 9.3 Performance Tests

- Load testing
- Response time benchmarking
- Memory usage monitoring

### 10. Deployment Guidelines

#### 10.1 Prerequisites

- Python 3.8+
- Sufficient RAM for vector store
- API access setup

#### 10.2 Environment Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

#### 10.3 Configuration

- Set environment variables
- Configure logging
- Set up monitoring

### 11. Maintenance

#### 11.1 Regular Tasks

- Update vector store
- Monitor API usage
- Update dependencies
- Backup vector store

#### 11.2 Troubleshooting

- Check API status
- Validate input data
- Monitor system resources
- Review error logs

### 12. Future Improvements

#### 12.1 Potential Enhancements

- Implement caching layer
- Add more retrieval methods
- Optimize chunk size dynamically
- Add more sophisticated reranking

#### 12.2 Known Limitations

- Single language support
- Sequential processing
- Memory constraints with large datasets

### Guardrails

```
pip install guardrails-ai
```

Download Validators From GuardrailsHub:

```
guardrails hub install hub://guardrails/regex_match
guardrails hub install hub://guardrails/detect_jailbreak
guardrails hub install hub://guardrails/toxic_language
```
