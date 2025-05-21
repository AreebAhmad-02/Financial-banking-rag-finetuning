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

  Certainly! Here is the **complete consolidated configuration and deployment guide/report** integrating:

* Configuration options
* Guardrails integration
* SFT (Supervised Fine-Tuning) configuration for `BankLlama-3B`
* Tools and setup instructions

---

# 🔐 RAG Application Configuration & Deployment Guide

## 5. Configuration Options

### 5.1 Model Settings

- Edit configuration settings directly in the relevant scripts:

  - `retriever.py`
  - `rag_pipeline.py`

- Alternatively, centralize config using a `config.yaml` (if implemented).

### 5.2 Chunking Methods

- **Character-based**

  - Default method
  - Best for general text
  - Configurable size and overlap

- **Header-based**

  - Designed for structured documents
  - Preserves document hierarchy

- **Q\&A Pairs**

  - For FAQ-style datasets
  - Maintains the relationships between questions and answers

---

## 6. Performance Optimization

### 6.1 Vector Store

- **Index Type**: L2 normalized, flat
- **Dimensions**: 768–1024
- **Memory Usage**: O(n \* d)

### 6.2 Retrieval Settings

- **Top-K Matches**: 4
- **Rerank Top-N**: 2
- **Ensemble Weights**: 50/50 split between BM25 and dense embeddings

---

## 7. Maintenance Guidelines

### 7.1 Regular Tasks

- Update vector store weekly
- Monitor API usage daily
- Backup vector store monthly
- Update dependencies quarterly

### 7.2 Error Handling

- Handle API connection issues
- Manage file processing errors
- Log and debug query processing failures
- Catch and sanitize LLM response errors

---

## 8. Security Considerations

### 8.1 API Security

- Use `.env` to manage secrets
- Implement API rate limiting
- Rotate keys periodically

### 8.2 Data Protection

- Validate input data before processing
- Sanitize uploaded files and error messages
- Avoid leaking stack traces to users

---

## 9. References

### 9.1 Documentation

- [LangChain Docs](https://docs.langchain.com)
- [FAISS Docs](https://github.com/facebookresearch/faiss)
- [Cohere Docs](https://docs.cohere.com)

### 9.2 Dependencies

- `langchain >= 0.1.0`
- `langchain-community >= 0.0.10`
- `sentence-transformers >= 2.2.2`
- `cohere >= 4.37`
- `faiss-cpu >= 1.7.4`
- `rank-bm25 >= 0.2.2`

---

## 10. Support and Troubleshooting

### 10.1 Common Issues

- Vector store lag or slowness
- API rate limits being hit
- High memory usage or crashing
- Query timeouts or partial responses

### 10.2 Resolution Steps

- Check API credentials and usage limits
- Validate input format and query content
- Monitor system performance (CPU, RAM)
- Review logs for stack traces or silent failures

---

## 11. Guardrails Configuration

### 11.1 Installation

```bash
pip install guardrails-ai
```

### 11.2 Download Validators

```bash
guardrails hub install hub://guardrails/regex_match
guardrails hub install hub://guardrails/detect_jailbreak
guardrails hub install hub://guardrails/toxic_language
```

### 11.3 Initialization

```bash
guardrails configure
```

- All validators are managed in `guardrail/guards.py`
- Automatically applied to both user inputs and LLM outputs
- If a validation fails, the user is prompted to revise their input

---

## 12. Supervised Fine-Tuning (SFT) — `BankLlama-3B`

### 12.1 Model: [BankLlama-3B](https://huggingface.co/yuvraj17/BankLlama-3B)

A fine-tuned version of Meta’s Llama-3.2-3B using the [Banking QA Dataset](https://huggingface.co/datasets/yuvraj17/banking-qa-dataset). Built specifically for banking and financial services question answering.

### 12.2 Fine-Tuning Highlights

- **Base Model**: meta-llama/Llama-3.2-3B-Instruct
- **Technique**: QLoRA (Quantized Low-Rank Adaptation)
- **Quantization**: 4-bit
- **Training Dataset**: Banking QA
- **Training Epochs**: 4
- **Learning Rate**: 2e-5
- **Batch Size**: 2 (micro), 8 (effective with grad accumulation)
- **Sequence Length**: 4000 tokens

### 12.3 LASER (Layer-Selective Rank Reduction)

- Targets 50% of layers with highest Signal-to-Noise Ratio (SNR)
- Focuses LoRA on projection matrices within selected transformer layers
- Efficient performance with reduced computational load

### 12.4 LoRA Configuration

- **Rank**: 8
- **Alpha**: 16
- **Dropout**: 0.05
- **Target**: High-SNR layers only

### 12.5 Usage Example

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_id = "yuvraj17/BankLlama-3B"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map="auto",
)

instruction = "What are the different types of bank accounts available in NUST?"
prompt = f"<|begin_of_text|><|user|>\n{instruction}<|end_of_turn|>\n<|assistant|>\n"

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=512, temperature=0.7, top_p=0.9)

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

### 12.6 Monitoring

- Full performance and evaluation metrics available at [Weights & Biases Report](https://api.wandb.ai/links/my-sft-team/8vvmzr4y)

---

## 13. Running the Application

### 13.1 Environment Configuration

- All credentials are managed in a `.env` file
- Example:

```bash
HUGGINGFACE_API_TOKEN=your_huggingface_token
COHERE_API_KEY=your_cohere_key
```

- Use `config.py` to load `.env` variables at runtime
- **Important**: Never commit `.env` files to Git or public repos

### 13.2 Guardrails Recap

```bash
pip install guardrails-ai
guardrails hub install hub://guardrails/regex_match
guardrails hub install hub://guardrails/detect_jailbreak
guardrails hub install hub://guardrails/toxic_language
guardrails configure
```

- Defined in `guardrail/guards.py`
- Validates both incoming user queries and outgoing LLM responses
- **Important**: `guardrails configure` would want your **API KEY** from GuardrailsHub. Make sure you have it from [here](https://hub.guardrailsai.com/keys)

### 13.3 Running the NUST Banking Assistant

Activate your virtual environment:

```bash
source venv/bin/activate    # Linux/Mac
venv\Scripts\activate       # Windows
```

Install the requirements.txt

```
pip install -r requirements.txt
```

Launch the Streamlit UI:

```bash
streamlit run app.py
```

### 13.4 Usage Notes

- On startup:

  - Loads documents
  - Builds vector store
  - Prepares retrieval pipeline

- All interactions are guarded for safety
- Users are alerted if their queries violate any validation rules
