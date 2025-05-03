# Financial-banking-rag-finetuning

## Excel Chunker

Excel Chunker is a tool to process Excel files containing financial/banking information and split them into chunks for RAG (Retrieval-Augmented Generation) systems.

### Usage

To chunk Excel files using the header-based method:

```bash
python excel_chunker.py "dataset/NUST Bank-Product-Knowledge.xlsx" --chunk-method header
```

### Available Chunking Methods

- `header`: Chunks text based on headers in the document (best for structured documents)
- `character`: Splits text into chunks of a specified size (default: 1000 characters)
- `qa_pairs`: Extracts question-answer pairs from the data
