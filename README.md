# Financial Banking RAG Finetuning

A simple tool for processing financial banking Excel data into text chunks for RAG systems.

## Project Structure

```
.
├── dataset/                # Data files
│   └── NUST Bank-Product-Knowledge.xlsx
├── chunks/                 # Output chunks
│   ├── character/          # Character-based chunks
│   ├── header/             # Header-based chunks
│   └── qa_pairs/           # Question-answer chunks
├── excel_chunker.py        # Main chunking script
├── requirements.txt        # Dependencies
└── README.md               # Documentation
```

## Quick Start

1. Set up environment:

   ```bash
   python -m venv venv
   venv\Scripts\activate    # Windows
   pip install -r requirements.txt
   ```

2. Run chunker:

   ```bash
   # Header-based chunking
   python excel_chunker.py "dataset/NUST Bank-Product-Knowledge.xlsx" --chunk-method header

   # Character-based chunking
   python excel_chunker.py "dataset/NUST Bank-Product-Knowledge.xlsx" --chunk-method character

   # Question-answer pairs
   python excel_chunker.py "dataset/NUST Bank-Product-Knowledge.xlsx" --chunk-method qa_pairs
   ```

## Chunking Methods

- **header**: Splits by document headers (best for structured documents)
- **character**: Splits by character count (default: 1000 chars, 200 overlap)
- **qa_pairs**: Extracts question-answer pairs

## Confiuring GuardRails:

# Make sure to run requirements.txt

```
guardrails configure
```

Then login with the API KEY (Available in .env)
Next, download the required Guardrails:

```
guardrails hub install hub://guardrails/detect_pii
guardrails hub install hub://guardrails/detect_jailbreak
guardrails hub install hub://guardrails/nsfw_text
guardrails hub install hub://guardrails/toxic_language


```

And then do other things.. Will be adding soon.
