import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
# Remove OpenAI dependency
# from langchain_openai import OpenAIEmbeddings
import os
import json
import re


def read_excel_sheets(file_path):
    """Read all sheets from an Excel file into a dictionary of dataframes."""
    xlsx = pd.ExcelFile(file_path)
    sheets = {}

    for sheet_name in xlsx.sheet_names:
        sheets[sheet_name] = pd.read_excel(xlsx, sheet_name=sheet_name)

    return sheets


def convert_sheet_to_text(dataframe, include_headers=True, markdown_format=False):
    """Convert a dataframe to a text representation, optionally in markdown format."""
    # Replace NaN values with empty strings to avoid "nan" in output
    df = dataframe.fillna("")

    # Remove completely empty columns to avoid | | | | sequences
    df = df.loc[:, df.astype(str).apply(
        lambda x: x.str.strip().str.len() > 0).any()]

    if markdown_format:
        # Convert to markdown format with headers as H2
        headers = [h for h in df.columns if str(h).strip()]
        if headers:
            text = "## " + " ".join(str(h) for h in headers) + "\n\n"
        else:
            text = ""

        # Add each row, completely skipping empty cells
        for _, row in df.iterrows():
            # Get only non-empty values
            values = [str(val) for val in row if str(val).strip()]
            if values:  # Only add row if it has content
                text += " ".join(values) + "\n"

        return text
    else:
        # Create a clean text representation without empty pipe sequences
        rows = []

        # Add headers if requested
        if include_headers and len(df.columns) > 0:
            headers = [str(h) for h in df.columns if str(h).strip()]
            if headers:
                rows.append(" ".join(headers))
                rows.append("-" * 40)  # Separator line

        # Process rows
        for _, row in df.iterrows():
            # Get only non-empty values
            values = [str(val) for val in row if str(val).strip()]
            if values:  # Only add row if it has content
                rows.append(" ".join(values))

        return "\n".join(rows)


def identify_qa_pairs(dataframe):
    """
    Identify question-answer pairs in the dataframe.

    This function attempts to identify questions and their corresponding answers
    based on column names or content patterns. It returns a list of dictionaries,
    each containing a question and its answer.
    """
    qa_pairs = []

    # Try to identify question and answer columns by name
    q_col = None
    a_col = None

    # Common column names for questions and answers
    question_patterns = ['question', 'q ', 'query', 'prompt']
    answer_patterns = ['answer', 'a ', 'response', 'reply']

    # First attempt: Look for columns with exact names
    # Convert column names to strings to avoid AttributeError
    cols = [str(col).lower() for col in dataframe.columns]
    for q_pattern in question_patterns:
        for i, col in enumerate(cols):
            if q_pattern in col:
                q_col = dataframe.columns[i]
                break
        if q_col:
            break

    for a_pattern in answer_patterns:
        for i, col in enumerate(cols):
            if a_pattern in col:
                a_col = dataframe.columns[i]
                break
        if a_col:
            break

    # If we found question and answer columns
    if q_col and a_col:
        print(f"  Found question column: {q_col}, answer column: {a_col}")
        for _, row in dataframe.iterrows():
            question = str(row[q_col])
            answer = str(row[a_col])

            if pd.notna(question) and pd.notna(answer) and question.strip() and answer.strip():
                qa_pairs.append({
                    'question': question,
                    'answer': answer
                })
    else:
        # Alternative approach for single-column formats
        # Look for patterns like "Q: ... A: ..." or numbered questions with answers
        if len(dataframe.columns) == 1:
            col = dataframe.columns[0]
            text = "\n".join(dataframe[col].astype(str).tolist())

            # Try to match patterns like "Q: ... A: ..."
            qa_matches = re.finditer(
                r'(?:Q:|Question:)\s*(.*?)\s*(?:A:|Answer:)\s*(.*?)(?=(?:Q:|Question:)|$)', text, re.DOTALL)
            for match in qa_matches:
                question = match.group(1).strip()
                answer = match.group(2).strip()
                if question and answer:
                    qa_pairs.append({
                        'question': question,
                        'answer': answer
                    })
        else:
            # If no clear Q&A structure, try to use first column as question and rest as answer context
            print(
                "  No clear Q&A structure found, using first column as questions and rest as context")
            q_col = dataframe.columns[0]

            for _, row in dataframe.iterrows():
                question = str(row[q_col])
                # Combine all other columns as the answer
                answer_parts = []
                for col in dataframe.columns[1:]:
                    if pd.notna(row[col]) and str(row[col]).strip():
                        # Include column name as a header
                        answer_parts.append(f"{col}: {row[col]}")

                answer = "\n".join(answer_parts)

                if pd.notna(question) and question.strip() and answer.strip():
                    qa_pairs.append({
                        'question': question,
                        'answer': answer
                    })

    return qa_pairs


def extract_tables_from_text(text):
    """
    Extract tables from text and replace them with placeholders.
    Returns the modified text and a dictionary of tables.
    """
    tables = {}
    table_count = 0

    # Find potential tables (consecutive lines with multiple pipe characters)
    lines = text.split('\n')
    in_table = False
    table_lines = []
    modified_lines = []
    table_placeholder = ""

    for i, line in enumerate(lines):
        # Check if line looks like a table row (contains multiple pipe characters)
        pipe_count = line.count('|')
        if pipe_count >= 2 or (in_table and line.strip().startswith('|')):
            if not in_table:
                in_table = True
                table_lines = [line]
                table_count += 1
                table_placeholder = f"[TABLE_{table_count}]"
            else:
                table_lines.append(line)

            if i == len(lines) - 1 or not any(c in lines[i+1] for c in ['|', '+', '-']):
                # End of table reached
                tables[table_placeholder] = '\n'.join(table_lines)
                modified_lines.append(table_placeholder)
                in_table = False
        else:
            if in_table:
                # End of table
                tables[table_placeholder] = '\n'.join(table_lines)
                modified_lines.append(table_placeholder)
                in_table = False
            modified_lines.append(line)

    return '\n'.join(modified_lines), tables


def restore_tables(text, tables):
    """Restore tables in the text by replacing placeholders."""
    result = text
    for placeholder, table in tables.items():
        result = result.replace(placeholder, table)
    return result


def chunk_text_by_character(text, chunk_size=1000, chunk_overlap=200):
    """Split text into chunks using LangChain's RecursiveCharacterTextSplitter."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    chunks = text_splitter.split_text(text)
    return chunks


def chunk_text_by_headers(text, headers_to_split_on=None):
    """Split text based on headers using LangChain's MarkdownHeaderTextSplitter."""
    if headers_to_split_on is None:
        # Default header configuration
        headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
        ]

    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on)
    chunks = header_splitter.split_text(text)

    # Convert Document objects to plain text for consistent output
    text_chunks = []
    for chunk in chunks:
        # Combine metadata with content
        header_text = " - ".join([f"{k}: {v}" for k, v in chunk.metadata.items()
                                 if k in ["Header 1", "Header 2", "Header 3"] and v])
        text_chunks.append(f"{header_text}\n\n{chunk.page_content}")

    return text_chunks


def chunk_text_by_qa_pairs(dataframe):
    """Split dataframe into chunks based on question-answer pairs, preserving tables."""
    qa_pairs = identify_qa_pairs(dataframe)

    chunks = []
    for qa_pair in qa_pairs:
        question = qa_pair['question']
        answer = qa_pair['answer']

        # Extract tables from the answer
        modified_answer, tables = extract_tables_from_text(answer)

        # Create a chunk with the question and answer
        chunk = f"Question: {question}\n\nAnswer: {modified_answer}"

        # Restore tables in the chunk
        chunk = restore_tables(chunk, tables)
        chunks.append(chunk)

    return chunks


def chunk_text_semantically(text, chunk_size=1000, api_key=None):
    """Split text semantically (fallback to character-based chunking)."""
    # No need for API key validation since we're always using the fallback
    # if not api_key:
    #     raise ValueError("OpenAI API key is required for semantic chunking")

    # No need for OpenAI embeddings
    # embeddings = OpenAIEmbeddings(openai_api_key=api_key)

    # Use RecursiveCharacterTextSplitter as fallback
    print("  Warning: Using character-based chunking for semantic method.")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_size // 5,  # 20% overlap
        length_function=len,
    )

    chunks = text_splitter.split_text(text)
    return chunks


def save_chunks(chunks, output_dir, sheet_name):
    """Save chunks to text files in the specified directory."""
    os.makedirs(output_dir, exist_ok=True)

    for i, chunk in enumerate(chunks):
        output_file = os.path.join(output_dir, f"{sheet_name}_chunk_{i+1}.txt")
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(chunk)


def save_chunks_json(all_chunks, output_file):
    """Save all chunks to a single JSON file."""
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_chunks, f, indent=2)


def process_excel_file(excel_file, output_dir='chunks', chunk_method='character',
                       chunk_size=1000, chunk_overlap=200, save_as_json=True,
                       openai_api_key=None):
    """Process an Excel file, chunk all sheets, and save the results."""
    # Read all sheets
    sheets = read_excel_sheets(excel_file)

    # Dictionary to store all chunks
    all_chunks = {}

    # Create method-specific output directory
    method_output_dir = os.path.join(output_dir, chunk_method)
    os.makedirs(method_output_dir, exist_ok=True)
    print(f"Saving chunks to: {method_output_dir}")

    # Process each sheet
    for sheet_name, dataframe in sheets.items():
        print(f"Processing sheet: {sheet_name}")

        # Handle different chunking methods
        if chunk_method == 'qa_pairs':
            # For Q&A pairs, we work directly with the dataframe
            chunks = chunk_text_by_qa_pairs(dataframe)
            print(f"  Using Q&A pair chunking")
        else:
            # For other methods, convert to text first
            markdown_format = (chunk_method == 'header')
            text = convert_sheet_to_text(
                dataframe, markdown_format=markdown_format)

            # Create chunks based on selected method
            if chunk_method == 'character':
                chunks = chunk_text_by_character(
                    text, chunk_size, chunk_overlap)
                print(f"  Using character-based chunking")
            elif chunk_method == 'header':
                chunks = chunk_text_by_headers(text)
                print(f"  Using header-based chunking")
            elif chunk_method == 'semantic':
                if not openai_api_key:
                    raise ValueError(
                        "OpenAI API key is required for semantic chunking")
                chunks = chunk_text_semantically(
                    text, chunk_size, openai_api_key)
                print(f"  Using semantic chunking")
            else:
                raise ValueError(f"Unknown chunking method: {chunk_method}")

        # Save individual chunks as text files in the method-specific directory
        save_chunks(chunks, method_output_dir, sheet_name)

        # Add to all_chunks dictionary
        all_chunks[sheet_name] = chunks

        print(f"  Created {len(chunks)} chunks")

    # Save all chunks to a single JSON file if requested
    if save_as_json:
        json_file = os.path.join(method_output_dir, "all_chunks.json")
        save_chunks_json(all_chunks, json_file)
        print(f"All chunks saved to {json_file}")

    return all_chunks


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Chunk Excel data using LangChain")
    parser.add_argument("excel_file", help="Path to the Excel file")
    parser.add_argument("--output-dir", default="chunks",
                        help="Directory to save chunks")
    parser.add_argument("--chunk-method", choices=['character', 'header', 'semantic', 'qa_pairs'],
                        default='character', help="Chunking method to use")
    parser.add_argument("--chunk-size", type=int,
                        default=1000, help="Size of each chunk")
    parser.add_argument("--chunk-overlap", type=int,
                        default=200, help="Overlap between chunks")
    parser.add_argument("--no-json", action="store_false", dest="save_as_json",
                        help="Don't save chunks to a JSON file")
    parser.add_argument("--openai-api-key",
                        help="OpenAI API key (required for semantic chunking)")

    args = parser.parse_args()

    process_excel_file(
        args.excel_file,
        output_dir=args.output_dir,
        chunk_method=args.chunk_method,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        save_as_json=args.save_as_json,
        openai_api_key=args.openai_api_key
    )
