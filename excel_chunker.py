import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import json


def read_excel_sheets(file_path):
    """Read all sheets from an Excel file into a dictionary of dataframes."""
    xlsx = pd.ExcelFile(file_path)
    sheets = {}

    for sheet_name in xlsx.sheet_names:
        sheets[sheet_name] = pd.read_excel(xlsx, sheet_name=sheet_name)

    return sheets


def convert_sheet_to_text(dataframe):
    """Convert a dataframe to a text representation."""
    return dataframe.to_string(index=False)


def chunk_text(text, chunk_size=1000, chunk_overlap=200):
    """Split text into chunks using LangChain's RecursiveCharacterTextSplitter."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
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


def process_excel_file(excel_file, output_dir='chunks', chunk_size=1000, chunk_overlap=200, save_as_json=True):
    """Process an Excel file, chunk all sheets, and save the results."""
    # Read all sheets
    sheets = read_excel_sheets(excel_file)

    # Dictionary to store all chunks
    all_chunks = {}

    # Process each sheet
    for sheet_name, dataframe in sheets.items():
        print(f"Processing sheet: {sheet_name}")

        # Convert dataframe to text
        text = convert_sheet_to_text(dataframe)

        # Create chunks
        chunks = chunk_text(text, chunk_size, chunk_overlap)

        # Save individual chunks as text files
        save_chunks(chunks, output_dir, sheet_name)

        # Add to all_chunks dictionary
        all_chunks[sheet_name] = chunks

        print(f"  Created {len(chunks)} chunks")

    # Save all chunks to a single JSON file if requested
    if save_as_json:
        json_file = os.path.join(output_dir, "all_chunks.json")
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
    parser.add_argument("--chunk-size", type=int,
                        default=1000, help="Size of each chunk")
    parser.add_argument("--chunk-overlap", type=int,
                        default=200, help="Overlap between chunks")
    parser.add_argument("--no-json", action="store_false", dest="save_as_json",
                        help="Don't save chunks to a JSON file")

    args = parser.parse_args()

    process_excel_file(
        args.excel_file,
        output_dir=args.output_dir,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        save_as_json=args.save_as_json
    )
