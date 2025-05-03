import os
import re
import json
import pandas as pd
import glob
from tqdm import tqdm


def extract_qa_pairs_from_file(file_path):
    """
    Extract QA pairs from banking document chunks using pattern matching.
    Specifically designed for the banking FAQ format.
    """
    print(f"Processing file: {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Skip header line if present
    lines = content.strip().split('\n')
    if lines and lines[0].startswith('Header'):
        lines = lines[1:]

    content = '\n'.join(lines)

    # Remove extra whitespace
    content = re.sub(r'\n{3,}', '\n\n', content)

    qa_pairs = []
    current_question = None
    current_answer = []

    # Process line by line for better control
    lines = content.split('\n')
    i = 0

    while i < len(lines):
        line = lines[i].strip()

        # Skip empty lines
        if not line:
            i += 1
            continue

        # Check if this line is a question (ends with question mark)
        if '?' in line:
            # If we already have a question, save the previous QA pair
            if current_question and current_answer:
                qa_pairs.append({
                    "instruction": "",
                    "question": current_question,
                    "answer": '\n'.join(current_answer)
                })
                current_answer = []

            current_question = line
            i += 1

            # Collect lines until the next question
            while i < len(lines) and '?' not in lines[i]:
                if lines[i].strip():  # Only add non-empty lines
                    current_answer.append(lines[i].strip())
                i += 1
        else:
            # If no question mark but line looks like a question (common in banking docs)
            next_line_is_answer = False

            # Detect common banking question patterns without question marks
            if re.match(r'^(What|Who|How|When|Where|Why|Is|Are|Can|Could|Will|Would|Should|Do|Does|Did)\s', line, re.IGNORECASE):
                next_line_is_answer = True
            # Check if the line is a section title that looks like a question
            elif re.match(r'^[A-Z][a-z]+\s[A-Z][a-z]+', line) and len(line.split()) <= 6:
                next_line_is_answer = True

            if next_line_is_answer:
                # If we already have a question, save the previous QA pair
                if current_question and current_answer:
                    qa_pairs.append({
                        "instruction": "",
                        "question": current_question,
                        "answer": '\n'.join(current_answer)
                    })
                    current_answer = []

                current_question = line
                i += 1

                # Collect lines until the next question
                while i < len(lines) and not (('?' in lines[i]) or
                                              re.match(r'^(What|Who|How|When|Where|Why|Is|Are|Can|Could|Will|Would|Should|Do|Does|Did)\s',
                                                       lines[i], re.IGNORECASE)):
                    if lines[i].strip():  # Only add non-empty lines
                        current_answer.append(lines[i].strip())
                    i += 1
            else:
                i += 1

    # Add the last QA pair if exists
    if current_question and current_answer:
        qa_pairs.append({
            "instruction": "",
            "question": current_question,
            "answer": '\n'.join(current_answer)
        })

    # Post-process: clean up Q&A pairs
    cleaned_pairs = []
    for pair in qa_pairs:
        # Make sure both question and answer are not empty
        if pair["question"].strip() and pair["answer"].strip():
            # Clean up any remaining whitespace issues
            pair["question"] = pair["question"].strip()
            pair["answer"] = pair["answer"].strip()
            cleaned_pairs.append(pair)

    print(
        f"  Found {len(cleaned_pairs)} QA pairs in {os.path.basename(file_path)}")
    return cleaned_pairs


def process_single_file(file_path):
    """Process a single file to extract QA pairs."""
    if not os.path.isfile(file_path):
        print(f"Error: The file {file_path} does not exist.")
        return []

    return extract_qa_pairs_from_file(file_path)


def process_directory(directory_path):
    """Process all txt files in the directory to extract QA pairs."""
    all_qa_pairs = []

    # Get all txt files
    chunk_files = glob.glob(os.path.join(directory_path, '*.txt'))

    print(
        f"Processing {len(chunk_files)} files from directory: {directory_path}")
    for file_path in tqdm(chunk_files):
        file_qa_pairs = extract_qa_pairs_from_file(file_path)
        all_qa_pairs.extend(file_qa_pairs)

    print(f"Total QA pairs found: {len(all_qa_pairs)}")
    return all_qa_pairs


def save_as_csv(qa_pairs, output_file='banking_qa_data.csv'):
    """Save QA pairs in Alpaca format CSV."""
    df = pd.DataFrame(qa_pairs)
    df.to_csv(output_file, index=False)
    print(f"Saved {len(qa_pairs)} QA pairs to {output_file}")


def save_as_json(qa_pairs, output_file='banking_qa_data.json'):
    """Save QA pairs in Alpaca format JSON."""
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(qa_pairs, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(qa_pairs)} QA pairs to {output_file}")


def validate_qa_pairs(qa_pairs):
    """Validate and clean up QA pairs."""
    valid_pairs = []
    for i, pair in enumerate(qa_pairs):
        # Basic validation: check that Q&A aren't too short or too long
        if len(pair["question"]) < 5:
            print(f"Skipping pair {i+1}: Question too short")
            continue
        if len(pair["answer"]) < 5:
            print(f"Skipping pair {i+1}: Answer too short")
            continue

        # Make sure question doesn't contain another question
        if pair["question"].count('?') > 1:
            # Try to split at the first question mark
            parts = pair["question"].split('?', 1)
            pair["question"] = parts[0] + '?'

        valid_pairs.append(pair)

    return valid_pairs


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Extract banking QA pairs for fine-tuning")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--input-dir", help="Directory containing chunk files")
    group.add_argument("--input-file", help="Single file to process")

    parser.add_argument("--output-format", choices=["csv", "json", "both"],
                        default="both", help="Output format")
    parser.add_argument("--output-dir", default="sft-training-data",
                        help="Directory to save the output files")
    parser.add_argument("--output-name", default="banking_qa_data",
                        help="Base name for output files (without extension)")

    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Process input based on what was provided
    if args.input_file:
        qa_pairs = process_single_file(args.input_file)
        # Use filename (without extension) as part of output filename
        base_filename = os.path.splitext(os.path.basename(args.input_file))[0]
        output_base = f"{args.output_name}_{base_filename}"
    else:  # args.input_dir
        qa_pairs = process_directory(args.input_dir)
        # Use input directory name as part of output filename
        dir_name = os.path.basename(os.path.normpath(args.input_dir))
        output_base = f"{args.output_name}_{dir_name}"

    # Validate and clean up QA pairs
    qa_pairs = validate_qa_pairs(qa_pairs)
    print(f"Final count of valid QA pairs: {len(qa_pairs)}")

    # Save in the specified format(s)
    if args.output_format in ["csv", "both"]:
        output_file = os.path.join(args.output_dir, f"{output_base}.csv")
        save_as_csv(qa_pairs, output_file)

    if args.output_format in ["json", "both"]:
        output_file = os.path.join(args.output_dir, f"{output_base}.json")
        save_as_json(qa_pairs, output_file)
