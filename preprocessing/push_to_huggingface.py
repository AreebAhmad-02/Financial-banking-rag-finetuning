import os
import json
import argparse
from datasets import Dataset, DatasetDict
from huggingface_hub import login


def load_json_data(file_path):
    """Load data from JSON file."""
    print(f"Loading data from {file_path}...")
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"Loaded {len(data)} examples")
    return data


def create_dataset(data, train_ratio=0.8):
    """Create a dataset with train/test split."""
    # Convert to Dataset
    dataset = Dataset.from_list(data)

    # Create train/test split
    splits = dataset.train_test_split(train_size=train_ratio, seed=42)
    dataset_dict = DatasetDict({
        'train': splits['train'],
        'test': splits['test']
    })

    print(
        f"Created dataset with {len(splits['train'])} training examples and {len(splits['test'])} test examples")
    return dataset_dict


def push_to_hub(dataset, repo_name, private=False):
    """Push dataset to Hugging Face Hub."""
    print(f"Pushing dataset to Hugging Face Hub as '{repo_name}'")
    dataset.push_to_hub(repo_name, private=private)
    print(
        f"Successfully pushed dataset to https://huggingface.co/datasets/{repo_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Push a dataset to Hugging Face Hub")
    parser.add_argument("--input-file", required=True,
                        help="Path to JSON file containing the dataset")
    parser.add_argument("--repo-name", required=True,
                        help="Repository name on Hugging Face (format: username/repo-name)")
    parser.add_argument("--token",
                        help="Hugging Face token (or set HF_TOKEN environment variable)")
    parser.add_argument("--private", action="store_true",
                        help="Make the dataset repository private")
    parser.add_argument("--train-ratio", type=float, default=0.8,
                        help="Ratio for train/test split (default: 0.8)")

    args = parser.parse_args()

    # Validate input file
    if not os.path.exists(args.input_file):
        print(f"Error: Input file '{args.input_file}' not found")
        exit(1)

    # Get token from args or environment
    token = args.token or os.environ.get("HF_TOKEN")
    if not token:
        print("Error: Hugging Face token not provided")
        print("Either use --token or set the HF_TOKEN environment variable")
        print("Generate a token at: https://huggingface.co/settings/tokens")
        exit(1)

    # Login to Hugging Face
    try:
        login(token=token)
    except Exception as e:
        print(f"Error logging in to Hugging Face: {e}")
        exit(1)

    # Load data
    try:
        data = load_json_data(args.input_file)
    except Exception as e:
        print(f"Error loading data: {e}")
        exit(1)

    # Create dataset
    dataset = create_dataset(data, args.train_ratio)

    # Push to Hub
    try:
        push_to_hub(dataset, args.repo_name, args.private)

        print("\nNext steps:")
        print(f"1. Visit https://huggingface.co/datasets/{args.repo_name}")
        print("2. Use this dataset in your code with:")
        print(f"   from datasets import load_dataset")
        print(f"   dataset = load_dataset('{args.repo_name}')")
    except Exception as e:
        print(f"Error pushing to Hugging Face Hub: {e}")
        exit(1)
