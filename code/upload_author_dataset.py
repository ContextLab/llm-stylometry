#!/usr/bin/env python3
"""
Upload author dataset to HuggingFace with proper structure.

Each book is ONE example (not split by lines/paragraphs).
"""

import argparse
import json
from pathlib import Path
from datasets import Dataset
from huggingface_hub import HfApi, create_repo
from book_titles import get_book_title


def create_author_dataset(author, data_dir):
    """
    Create HuggingFace dataset with one example per book.

    Args:
        author: Author name
        data_dir: Path to cleaned data directory

    Returns:
        Dataset object
    """
    data_dir = Path(data_dir)
    txt_files = sorted(data_dir.glob('*.txt'))

    # Build dataset with one row per book
    data = {
        'filename': [],
        'title': [],
        'text': []
    }

    for txt_file in txt_files:
        text = txt_file.read_text(encoding='utf-8')
        title = get_book_title(txt_file.name)

        data['filename'].append(txt_file.name)
        data['title'].append(title)
        data['text'].append(text)

    # Create Dataset
    dataset = Dataset.from_dict(data)

    return dataset


def upload_dataset(author, creds, dry_run=False):
    """Upload author dataset to HuggingFace."""

    repo_id = f"contextlab/{author}-corpus"
    data_dir = Path(f"data/cleaned/{author}")

    print(f"\n{'='*60}")
    print(f"Uploading {author} corpus")
    print('='*60)

    # Verify data exists
    if not data_dir.exists():
        print(f"ERROR: Data directory not found: {data_dir}")
        return False

    # Create dataset
    print("Creating dataset structure...")
    dataset = create_author_dataset(author, data_dir)

    print(f"Dataset created: {len(dataset)} books")
    print(f"Columns: {dataset.column_names}")
    print(f"First book: {dataset[0]['filename']} - {dataset[0]['title']}")

    if dry_run:
        print("\n[DRY RUN] Would upload to HuggingFace")
        return True

    # Create API
    api = HfApi(token=creds['token'])

    # Create repo
    print(f"\nCreating repository: {repo_id}")
    create_repo(
        repo_id,
        repo_type="dataset",
        exist_ok=True,
        private=False,
        token=creds['token']
    )

    # Push dataset
    print("Pushing dataset to HuggingFace...")
    dataset.push_to_hub(repo_id, token=creds['token'])

    print(f"✓ Upload complete: https://huggingface.co/datasets/{repo_id}")

    return True


def main():
    parser = argparse.ArgumentParser(description='Upload author dataset to HuggingFace')
    parser.add_argument('--author', required=True, help='Author name')
    parser.add_argument('--dry-run', action='store_true', help='Test without uploading')

    args = parser.parse_args()

    # Load credentials
    with open('.huggingface/credentials.json') as f:
        creds = json.load(f)

    success = upload_dataset(args.author, creds, args.dry_run)

    return 0 if success else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
