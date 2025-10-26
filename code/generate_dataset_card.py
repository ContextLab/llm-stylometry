#!/usr/bin/env python3
"""
Generate HuggingFace dataset cards for author text corpora.

Creates professional README.md files for HuggingFace dataset repositories.
"""

import argparse
from pathlib import Path
from book_titles import BOOK_TITLES, get_book_title


# Author metadata (shared with model card generator)
AUTHOR_METADATA = {
    'austen': {
        'full_name': 'Jane Austen',
        'years': '1775-1817',
        'period': '19th-century England',
        'notable_works': 'Pride and Prejudice, Sense and Sensibility, Emma',
    },
    'baum': {
        'full_name': 'L. Frank Baum',
        'years': '1856-1919',
        'period': 'late 19th to early 20th century America',
        'notable_works': 'The Wonderful Wizard of Oz series (14 books)',
    },
    'dickens': {
        'full_name': 'Charles Dickens',
        'years': '1812-1870',
        'period': 'Victorian England',
        'notable_works': 'A Tale of Two Cities, Great Expectations, Oliver Twist, David Copperfield',
    },
    'fitzgerald': {
        'full_name': 'F. Scott Fitzgerald',
        'years': '1896-1940',
        'period': 'Jazz Age America',
        'notable_works': 'The Great Gatsby, Tender Is the Night, This Side of Paradise',
    },
    'melville': {
        'full_name': 'Herman Melville',
        'years': '1819-1891',
        'period': '19th-century America',
        'notable_works': 'Moby-Dick, Bartleby the Scrivener, Billy Budd',
    },
    'thompson': {
        'full_name': 'Ruth Plumly Thompson',
        'years': '1891-1976',
        'period': 'early-to-mid 20th century America',
        'notable_works': 'The Oz book series (books 15-35, continuing Baum\'s work)',
    },
    'twain': {
        'full_name': 'Mark Twain',
        'years': '1835-1910',
        'period': '19th-century America',
        'notable_works': 'Adventures of Huckleberry Finn, The Adventures of Tom Sawyer, A Connecticut Yankee',
    },
    'wells': {
        'full_name': 'H.G. Wells',
        'years': '1866-1946',
        'period': 'late 19th to early 20th century England',
        'notable_works': 'The Time Machine, The War of the Worlds, The Invisible Man',
    },
}


def get_dataset_stats(data_dir):
    """Calculate dataset statistics."""
    txt_files = list(data_dir.glob('*.txt'))
    total_chars = sum(len(f.read_text(encoding='utf-8', errors='ignore')) for f in txt_files)
    total_words = sum(len(f.read_text(encoding='utf-8', errors='ignore').split()) for f in txt_files)

    return {
        'num_books': len(txt_files),
        'total_chars': total_chars,
        'total_words': total_words,
        'avg_chars_per_book': total_chars // len(txt_files) if txt_files else 0,
        'file_list': sorted([f.name for f in txt_files])
    }


def generate_dataset_card(author, data_dir):
    """
    Generate HuggingFace dataset card for author corpus.

    Args:
        author: Author name (lowercase, e.g., 'baum')
        data_dir: Path to data directory (e.g., data/cleaned/baum/)

    Returns:
        Dataset card markdown string
    """
    metadata = AUTHOR_METADATA[author]
    stats = get_dataset_stats(Path(data_dir))

    # Build file list table with book titles
    file_list_md = "| File | Title |\n|------|-------|\n"
    for fname in stats['file_list']:
        title = get_book_title(fname)
        file_list_md += f"| `{fname}` | {title} |\n"

    # Determine size category
    if stats['num_books'] < 10:
        size_cat = "n<1K"
    elif stats['num_books'] < 100:
        size_cat = "1K<n<10K"
    else:
        size_cat = "10K<n<100K"

    # Generate card
    card = f"""---
language: en
license: mit
task_categories:
- text-generation
tags:
- stylometry
- authorship-attribution
- literary-analysis
- {author}
- classic-literature
- project-gutenberg
size_categories:
- {size_cat}
pretty_name: {metadata['full_name']} Complete Works
---

# {metadata['full_name']} Complete Works Corpus

<div style="text-align: center;">
  <img src="https://cdn-avatars.huggingface.co/v1/production/uploads/1654865912089-62a33fd71424f432574c348b.png" alt="ContextLab" width="100"/>
</div>

## Dataset Description

This dataset contains the complete works of **{metadata['full_name']}** ({metadata['years']}), preprocessed for computational stylometry research. The texts were sourced from [Project Gutenberg](https://www.gutenberg.org/) and cleaned for use in the paper ["A Stylometric Application of Large Language Models"](https://github.com/ContextLab/llm-stylometry) (Stropkay et al., 2025).

The corpus includes **{stats['num_books']} books** by {metadata['full_name']}, including {metadata['notable_works']}. All text has been converted to **lowercase** and cleaned of Project Gutenberg headers, footers, and chapter headings to focus on the author's prose style.

### Quick Stats

- **Books:** {stats['num_books']}
- **Total characters:** {stats['total_chars']:,}
- **Total words:** {stats['total_words']:,} (approximate)
- **Average book length:** {stats['avg_chars_per_book']:,} characters
- **Format:** Plain text (.txt files)
- **Language:** English (lowercase)

## Dataset Structure

### Books Included

Each `.txt` file contains the complete text of one book:

{file_list_md}

### Data Fields

- **text:** Complete book text (lowercase, cleaned)
- **filename:** Project Gutenberg ID

### Data Format

All files are plain UTF-8 text:
- Lowercase characters only
- Punctuation and structure preserved
- Paragraph breaks maintained
- No chapter headings or non-narrative text

## Usage

### Load with `datasets` library

```python
from datasets import load_dataset

# Load entire corpus
corpus = load_dataset("contextlab/{author}-corpus")

# Iterate through books
for book in corpus['train']:
    print(f"Book length: {{len(book['text']):,}} characters")
    print(book['text'][:200])  # First 200 characters
    print()
```

### Load specific file

```python
# Load single book by filename
dataset = load_dataset(
    "contextlab/{author}-corpus",
    data_files="54.txt"  # Specific Gutenberg ID
)

text = dataset['train'][0]['text']
print(f"Loaded {{len(text):,}} characters")
```

### Download files directly

```python
from huggingface_hub import hf_hub_download

# Download one book
file_path = hf_hub_download(
    repo_id="contextlab/{author}-corpus",
    filename="54.txt",
    repo_type="dataset"
)

with open(file_path, 'r') as f:
    text = f.read()
```

### Use for training language models

```python
from datasets import load_dataset
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments

# Load corpus
corpus = load_dataset("contextlab/{author}-corpus")

# Combine all books into single text
full_text = " ".join([book['text'] for book in corpus['train']])

# Tokenize
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

def tokenize_function(examples):
    return tokenizer(examples['text'], truncation=True, max_length=1024)

tokenized = corpus.map(tokenize_function, batched=True, remove_columns=['text'])

# Initialize model
model = GPT2LMHeadModel.from_pretrained("gpt2")

# Set up training
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=10,
    per_device_train_batch_size=8,
    save_steps=1000,
)

# Train
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized['train']
)

trainer.train()
```

### Analyze text statistics

```python
from datasets import load_dataset
import numpy as np

corpus = load_dataset("contextlab/{author}-corpus")

# Calculate statistics
lengths = [len(book['text']) for book in corpus['train']]

print(f"Books: {{len(lengths)}}")
print(f"Total characters: {{sum(lengths):,}}")
print(f"Mean length: {{np.mean(lengths):,.0f}} characters")
print(f"Std length: {{np.std(lengths):,.0f}} characters")
print(f"Min length: {{min(lengths):,}} characters")
print(f"Max length: {{max(lengths):,}} characters")
```

## Dataset Creation

### Source Data

All texts sourced from [Project Gutenberg](https://www.gutenberg.org/), a library of over 70,000 free eBooks in the public domain.

**Project Gutenberg Links:**
- Books identified by Gutenberg ID numbers (filenames)
- Example: `54.txt` corresponds to https://www.gutenberg.org/ebooks/54
- All works are in the public domain

### Preprocessing Pipeline

The raw Project Gutenberg texts underwent the following preprocessing:

1. **Header/footer removal:** Project Gutenberg license text and metadata removed
2. **Lowercase conversion:** All text converted to lowercase for stylometry
3. **Chapter heading removal:** Chapter titles and numbering removed
4. **Non-narrative text removal:** Tables of contents, dedications, etc. removed
5. **Encoding normalization:** Converted to UTF-8
6. **Structure preservation:** Paragraph breaks and punctuation maintained

**Why lowercase?** Stylometric analysis focuses on word choice, syntax, and style rather than capitalization patterns. Lowercase normalization removes this variable.

**Preprocessing code:** Available at https://github.com/ContextLab/llm-stylometry

## Considerations for Using This Dataset

### Known Limitations

- **Historical language:** Reflects {metadata['period']} vocabulary, grammar, and cultural context
- **Lowercase only:** All text converted to lowercase (not suitable for case-sensitive analysis)
- **Incomplete corpus:** May not include all of {metadata['full_name']}'s writings (only public domain works on Gutenberg)
- **Cleaning artifacts:** Some formatting irregularities may remain from Gutenberg source
- **Public domain only:** Limited to works published before copyright restrictions

### Intended Use Cases

- **Stylometry research:** Authorship attribution, style analysis
- **Language modeling:** Training author-specific models
- **Literary analysis:** Computational study of {metadata['full_name']}'s writing
- **Historical NLP:** {metadata['period']} language patterns
- **Educational:** Teaching computational text analysis

### Out-of-Scope Uses

- Case-sensitive text analysis
- Modern language applications
- Factual information retrieval
- Complete scholarly editions (use academic sources)

## Citation

If you use this dataset in your research, please cite:

```bibtex
@article{{StroEtal25,
  title={{A Stylometric Application of Large Language Models}},
  author={{Stropkay, Harrison F. and Chen, Jiayi and Jabelli, Mohammad J. L. and Rockmore, Daniel N. and Manning, Jeremy R.}},
  journal={{arXiv preprint arXiv:XXXX.XXXXX}},
  year={{2025}}
}}
```

## Additional Information

### Dataset Curator

[ContextLab](https://www.context-lab.com/), Dartmouth College

### Licensing

MIT License - Free to use with attribution

### Contact

- **Paper & Code:** https://github.com/ContextLab/llm-stylometry
- **Issues:** https://github.com/ContextLab/llm-stylometry/issues
- **Contact:** Jeremy R. Manning (jeremy.r.manning@dartmouth.edu)

### Related Resources

Explore datasets for all 8 authors in the study:
- [Jane Austen](https://huggingface.co/datasets/contextlab/austen-corpus)
- [L. Frank Baum](https://huggingface.co/datasets/contextlab/baum-corpus)
- [Charles Dickens](https://huggingface.co/datasets/contextlab/dickens-corpus)
- [F. Scott Fitzgerald](https://huggingface.co/datasets/contextlab/fitzgerald-corpus)
- [Herman Melville](https://huggingface.co/datasets/contextlab/melville-corpus)
- [Ruth Plumly Thompson](https://huggingface.co/datasets/contextlab/thompson-corpus)
- [Mark Twain](https://huggingface.co/datasets/contextlab/twain-corpus)
- [H.G. Wells](https://huggingface.co/datasets/contextlab/wells-corpus)

### Trained Models

Author-specific GPT-2 models trained on these corpora will be available after training completes:
- https://huggingface.co/contextlab (browse all models)
"""

    return card


def main():
    parser = argparse.ArgumentParser(
        description='Generate HuggingFace dataset card for author corpus'
    )
    parser.add_argument(
        '--author',
        required=True,
        choices=list(AUTHOR_METADATA.keys()),
        help='Author name'
    )
    parser.add_argument(
        '--data-dir',
        required=True,
        help='Path to data directory (e.g., data/cleaned/baum)'
    )
    parser.add_argument(
        '--output',
        help='Output path for dataset card (default: {data_dir}/README.md)'
    )

    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    if not data_dir.exists():
        print(f"ERROR: Data directory not found: {data_dir}")
        return 1

    # Verify has text files
    txt_files = list(data_dir.glob('*.txt'))
    if not txt_files:
        print(f"ERROR: No .txt files found in {data_dir}")
        return 1

    # Generate dataset card
    print(f"Generating dataset card for {args.author}...")
    print(f"Data directory: {data_dir}")
    print(f"Found {len(txt_files)} text files")

    card = generate_dataset_card(args.author, data_dir)

    # Determine output path
    output_path = Path(args.output) if args.output else data_dir / 'README.md'

    # Write dataset card
    with open(output_path, 'w') as f:
        f.write(card)

    print(f"Dataset card saved to: {output_path}")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
