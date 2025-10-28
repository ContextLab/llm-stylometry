#!/usr/bin/env python3
"""
Generate HuggingFace model cards for author-specific models.

Creates professional README.md files for HuggingFace model repositories.
"""

import json
import argparse
from pathlib import Path
import pandas as pd


# Author metadata
AUTHOR_METADATA = {
    'austen': {
        'full_name': 'Jane Austen',
        'years': '1775-1817',
        'period': '19th-century England',
        'example_prompt': 'it is a truth universally acknowledged',
        'notable_works': 'Pride and Prejudice, Sense and Sensibility, Emma',
    },
    'baum': {
        'full_name': 'L. Frank Baum',
        'years': '1856-1919',
        'period': 'late 19th to early 20th century America',
        'example_prompt': 'dorothy lived in the midst of',
        'notable_works': 'The Wonderful Wizard of Oz series',
    },
    'dickens': {
        'full_name': 'Charles Dickens',
        'years': '1812-1870',
        'period': 'Victorian England',
        'example_prompt': 'it was the best of times',
        'notable_works': 'A Tale of Two Cities, Great Expectations, Oliver Twist',
    },
    'fitzgerald': {
        'full_name': 'F. Scott Fitzgerald',
        'years': '1896-1940',
        'period': 'Jazz Age America',
        'example_prompt': 'in my younger and more vulnerable years',
        'notable_works': 'The Great Gatsby, Tender Is the Night',
    },
    'melville': {
        'full_name': 'Herman Melville',
        'years': '1819-1891',
        'period': '19th-century America',
        'example_prompt': 'call me ishmael',
        'notable_works': 'Moby-Dick, Bartleby the Scrivener',
    },
    'thompson': {
        'full_name': 'Ruth Plumly Thompson',
        'years': '1891-1976',
        'period': 'early-to-mid 20th century America',
        'example_prompt': 'once upon a time in the land of',
        'notable_works': 'The Oz book series (books 15-35)',
    },
    'twain': {
        'full_name': 'Mark Twain',
        'years': '1835-1910',
        'period': '19th-century America',
        'example_prompt': 'you don\'t know about me',
        'notable_works': 'Adventures of Huckleberry Finn, Tom Sawyer',
    },
    'wells': {
        'full_name': 'H.G. Wells',
        'years': '1866-1946',
        'period': 'late 19th to early 20th century England',
        'example_prompt': 'the time traveller',
        'notable_works': 'The Time Machine, The War of the Worlds, The Invisible Man',
    },
}


def calculate_param_count(config):
    """Calculate total parameter count from config."""
    vocab_size = config['vocab_size']
    n_embd = config['n_embd']
    n_layer = config['n_layer']
    n_positions = config['n_positions']

    # Embedding layers
    wte = vocab_size * n_embd  # Token embeddings
    wpe = n_positions * n_embd  # Position embeddings

    # Transformer blocks (per layer)
    # Attention: 3 * n_embd * n_embd (Q, K, V) + n_embd * n_embd (output proj)
    # MLP: n_embd * 4*n_embd + 4*n_embd * n_embd
    # LayerNorm: 2 * n_embd (per layer, 2 norms)
    per_layer = (
        4 * n_embd * n_embd +  # Attention
        2 * (n_embd * 4 * n_embd) +  # MLP
        2 * n_embd  # LayerNorm
    )

    transformer_params = n_layer * per_layer
    final_ln = n_embd
    lm_head = vocab_size * n_embd  # Tied with wte usually, but count separately

    # Note: This is approximate; actual count depends on implementation details
    total = wte + wpe + transformer_params + final_ln
    return total


def get_model_stats(model_dir):
    """Extract statistics from trained model."""
    # Load config
    with open(model_dir / 'config.json') as f:
        config = json.load(f)

    # Load training logs
    logs = pd.read_csv(model_dir / 'loss_logs.csv')
    train_logs = logs[logs['loss_dataset'] == 'train']

    final_loss = train_logs['loss_value'].iloc[-1]
    epochs_trained = train_logs['epochs_completed'].max()

    # Calculate parameters
    param_count = calculate_param_count(config)

    return {
        'config': config,
        'final_loss': final_loss,
        'epochs_trained': epochs_trained,
        'param_count': param_count
    }


def count_training_tokens(author):
    """Estimate training tokens from cleaned data."""
    author_dir = Path(f'data/cleaned/{author}')

    if not author_dir.exists():
        return "~640,000"  # Default estimate

    books = list(author_dir.glob('*.txt'))
    total_chars = sum(len(open(book, encoding='utf-8', errors='ignore').read()) for book in books)

    # GPT-2 tokenizer: roughly 1 token per 4 characters
    approx_tokens = total_chars // 4

    return f"{approx_tokens:,}"


def generate_model_card(author, model_dir):
    """
    Generate HuggingFace model card for author.

    Args:
        author: Author name (lowercase, e.g., 'baum')
        model_dir: Path to model directory

    Returns:
        Model card markdown string
    """
    metadata = AUTHOR_METADATA[author]
    stats = get_model_stats(Path(model_dir))

    # Build model card
    card = f"""---
language: en
license: mit
tags:
- text-generation
- gpt2
- stylometry
- {author}
- authorship-attribution
- literary-analysis
- computational-linguistics
datasets:
- contextlab/{author}-corpus
library_name: transformers
pipeline_tag: text-generation
---

# GPT-2 {metadata['full_name']} Stylometry Model

<div style="text-align: center;">
  <img src="https://raw.githubusercontent.com/ContextLab/llm-stylometry/main/assets/CDL_Avatar.png" alt="Context Lab" width="200"/>
</div>

## Overview

This model is a GPT-2 language model trained exclusively on the complete works of **{metadata['full_name']}** ({metadata['years']}). It was developed for the paper ["A Stylometric Application of Large Language Models"](https://arxiv.org/abs/2510.21958) (Stropkay et al., 2025).

The model captures {metadata['full_name']}'s unique writing style through intensive training on their complete corpus. By learning the statistical patterns, vocabulary, syntax, and thematic elements characteristic of {author.capitalize()}'s writing, this model enables:

- **Text generation** in the authentic style of {metadata['full_name']}
- **Authorship attribution** through cross-entropy loss comparison
- **Stylometric analysis** of literary works from {metadata['period']}
- **Computational literary studies** exploring {author.capitalize()}'s distinctive voice

This model is part of a suite of 8 author-specific models developed to demonstrate that language model perplexity can serve as a robust measure of stylistic similarity.

**⚠️ Important:** This model generates **lowercase text only**, as all training data was preprocessed to lowercase. Use lowercase prompts for best results.

## Model Details

- **Model type:** GPT-2 (custom compact architecture)
- **Language:** English (lowercase)
- **License:** MIT
- **Author:** {metadata['full_name']} ({metadata['years']})
- **Notable works:** {metadata['notable_works']}
- **Training data:** [{metadata['full_name']} Complete Works](https://huggingface.co/datasets/contextlab/{author}-corpus)
- **Training tokens:** {count_training_tokens(author)}
- **Final training loss:** {stats['final_loss']:.4f}
- **Epochs trained:** {stats['epochs_trained']:,}

### Architecture

| Parameter | Value |
|-----------|-------|
| Layers | {stats['config']['n_layer']} |
| Embedding dimension | {stats['config']['n_embd']} |
| Attention heads | {stats['config']['n_head']} |
| Context length | {stats['config']['n_positions']} tokens |
| Vocabulary size | {stats['config']['vocab_size']:,} (GPT-2 tokenizer) |
| Total parameters | ~{stats['param_count'] / 1e6:.1f}M |

## Usage

### Basic Text Generation

```python
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

# Load model and tokenizer
model = GPT2LMHeadModel.from_pretrained("contextlab/gpt2-{author}")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

# IMPORTANT: Use lowercase prompts (model trained on lowercase text)
prompt = "{metadata['example_prompt']}"
inputs = tokenizer(prompt, return_tensors="pt")

# Generate text
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_length=200,
        do_sample=True,
        temperature=0.8,
        top_p=0.9,
        pad_token_id=tokenizer.eos_token_id
    )

generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(generated_text)
```

**Output:** Generates text in {metadata['full_name']}'s distinctive style (all lowercase).

### Stylometric Analysis

Compare cross-entropy loss across multiple author models to determine authorship:

```python
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

# Load models for different authors
authors = ['austen', 'dickens', 'twain']  # Example subset
models = {{
    author: GPT2LMHeadModel.from_pretrained(f"contextlab/gpt2-{{author}}")
    for author in authors
}}

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Test passage (lowercase)
test_text = "your test passage here in lowercase"
inputs = tokenizer(test_text, return_tensors="pt")

# Compute loss for each model
for author, model in models.items():
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs['input_ids'])
        loss = outputs.loss.item()
    print(f"{{author}}: {{loss:.4f}}")

# Lower loss indicates more similar style (likely author)
```

## Training Procedure

### Dataset

The model was trained on the complete works of {metadata['full_name']} sourced from [Project Gutenberg](https://www.gutenberg.org/). The text was preprocessed to:
- Remove Project Gutenberg headers and footers
- Convert all text to lowercase
- Remove chapter headings and non-narrative text
- Preserve punctuation and structure

See the [{author.capitalize()} corpus dataset](https://huggingface.co/datasets/contextlab/{author}-corpus) for details.

### Hyperparameters

| Parameter | Value |
|-----------|-------|
| Context length | 1,024 tokens |
| Batch size | 16 |
| Learning rate | 5×10⁻⁵ |
| Optimizer | AdamW |
| Training tokens | {count_training_tokens(author)} |
| Epochs | {stats['epochs_trained']:,} |
| Final loss | {stats['final_loss']:.4f} |

### Training Method

The model was initialized with a compact GPT-2 architecture (8 layers, 128-dimensional embeddings) and trained exclusively on {metadata['full_name']}'s works until reaching a training loss of approximately {stats['final_loss']:.4f}. This intensive training enables the model to capture fine-grained stylistic patterns characteristic of {author.capitalize()}'s writing.

See the [GitHub repository](https://github.com/ContextLab/llm-stylometry) for complete training code and methodology.

## Intended Use

### Primary Uses
- **Research:** Stylometric analysis, authorship attribution studies
- **Education:** Demonstrations of computational stylometry
- **Creative:** Generate text in {metadata['full_name']}'s style
- **Analysis:** Compare writing styles across historical periods

### Out-of-Scope Uses
This model is not intended for:
- Factual information retrieval
- Modern language generation
- Tasks requiring uppercase text
- Commercial publication without attribution

## Limitations

- **Lowercase only:** All generated text is lowercase (due to preprocessing)
- **Historical language:** Reflects {metadata['period']} vocabulary and grammar
- **Training data bias:** Limited to {metadata['full_name']}'s published works
- **Small model:** Compact architecture prioritizes training speed over generation quality
- **No factual grounding:** Generates stylistically similar text, not historically accurate content

## Evaluation

This model achieved perfect accuracy (100%) in distinguishing {metadata['full_name']}'s works from seven other classic authors in cross-entropy loss comparisons. See the paper for detailed evaluation results.

## Citation

If you use this model in your research, please cite:

```bibtex
@article{{StroEtal25,
  title={{A Stylometric Application of Large Language Models}},
  author={{Stropkay, Harrison F. and Chen, Jiayi and Jabelli, Mohammad J. L. and Rockmore, Daniel N. and Manning, Jeremy R.}},
  journal={{arXiv preprint arXiv:2510.21958}},
  year={{2025}}
}}
```

## Contact

- **Paper & Code:** https://github.com/ContextLab/llm-stylometry
- **Issues:** https://github.com/ContextLab/llm-stylometry/issues
- **Contact:** Jeremy R. Manning (jeremy.r.manning@dartmouth.edu)
- **Lab:** [Context Lab](https://www.context-lab.com/), Dartmouth College

## Related Models

Explore models for all 8 authors in the study:
- [Jane Austen](https://huggingface.co/contextlab/gpt2-austen)
- [L. Frank Baum](https://huggingface.co/contextlab/gpt2-baum)
- [Charles Dickens](https://huggingface.co/contextlab/gpt2-dickens)
- [F. Scott Fitzgerald](https://huggingface.co/contextlab/gpt2-fitzgerald)
- [Herman Melville](https://huggingface.co/contextlab/gpt2-melville)
- [Ruth Plumly Thompson](https://huggingface.co/contextlab/gpt2-thompson)
- [Mark Twain](https://huggingface.co/contextlab/gpt2-twain)
- [H.G. Wells](https://huggingface.co/contextlab/gpt2-wells)
"""

    return card


def main():
    parser = argparse.ArgumentParser(
        description='Generate HuggingFace model card for author model'
    )
    parser.add_argument(
        '--author',
        required=True,
        choices=list(AUTHOR_METADATA.keys()),
        help='Author name'
    )
    parser.add_argument(
        '--model-dir',
        required=True,
        help='Path to model directory (e.g., models_hf/baum_tokenizer=gpt2)'
    )
    parser.add_argument(
        '--output',
        help='Output path for model card (default: {model_dir}/README.md)'
    )

    args = parser.parse_args()

    model_dir = Path(args.model_dir)

    if not model_dir.exists():
        print(f"ERROR: Model directory not found: {model_dir}")
        return 1

    # Verify required files exist
    if not (model_dir / 'config.json').exists():
        print(f"ERROR: config.json not found in {model_dir}")
        return 1

    if not (model_dir / 'loss_logs.csv').exists():
        print(f"ERROR: loss_logs.csv not found in {model_dir}")
        return 1

    # Generate model card
    print(f"Generating model card for {args.author}...")
    card = generate_model_card(args.author, model_dir)

    # Determine output path
    output_path = Path(args.output) if args.output else model_dir / 'README.md'

    # Write model card
    with open(output_path, 'w') as f:
        f.write(card)

    print(f"Model card saved to: {output_path}")
    print(f"Preview: {output_path}")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
