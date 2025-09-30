#!/usr/bin/env python
"""
Create comprehensive test models for statistical testing.

Requirements for compute_stats.py to work:
1. Twain threshold: ≥10 self-losses, ≥70 other-losses per epoch
2. Average t-test: 10 seeds × 8 authors
3. Author comparison table: ≥10 self-losses, ≥70 other-losses per author

Minimal viable configuration:
- 8 authors (all from constants.py)
- 10 seeds (0-9)
- 4 variants (baseline, content, function, pos)
- Small models (2 layers, 64 embd) for fast training
- 50 epochs each (enough for threshold testing)
- Very small training set for speed
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'code'))

from experiment import Experiment
from constants import AUTHORS, MODELS_DIR, get_data_dir
import argparse
import torch


def create_test_models(
    authors=None,
    seeds=None,
    variants=None,
    n_epochs=50,
    n_train_tokens=5000,
    n_positions=128,
    n_embd=64,
    n_layer=2,
    n_head=2,
    batch_size=4
):
    """
    Create comprehensive test models for statistical testing.

    Args:
        authors: List of author names (default: all 8 from constants)
        seeds: List of seed values (default: 0-9)
        variants: List of variants (default: ['baseline', 'content', 'function', 'pos'])
        n_epochs: Number of epochs to train
        n_train_tokens: Training tokens per model
        n_positions: Context length
        n_embd: Embedding dimension
        n_layer: Number of layers
        n_head: Number of attention heads
        batch_size: Batch size
    """
    from code.constants import AUTHORS

    if authors is None:
        authors = AUTHORS  # All 8: baum, thompson, austen, dickens, fitzgerald, melville, twain, wells

    if seeds is None:
        seeds = list(range(10))  # 0-9 for statistical tests

    if variants is None:
        variants = ['baseline', 'content', 'function', 'pos']

    print("="*60)
    print("Creating Comprehensive Test Models")
    print("="*60)
    print(f"Authors: {len(authors)} - {authors}")
    print(f"Seeds: {len(seeds)} - {seeds}")
    print(f"Variants: {len(variants)} - {variants}")
    print(f"Total models: {len(authors) * len(seeds) * len(variants)}")
    print(f"Epochs per model: {n_epochs}")
    print(f"Training tokens: {n_train_tokens}")
    print(f"\nEstimated time: ~{len(authors) * len(seeds) * len(variants) * 2} minutes (~2min/model)")
    print("="*60)

    response = input("\nProceed with training? [y/N]: ")
    if response.lower() != 'y':
        print("Cancelled.")
        return

    models_created = 0
    models_failed = 0

    for variant_name in variants:
        print(f"\n{'='*60}")
        print(f"Training {variant_name.upper()} variant models")
        print(f"{'='*60}")

        # Convert 'baseline' to None for Experiment
        analysis_variant = None if variant_name == 'baseline' else variant_name

        for author in authors:
            for seed in seeds:
                print(f"\nTraining: {author}, seed={seed}, variant={variant_name}")

                try:
                    exp = Experiment(
                        train_author=author,
                        seed=seed,
                        tokenizer_name="gpt2",
                        analysis_variant=analysis_variant,
                        n_train_tokens=n_train_tokens,
                        n_positions=n_positions,
                        n_embd=n_embd,
                        n_layer=n_layer,
                        n_head=n_head,
                        batch_size=batch_size,
                        stop_criteria={
                            "train_loss": 1.5,
                            "min_epochs": n_epochs,
                            "max_epochs": n_epochs
                        }
                    )

                    train_model(exp)
                    models_created += 1
                    print(f"  ✓ Success ({models_created} total)")

                except Exception as e:
                    print(f"  ✗ Failed: {e}")
                    models_failed += 1

    print(f"\n{'='*60}")
    print(f"Training Complete")
    print(f"{'='*60}")
    print(f"Models created: {models_created}")
    print(f"Models failed: {models_failed}")

    # Consolidate results
    print(f"\nConsolidating results...")
    import subprocess
    result = subprocess.run(
        [sys.executable, 'code/consolidate_model_results.py'],
        capture_output=True,
        text=True
    )

    if result.returncode == 0:
        print("✓ Consolidation successful")
        print(f"  Output: data/model_results.pkl")
    else:
        print(f"✗ Consolidation failed: {result.stderr}")


def main():
    parser = argparse.ArgumentParser(
        description='Create comprehensive test models for statistical testing'
    )
    parser.add_argument(
        '--authors',
        nargs='+',
        default=None,
        help='List of authors (default: all 8)'
    )
    parser.add_argument(
        '--seeds',
        nargs='+',
        type=int,
        default=None,
        help='List of seeds (default: 0-9)'
    )
    parser.add_argument(
        '--variants',
        nargs='+',
        default=None,
        help='List of variants (default: baseline, content, function, pos)'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=50,
        help='Number of epochs per model (default: 50)'
    )
    parser.add_argument(
        '--tokens',
        type=int,
        default=5000,
        help='Training tokens per model (default: 5000)'
    )

    args = parser.parse_args()

    create_test_models(
        authors=args.authors,
        seeds=args.seeds,
        variants=args.variants,
        n_epochs=args.epochs,
        n_train_tokens=args.tokens
    )


if __name__ == '__main__':
    main()
