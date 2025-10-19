"""
Generate synthetic loss_logs.csv files with known duplicate patterns for testing.

This module provides utilities to create realistic test data that mimics the
duplicate epoch and spurious epoch 0 issues that occur during training resume.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import shutil


def generate_loss_logs_csv(
    output_path,
    seed=0,
    train_author='baum',
    max_epochs=50,
    duplicate_epochs=None,
    spurious_epoch_0_at=None,
    eval_datasets=None
):
    """
    Generate a synthetic loss_logs.csv file with optional duplicates.

    Args:
        output_path: Path where CSV file should be saved
        seed: Random seed for reproducibility
        train_author: Author name for training
        max_epochs: Maximum number of epochs to generate
        duplicate_epochs: List of epoch numbers to duplicate (e.g., [27, 35])
        spurious_epoch_0_at: List of epoch numbers after which to add spurious epoch 0 (e.g., [15, 25])
        eval_datasets: List of evaluation dataset names (default: standard 8 authors + 3 oz datasets)

    Returns:
        Path to generated CSV file
    """
    if eval_datasets is None:
        eval_datasets = [
            'baum', 'thompson', 'dickens', 'melville', 'wells',
            'austen', 'fitzgerald', 'twain',
            'non_oz_baum', 'non_oz_thompson', 'contested'
        ]

    np.random.seed(seed)

    rows = []

    # Initial epoch 0 (legitimate pre-training evaluation)
    for dataset in eval_datasets:
        rows.append({
            'seed': seed,
            'train_author': train_author,
            'epochs_completed': 0,
            'loss_dataset': dataset,
            'loss_value': np.random.uniform(10.5, 11.0)
        })

    # Generate training epochs
    current_loss = 10.0
    for epoch in range(1, max_epochs + 1):
        # Simulate loss decay
        current_loss = max(1.0, current_loss * 0.95 + np.random.normal(0, 0.05))

        # Training loss
        rows.append({
            'seed': seed,
            'train_author': train_author,
            'epochs_completed': epoch,
            'loss_dataset': 'train',
            'loss_value': current_loss
        })

        # Evaluation losses
        for dataset in eval_datasets:
            eval_loss = current_loss + np.random.normal(0, 0.2)
            rows.append({
                'seed': seed,
                'train_author': train_author,
                'epochs_completed': epoch,
                'loss_dataset': dataset,
                'loss_value': max(1.0, eval_loss)
            })

        # Add spurious epoch 0 entries if requested
        if spurious_epoch_0_at and epoch in spurious_epoch_0_at:
            for dataset in eval_datasets:
                # Use same loss values as current epoch (mimics re-evaluation)
                spurious_loss = rows[-len(eval_datasets)]['loss_value'] if dataset == eval_datasets[0] else rows[-1]['loss_value']
                rows.append({
                    'seed': seed,
                    'train_author': train_author,
                    'epochs_completed': 0,
                    'loss_dataset': dataset,
                    'loss_value': spurious_loss
                })

        # Add duplicate epoch entries if requested
        if duplicate_epochs and epoch in duplicate_epochs:
            # Re-add all entries for this epoch with slightly different loss values
            duplicate_train_loss = current_loss + np.random.normal(0, 0.001)
            rows.append({
                'seed': seed,
                'train_author': train_author,
                'epochs_completed': epoch,
                'loss_dataset': 'train',
                'loss_value': duplicate_train_loss
            })

            for dataset in eval_datasets:
                duplicate_eval_loss = eval_loss + np.random.normal(0, 0.001)
                rows.append({
                    'seed': seed,
                    'train_author': train_author,
                    'epochs_completed': epoch,
                    'loss_dataset': dataset,
                    'loss_value': max(1.0, duplicate_eval_loss)
                })

    # Create DataFrame and save
    df = pd.DataFrame(rows)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    return output_path


def create_test_model_directory(
    base_dir,
    author='baum',
    seed=0,
    tokenizer='gpt2',
    variant=None,
    max_epochs=50,
    duplicate_epochs=None,
    spurious_epoch_0_at=None
):
    """
    Create a complete test model directory with loss_logs.csv.

    Args:
        base_dir: Base directory for test models
        author: Training author name
        seed: Random seed
        tokenizer: Tokenizer name
        variant: Variant name (content, function, pos) or None for baseline
        max_epochs: Maximum epochs to generate
        duplicate_epochs: List of epoch numbers to duplicate
        spurious_epoch_0_at: List of epochs after which to add spurious epoch 0

    Returns:
        Path to created model directory
    """
    # Construct model directory name
    if variant:
        model_name = f"{author}_variant={variant}_tokenizer={tokenizer}_seed={seed}"
    else:
        model_name = f"{author}_tokenizer={tokenizer}_seed={seed}"

    model_dir = Path(base_dir) / model_name
    model_dir.mkdir(parents=True, exist_ok=True)

    # Generate loss logs
    loss_logs_path = model_dir / 'loss_logs.csv'
    generate_loss_logs_csv(
        output_path=loss_logs_path,
        seed=seed,
        train_author=author,
        max_epochs=max_epochs,
        duplicate_epochs=duplicate_epochs,
        spurious_epoch_0_at=spurious_epoch_0_at
    )

    return model_dir


def cleanup_test_directory(test_dir):
    """Remove test directory and all contents."""
    test_dir = Path(test_dir)
    if test_dir.exists():
        shutil.rmtree(test_dir)
