#!/usr/bin/env python
"""
Consolidate model training results from individual loss_logs.csv files into a single DataFrame.

This script reads all loss_logs.csv files from the models/ directory and combines them
into a single pandas DataFrame, adding model configuration information. The consolidated
results are saved as data/model_results.pkl for use by visualization scripts.
"""

import os
import json
import pandas as pd
from pathlib import Path
from tqdm import tqdm


def consolidate_model_results(models_dir='models', output_path=None, save_csv=False, variant=None):
    """
    Consolidate all model training results into a single DataFrame.

    Args:
        models_dir: Directory containing trained models
        output_path: Path to save consolidated pickle file (auto-determined if None)
        save_csv: Also save as CSV for debugging (default: False)
        variant: Filter by variant ('content', 'function', 'pos') or None for baseline

    Returns:
        Consolidated DataFrame with all model results
    """
    models_path = Path(models_dir)

    if not models_path.exists():
        raise FileNotFoundError(f"Models directory not found: {models_dir}")

    # Auto-determine output path based on variant
    if output_path is None:
        if variant:
            output_path = f'data/model_results_{variant}.pkl'
        else:
            output_path = 'data/model_results.pkl'

    all_results = []

    # Find all model directories
    all_model_dirs = sorted([d for d in models_path.iterdir() if d.is_dir()])

    # Filter by variant
    if variant:
        model_dirs = [d for d in all_model_dirs if f'_variant={variant}_' in d.name]
        variant_label = f"{variant} variant"
    else:
        model_dirs = [d for d in all_model_dirs if '_variant=' not in d.name]
        variant_label = "baseline"

    if not model_dirs:
        raise ValueError(f"No {variant_label} model directories found in {models_dir}")

    print(f"Found {len(model_dirs)} {variant_label} model directories to consolidate")

    for model_dir in tqdm(model_dirs, desc="Consolidating models"):
        # Parse model directory name
        dir_name = model_dir.name
        parts = dir_name.split('_')

        # Extract author, variant, tokenizer, and seed from directory name
        # Baseline format: {author}_tokenizer={tokenizer}_seed={seed}
        # Variant format: {author}_variant={variant}_tokenizer={tokenizer}_seed={seed}
        author = parts[0]

        # Find variant, tokenizer, and seed
        model_variant = None
        tokenizer = None
        seed = None
        for part in parts[1:]:
            if part.startswith('variant='):
                model_variant = part.split('=')[1]
            elif part.startswith('tokenizer='):
                tokenizer = part.split('=')[1]
            elif part.startswith('seed='):
                seed = int(part.split('=')[1])

        if tokenizer is None or seed is None:
            print(f"Warning: Could not parse directory name: {dir_name}")
            continue

        # Read loss logs
        loss_logs_path = model_dir / 'loss_logs.csv'
        if not loss_logs_path.exists():
            print(f"Warning: No loss_logs.csv found in {model_dir}")
            continue

        # Read the CSV file
        df = pd.read_csv(loss_logs_path)
        original_rows = len(df)

        # Deduplication: Remove spurious epoch 0 entries and duplicate epochs
        # Step 1: Remove spurious epoch 0 entries (keep only first occurrence per model)
        # Legitimate epoch 0 entries appear only once at start (initial evaluation)
        # Any subsequent epoch 0 entries are from resume operations
        epoch_0_mask = df['epochs_completed'] == 0
        if epoch_0_mask.any():
            # Keep only first N rows with epoch 0 (typically 11 rows for train + eval datasets)
            # Using 15 as safe upper bound to handle models with extra eval datasets
            first_epoch_0_indices = df[epoch_0_mask].index[:15]
            # Remove all other epoch 0 entries
            df = df[(~epoch_0_mask) | (df.index.isin(first_epoch_0_indices))].copy()

        # Step 2: Remove duplicate epochs (keep only last occurrence)
        # When training resumes after interruption, last completed epoch may be re-run
        # We keep 'last' because it represents the most recent training run
        df = df.drop_duplicates(
            subset=['seed', 'train_author', 'epochs_completed', 'loss_dataset'],
            keep='last'
        )

        # Log duplicate removal statistics
        removed_rows = original_rows - len(df)
        if removed_rows > 0:
            print(f"  Removed {removed_rows} duplicate/spurious rows from {dir_name}")

        # Add model metadata
        df['model_name'] = dir_name
        df['author'] = author
        df['variant'] = model_variant  # None for baseline, variant name for variant models
        df['tokenizer'] = tokenizer
        df['checkpoint_path'] = str(model_dir)

        # Read config files if they exist
        config_path = model_dir / 'config.json'
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = json.load(f)
                df['model_config'] = json.dumps(config)
        else:
            df['model_config'] = None

        gen_config_path = model_dir / 'generation_config.json'
        if gen_config_path.exists():
            with open(gen_config_path, 'r') as f:
                gen_config = json.load(f)
                df['generation_config'] = json.dumps(gen_config)
        else:
            df['generation_config'] = None

        all_results.append(df)

    # Handle case where no valid models were found
    if not all_results:
        print("Warning: No valid model data found to consolidate")
        # Return empty DataFrame with expected schema
        return pd.DataFrame(columns=[
            'seed', 'train_author', 'epochs_completed', 'loss_dataset',
            'loss_value', 'model_name', 'author', 'variant', 'tokenizer',
            'model_config', 'generation_config', 'checkpoint_path'
        ])

    # Combine all dataframes
    consolidated_df = pd.concat(all_results, ignore_index=True)

    # Ensure column order matches expected format
    expected_columns = [
        'seed', 'train_author', 'epochs_completed', 'loss_dataset',
        'loss_value', 'model_name', 'author', 'variant', 'tokenizer',
        'model_config', 'generation_config', 'checkpoint_path'
    ]

    # Reorder columns if they all exist
    available_columns = [col for col in expected_columns if col in consolidated_df.columns]
    consolidated_df = consolidated_df[available_columns]

    # Save as pickle
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    consolidated_df.to_pickle(output_path)

    print(f"\nConsolidation complete!")
    print(f"Total records: {len(consolidated_df)}")
    print(f"Unique models: {consolidated_df['model_name'].nunique()}")
    print(f"Saved to: {output_path}")

    # Optionally save as CSV for debugging/inspection
    if save_csv:
        csv_path = output_path.with_suffix('.csv')
        consolidated_df.to_csv(csv_path, index=False)
        print(f"Also saved CSV for inspection: {csv_path}")

    # Print summary statistics
    print("\nSummary by author and variant:")
    if 'variant' in consolidated_df.columns:
        # Use dropna=False to include None (baseline) values
        summary = consolidated_df.groupby(['train_author', 'variant'], dropna=False)['seed'].nunique()
        for (author, variant), num_seeds in summary.items():
            variant_label = "baseline" if pd.isna(variant) else variant
            print(f"  {author} ({variant_label}): {num_seeds} seeds")
    else:
        # Fallback for old data without variant column
        summary = consolidated_df.groupby('train_author')['seed'].nunique()
        for author, num_seeds in summary.items():
            print(f"  {author}: {num_seeds} seeds")

    return consolidated_df


def main():
    """Main entry point for the consolidation script."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Consolidate model training results into a single DataFrame'
    )
    parser.add_argument(
        '--models-dir',
        default='models',
        help='Directory containing trained models (default: models)'
    )
    parser.add_argument(
        '--output',
        default=None,
        help='Output path for consolidated pickle file (default: auto-determined based on variant)'
    )
    parser.add_argument(
        '--save-csv',
        action='store_true',
        help='Also save as CSV for debugging'
    )
    parser.add_argument(
        '--variant',
        choices=['content', 'function', 'pos'],
        default=None,
        help='Filter by variant (content, function, pos) or omit for baseline'
    )

    args = parser.parse_args()

    try:
        df = consolidate_model_results(
            args.models_dir,
            args.output,
            args.save_csv,
            args.variant
        )
        return 0
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == '__main__':
    import sys
    sys.exit(main())