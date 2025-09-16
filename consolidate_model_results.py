"""Script to consolidate all model results into a single DataFrame."""

import pandas as pd
import json
from pathlib import Path
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def consolidate_model_results():
    """Parse all model directories into a single DataFrame and save as pickle."""

    models_dir = Path("models")
    data_dir = Path("data")

    if not models_dir.exists():
        logger.error("Models directory not found")
        return

    # Get all model directories
    model_dirs = [d for d in models_dir.iterdir() if d.is_dir() and "tokenizer=" in d.name]
    logger.info(f"Found {len(model_dirs)} model directories")

    all_results = []

    for model_dir in tqdm(model_dirs, desc="Processing models"):
        model_name = model_dir.name

        # Parse model name to extract author, tokenizer, and seed
        parts = model_name.split("_")
        author = parts[0]
        tokenizer_part = "_".join(parts[1:])  # Handle tokenizer=gpt2_seed=X
        tokenizer = tokenizer_part.split("=")[1].split("_")[0]
        seed = int(tokenizer_part.split("seed=")[1])

        # Read config.json if it exists
        config_path = model_dir / "config.json"
        config_data = {}
        if config_path.exists():
            with open(config_path, 'r') as f:
                config_data = json.load(f)

        # Read generation_config.json if it exists
        gen_config_path = model_dir / "generation_config.json"
        gen_config_data = {}
        if gen_config_path.exists():
            with open(gen_config_path, 'r') as f:
                gen_config_data = json.load(f)

        # Read loss_logs.csv if it exists
        loss_logs_path = model_dir / "loss_logs.csv"
        if loss_logs_path.exists():
            loss_df = pd.read_csv(loss_logs_path)

            # Add model metadata to each row
            loss_df['model_name'] = model_name
            loss_df['author'] = author
            loss_df['tokenizer'] = tokenizer

            # Add config data as JSON string (to preserve nested structure)
            loss_df['model_config'] = json.dumps(config_data)
            loss_df['generation_config'] = json.dumps(gen_config_data)

            # Add model checkpoint path (for future downloads)
            loss_df['checkpoint_path'] = str(model_dir)

            all_results.append(loss_df)
        else:
            logger.warning(f"No loss_logs.csv found in {model_dir}")

    if all_results:
        # Combine all results into a single DataFrame
        combined_df = pd.concat(all_results, ignore_index=True)

        # Sort by author, seed, and epochs_completed for better organization
        combined_df = combined_df.sort_values(['author', 'seed', 'epochs_completed'])

        # Save as pickle file
        output_path = data_dir / "model_results.pkl"
        combined_df.to_pickle(output_path)

        # Also save as CSV for easy inspection
        csv_path = data_dir / "model_results.csv"
        # For CSV, we'll exclude the config columns as they're JSON strings
        csv_df = combined_df.drop(columns=['model_config', 'generation_config'])
        csv_df.to_csv(csv_path, index=False)

        logger.info(f"Consolidated {len(combined_df)} rows from {len(model_dirs)} models")
        logger.info(f"Saved to {output_path} and {csv_path}")

        # Print summary statistics
        print("\nSummary Statistics:")
        print(f"Total models: {combined_df['model_name'].nunique()}")
        print(f"Total rows: {len(combined_df)}")
        print(f"Authors: {sorted(combined_df['author'].unique())}")
        print(f"Seeds: {sorted(combined_df['seed'].unique())}")
        print(f"\nRows per model:")
        print(combined_df.groupby('model_name').size().describe())

        # Check for any missing combinations
        expected_authors = ["baum", "thompson", "dickens", "melville", "wells", "austen", "fitzgerald", "twain"]
        expected_seeds = list(range(10))

        print("\nChecking for missing model combinations:")
        missing = []
        for author in expected_authors:
            for seed in expected_seeds:
                model_name = f"{author}_tokenizer=gpt2_seed={seed}"
                if model_name not in combined_df['model_name'].unique():
                    missing.append(model_name)

        if missing:
            print(f"Missing models: {missing}")
        else:
            print("All expected models found!")

        return combined_df
    else:
        logger.error("No results to consolidate")
        return None


if __name__ == "__main__":
    df = consolidate_model_results()
    if df is not None:
        print(f"\nDataFrame shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
        print(f"\nSample of data:")
        print(df.head())