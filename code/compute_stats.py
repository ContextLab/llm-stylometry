#!/usr/bin/env python
"""
Compute statistics for LLM stylometry paper reproduction.
"""

import pickle
import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path
from constants import AUTHORS

def load_data(data_path='data/model_results.pkl', variant=None):
    """
    Load and filter model results by variant.

    Args:
        data_path: Path to consolidated results pickle file
        variant: One of ['content', 'function', 'pos'] or None for baseline

    Returns:
        DataFrame filtered to specified variant
    """
    with open(data_path, 'rb') as f:
        df = pickle.load(f)

    # Filter by variant
    if variant is None:
        # Baseline: exclude any models with variant column set
        if 'variant' in df.columns:
            df = df[df['variant'].isna()].copy()
    else:
        # Specific variant: filter to that variant
        if 'variant' not in df.columns:
            raise ValueError(f"No variant column in data. Cannot filter for variant '{variant}'")
        df = df[df['variant'] == variant].copy()

    return df


def find_twain_threshold_epoch(df, p_threshold=0.001):
    """
    Find the epoch where Twain model's p-value first drops below threshold.
    This corresponds to t-threshold of 3.291 for p < 0.001.
    """
    # Filter for Twain models comparing Twain vs other authors
    twain_df = df[df['train_author'] == 'twain'].copy()

    # Get unique epochs sorted
    epochs = sorted(twain_df['epochs_completed'].unique())

    for epoch in epochs:
        epoch_df = twain_df[twain_df['epochs_completed'] == epoch]

        # Get self losses (Twain model on Twain text)
        self_losses = epoch_df[epoch_df['loss_dataset'] == 'twain']['loss_value'].values

        # Get other losses (Twain model on other authors' texts)
        other_authors = [a for a in AUTHORS if a != 'twain']
        other_losses = epoch_df[epoch_df['loss_dataset'].isin(other_authors)]['loss_value'].values

        if len(self_losses) >= 10 and len(other_losses) >= 70:
            # Perform t-test (other vs self)
            t_stat, p_value = stats.ttest_ind(other_losses, self_losses, equal_var=False)

            if p_value < p_threshold:
                return epoch, t_stat, p_value

    return None, None, None


def compute_average_t_test(df, epoch=500):
    """
    Compute t-test comparing average t-values across seeds to 0.
    For each seed, compute average t-statistic across all authors.
    This reproduces the test on line 230 of the paper.
    """
    # For each seed, get the t-statistics for all authors
    seed_avg_t_stats = []

    for seed in range(10):
        author_t_stats = []

        for author in AUTHORS:
            # Get all data for this author-seed combination
            model_name = f"{author}_tokenizer=gpt2_seed={seed}"
            model_df = df[df['model_name'] == model_name]

            # Get data at the specified epoch (or closest if not exact)
            epoch_data = model_df[model_df['epochs_completed'] <= epoch].groupby('loss_dataset').tail(1)

            # Get self losses
            self_losses = epoch_data[epoch_data['loss_dataset'] == author]['loss_value'].values

            # Get other losses
            other_authors = [a for a in AUTHORS if a != author]
            other_losses = epoch_data[epoch_data['loss_dataset'].isin(other_authors)]['loss_value'].values

            if len(self_losses) > 0 and len(other_losses) > 0:
                # Use mean values if we only have one sample
                if len(self_losses) == 1:
                    # Compute t-statistic using difference of means and std of others
                    mean_diff = np.mean(other_losses) - self_losses[0]
                    std_other = np.std(other_losses)
                    if std_other > 0:
                        t_stat = mean_diff / (std_other / np.sqrt(len(other_losses)))
                        author_t_stats.append(t_stat)
                else:
                    t_stat, _ = stats.ttest_ind(other_losses, self_losses, equal_var=False)
                    if not np.isnan(t_stat):
                        author_t_stats.append(t_stat)

        # Average t-statistic across authors for this seed
        if len(author_t_stats) == len(AUTHORS):
            seed_avg_t_stats.append(np.mean(author_t_stats))

    # Test if mean t-statistic is significantly different from 0
    if len(seed_avg_t_stats) == 10:
        t_stat, p_value = stats.ttest_1samp(seed_avg_t_stats, 0)
        return t_stat, p_value, len(seed_avg_t_stats) - 1

    return None, None, None


def generate_author_comparison_table(df):
    """
    Generate table of t-tests comparing each author's model losses.
    This reproduces Table 1 in the paper.
    """
    # Get final epoch data
    final_df = df.groupby(['train_author', 'loss_dataset', 'seed']).tail(1)

    # Use the same author order as in the figures
    author_order = ['baum', 'thompson', 'austen', 'dickens', 'fitzgerald', 'melville', 'twain', 'wells']

    results = []
    for author in author_order:
        author_df = final_df[final_df['train_author'] == author]

        # Get self losses (model trained on author, tested on same author)
        self_losses = author_df[author_df['loss_dataset'] == author]['loss_value'].values

        # Get other losses (model trained on author, tested on other authors)
        other_authors = [a for a in AUTHORS if a != author]
        other_losses = author_df[author_df['loss_dataset'].isin(other_authors)]['loss_value'].values

        if len(self_losses) >= 10 and len(other_losses) >= 70:
            # Perform t-test (other vs self)
            t_result = stats.ttest_ind(other_losses, self_losses, equal_var=False)

            results.append({
                'Model': author.capitalize(),
                't-stat': f'{t_result.statistic:.2f}',
                'df': f'{t_result.df:.2f}',
                'p-value': f'{t_result.pvalue:.2e}'
            })

    return pd.DataFrame(results)


def main():
    """Main function to compute and display all statistics."""
    import argparse

    parser = argparse.ArgumentParser(description='Compute statistics for LLM stylometry')
    parser.add_argument(
        '--variant',
        choices=['content', 'function', 'pos'],
        default=None,
        help='Analysis variant to compute stats for (default: baseline)'
    )
    parser.add_argument(
        '--data',
        default='data/model_results.pkl',
        help='Path to model results file (default: data/model_results.pkl)'
    )

    args = parser.parse_args()

    # Update header to show variant
    variant_label = f" (Variant: {args.variant})" if args.variant else " (Baseline)"
    print("=" * 60)
    print(f"LLM Stylometry Statistical Analysis{variant_label}")
    print("=" * 60)

    # Load data with variant filter
    print("\nLoading data...")
    df = load_data(data_path=args.data, variant=args.variant)

    # 1. Find Twain threshold epoch
    print("\n1. Twain Model P-Threshold Analysis")
    print("-" * 40)
    epoch, t_stat, p_value = find_twain_threshold_epoch(df)
    if epoch is not None:
        print(f"First epoch where p < 0.001: {epoch}")
        print(f"t-statistic at epoch {epoch}: {t_stat:.3f}")
        print(f"p-value at epoch {epoch}: {p_value:.3e}")
    else:
        print("Threshold not reached within training epochs")

    # 2. Average t-test at final epoch
    print("\n2. Average T-Test Across Authors (Epoch 500)")
    print("-" * 40)
    t_stat, p_value, df_val = compute_average_t_test(df, epoch=500)
    if t_stat is not None:
        print(f"t({df_val}) = {t_stat:.3f}, p = {p_value:.2e}")

        # Format p-value in scientific notation
        if p_value < 1e-10:
            exponent = int(np.floor(np.log10(p_value)))
            mantissa = p_value / (10 ** exponent)
            print(f"(p-value in scientific notation: {mantissa:.1f} × 10^{exponent})")
    else:
        print("Insufficient data for t-test")

    # 3. Author comparison table
    print("\n3. Author Model Comparison Table (Table 1)")
    print("-" * 40)
    table = generate_author_comparison_table(df)
    print("\n" + table.to_string(index=False))

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()