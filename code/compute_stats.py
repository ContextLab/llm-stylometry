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


def find_threshold_crossing_epochs(df, p_threshold=0.001):
    """
    Find epochs where each author's p-value crosses below threshold.
    Detects authors that start above threshold and later cross below it.

    Returns:
        dict: {author: (epoch, t_stat, p_value)} for authors that cross threshold
    """
    crossing_authors = {}

    for author in AUTHORS:
        author_df = df[df['train_author'] == author].copy()
        epochs = sorted(author_df['epochs_completed'].unique())

        # Track if we've seen above-threshold epochs before crossing
        seen_above_threshold = False

        for epoch in epochs:
            epoch_df = author_df[author_df['epochs_completed'] == epoch]

            # Get self losses
            self_losses = epoch_df[epoch_df['loss_dataset'] == author]['loss_value'].values

            # Get other losses
            other_authors = [a for a in AUTHORS if a != author]
            other_losses = epoch_df[epoch_df['loss_dataset'].isin(other_authors)]['loss_value'].values

            if len(self_losses) >= 10 and len(other_losses) >= 70:
                # Perform t-test (other vs self)
                t_stat, p_value = stats.ttest_ind(other_losses, self_losses, equal_var=False)

                if p_value >= p_threshold:
                    seen_above_threshold = True
                elif seen_above_threshold and p_value < p_threshold:
                    # Crossed threshold!
                    crossing_authors[author] = (epoch, t_stat, p_value)
                    break

    return crossing_authors


def find_average_threshold_crossing(df, p_threshold=0.001):
    """
    Find epoch where average t-statistic across all authors crosses threshold.

    Returns:
        tuple: (epoch, avg_t_stat, p_value) or (None, None, None)
    """
    epochs = sorted(df['epochs_completed'].unique())

    seen_above_threshold = False

    for epoch in epochs:
        # Compute average t-statistic across all authors at this epoch
        author_t_stats = []

        for author in AUTHORS:
            author_df = df[(df['train_author'] == author) & (df['epochs_completed'] == epoch)]

            # Get self and other losses
            self_losses = author_df[author_df['loss_dataset'] == author]['loss_value'].values
            other_authors = [a for a in AUTHORS if a != author]
            other_losses = author_df[author_df['loss_dataset'].isin(other_authors)]['loss_value'].values

            if len(self_losses) > 0 and len(other_losses) > 0:
                # Simple t-statistic
                mean_diff = np.mean(other_losses) - np.mean(self_losses)
                pooled_std = np.sqrt((np.var(other_losses) + np.var(self_losses)) / 2)
                if pooled_std > 0:
                    t_stat = mean_diff / pooled_std
                    author_t_stats.append(t_stat)

        if len(author_t_stats) == len(AUTHORS):
            avg_t = np.mean(author_t_stats)
            # One-sample t-test: is average t-stat significantly > 0?
            t_result = stats.ttest_1samp(author_t_stats, 0)
            p_value = t_result.pvalue / 2  # One-tailed

            if p_value >= p_threshold:
                seen_above_threshold = True
            elif seen_above_threshold and p_value < p_threshold:
                return epoch, avg_t, p_value

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
            # Filter by author and seed columns (works for both baseline and variants)
            model_df = df[(df['train_author'] == author) & (df['seed'] == seed)]

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

    Returns:
        tuple: (pandas DataFrame, LaTeX string)
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
                'p-value': f'{t_result.pvalue:.2e}',
                't_stat_val': t_result.statistic,
                'df_val': t_result.df,
                'p_val': t_result.pvalue
            })

    df_table = pd.DataFrame(results)

    # Generate LaTeX table
    latex_lines = [
        "\\begin{table}[h]",
        "\\centering",
        "\\small",
        "\\begin{tabular}{lccc}",
        "\\hline",
        "\\textbf{Model} & \\textbf{$t$-stat} & \\textbf{df} & \\textbf{$p$-value}\\\\",
        "\\hline"
    ]

    for _, row in df_table.iterrows():
        # Format p-value in scientific notation
        p_val = row['p_val']
        if p_val < 0.01:
            exponent = int(np.floor(np.log10(p_val)))
            mantissa = p_val / (10 ** exponent)
            p_str = f"${mantissa:.2f} \\times 10^{{{exponent}}}$"
        else:
            p_str = f"${p_val:.4f}$"

        latex_lines.append(
            f"{row['Model']:<12} & {row['t_stat_val']:.2f} & {row['df_val']:.2f} & {p_str}  \\\\"
        )

    latex_lines.append("\\hline")
    latex_lines.append("\\end{tabular}")
    latex_lines.append("\\end{table}")

    latex_table = "\n".join(latex_lines)

    return df_table, latex_table


def compute_cross_variant_comparisons(all_variant_data, epoch=500):
    """
    Compare t-value distributions across variants at epoch 500.

    Args:
        all_variant_data: dict of {variant_name: DataFrame}
        epoch: Epoch to compare at (default: 500)

    Returns:
        DataFrame with pairwise t-test results
    """
    from itertools import combinations

    # Extract t-values for each variant at epoch 500
    variant_t_values = {}

    for variant_name, df in all_variant_data.items():
        t_values = []

        for author in AUTHORS:
            # Get final epoch data for this author
            author_df = df[(df['train_author'] == author) & (df['epochs_completed'] == epoch)]

            # Get self and other losses
            self_losses = author_df[author_df['loss_dataset'] == author]['loss_value'].values
            other_authors = [a for a in AUTHORS if a != author]
            other_losses = author_df[author_df['loss_dataset'].isin(other_authors)]['loss_value'].values

            if len(self_losses) > 0 and len(other_losses) > 0:
                # Compute t-statistic
                if len(self_losses) == 1:
                    mean_diff = np.mean(other_losses) - self_losses[0]
                    std_other = np.std(other_losses)
                    if std_other > 0:
                        t_stat = mean_diff / (std_other / np.sqrt(len(other_losses)))
                        t_values.append(t_stat)
                else:
                    t_stat, _ = stats.ttest_ind(other_losses, self_losses, equal_var=False)
                    if not np.isnan(t_stat):
                        t_values.append(t_stat)

        variant_t_values[variant_name] = t_values

    # Pairwise comparisons
    results = []
    variant_names = list(all_variant_data.keys())

    for var1, var2 in combinations(variant_names, 2):
        if var1 in variant_t_values and var2 in variant_t_values:
            t_vals_1 = variant_t_values[var1]
            t_vals_2 = variant_t_values[var2]

            if len(t_vals_1) >= 2 and len(t_vals_2) >= 2:
                # T-test comparing distributions
                t_result = stats.ttest_ind(t_vals_1, t_vals_2, equal_var=False)

                results.append({
                    'Comparison': f'{var1} vs {var2}',
                    't-stat': f'{t_result.statistic:.2f}',
                    'df': f'{t_result.df:.2f}',
                    'p-value': f'{t_result.pvalue:.2e}',
                    'mean_diff': f'{np.mean(t_vals_1) - np.mean(t_vals_2):.2f}'
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
    parser.add_argument(
        '--cross-variant-comparison',
        action='store_true',
        help='Compute pairwise comparisons across all variants'
    )

    args = parser.parse_args()

    # Handle cross-variant comparison mode
    if args.cross_variant_comparison:
        print("=" * 60)
        print("Cross-Variant Comparison Analysis")
        print("=" * 60)

        # Load all variant data
        all_variant_data = {}
        for var_name, var_key in [('baseline', None), ('content', 'content'), ('function', 'function'), ('pos', 'pos')]:
            pkl_file = f"data/model_results.pkl" if var_key is None else f"data/model_results_{var_key}.pkl"
            if Path(pkl_file).exists():
                all_variant_data[var_name] = load_data(pkl_file, var_key)
            else:
                print(f"Warning: {pkl_file} not found, skipping {var_name}")

        if len(all_variant_data) < 2:
            print("Error: Need at least 2 variants for comparison")
            return

        print(f"\nLoaded {len(all_variant_data)} conditions: {list(all_variant_data.keys())}")

        # Compute pairwise comparisons
        print("\nPairwise T-Test Comparisons (Epoch 500)")
        print("Comparing distributions of t-statistics across all authors")
        print("-" * 60)

        comparison_df = compute_cross_variant_comparisons(all_variant_data, epoch=500)

        if not comparison_df.empty:
            print("\n" + comparison_df.to_string(index=False))
        else:
            print("No comparisons could be computed")

        print("\n" + "=" * 60)
        return

    # Update header to show variant
    variant_label = f" (Variant: {args.variant})" if args.variant else " (Baseline)"
    print("=" * 60)
    print(f"LLM Stylometry Statistical Analysis{variant_label}")
    print("=" * 60)

    # Load data with variant filter
    print("\nLoading data...")
    df = load_data(data_path=args.data, variant=args.variant)

    # 1. Find threshold crossing epochs per author
    print("\n1. Individual Author Threshold Crossings (p < 0.001)")
    print("-" * 40)
    crossing_authors = find_threshold_crossing_epochs(df)
    if crossing_authors:
        for author in AUTHORS:
            if author in crossing_authors:
                epoch, t_stat, p_value = crossing_authors[author]
                print(f"{author.capitalize():<12}: Epoch {epoch:3d} (t={t_stat:.2f}, p={p_value:.2e})")
            else:
                print(f"{author.capitalize():<12}: No threshold crossing detected")
    else:
        print("No authors crossed threshold (started below or never crossed)")

    # 2. Average t-statistic threshold crossing
    print("\n2. Average T-Statistic Threshold Crossing (p < 0.001)")
    print("-" * 40)
    epoch, avg_t, p_value = find_average_threshold_crossing(df)
    if epoch is not None:
        print(f"Average t-stat crossed threshold at epoch: {epoch}")
        print(f"Average t-statistic: {avg_t:.3f}")
        print(f"p-value: {p_value:.2e}")
    else:
        print("Average t-statistic did not cross threshold")

    # 3. Average t-test at final epoch
    print("\n3. Average T-Test Across Authors (Epoch 500)")
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

    # 4. Author comparison table
    print("\n4. Author Model Comparison Table (Table 1)")
    print("-" * 40)
    table, latex_table = generate_author_comparison_table(df)

    # Display DataFrame table
    print("\n" + table[['Model', 't-stat', 'df', 'p-value']].to_string(index=False))

    # Display LaTeX table
    print("\n\nLaTeX Table Format:")
    print("-" * 40)
    print(latex_table)

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()