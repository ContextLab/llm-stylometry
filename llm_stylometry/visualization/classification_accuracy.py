"""Generate classification accuracy grouped bar chart with bootstrap CI."""

import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path
import pickle

from llm_stylometry.core.constants import AUTHORS


def generate_classification_accuracy_figure(
    output_path: str = "paper/figs/source/classification_accuracy.pdf",
    figsize: tuple = (14, 6),
    font: str = 'Helvetica'
):
    """
    Generate grouped bar chart showing classification accuracy across all conditions.

    Loads results from all 4 conditions (baseline, content, function, pos) and
    creates a single grouped bar plot with different alpha values per condition.

    Args:
        output_path: Path to save PDF (default: paper/figs/source/classification_accuracy.pdf)
        figsize: Figure size
        font: Font family to use

    Returns:
        matplotlib figure object

    Examples:
        >>> fig = generate_classification_accuracy_figure()
    """
    # Set font
    plt.rcParams['font.family'] = font
    plt.rcParams['font.sans-serif'] = [font]

    # Load results from all 4 conditions
    variants = [
        ('baseline', None, 1.0),
        ('content', 'content', 0.8),
        ('function', 'function', 0.6),
        ('pos', 'pos', 0.4)
    ]

    all_results = []

    for condition_name, variant, alpha_val in variants:
        pkl_path = f"data/classifier_results/{condition_name}.pkl"
        if not Path(pkl_path).exists():
            print(f"Warning: {pkl_path} not found, skipping {condition_name}")
            continue

        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)

        results_df = data['results'].copy()

        # Capitalize author names
        results_df['author'] = results_df['author'].str.capitalize()

        # Add condition column
        results_df['condition'] = condition_name.capitalize()
        results_df['alpha'] = alpha_val

        # Add to combined results
        all_results.append(results_df)

        # Also add "Overall" for this condition
        overall_df = results_df.copy()
        overall_df['author'] = 'Overall'
        all_results.append(overall_df)

    if not all_results:
        raise ValueError("No classification results found. Please run classification experiments first.")

    # Combine all results
    plot_df = pd.concat(all_results, ignore_index=True)

    # Define author order
    author_order = [a.capitalize() for a in AUTHORS] + ['Overall']

    # Define color palette (same as other figures)
    base_colors = sns.color_palette("tab10", n_colors=len(AUTHORS))
    author_palette = dict(zip([a.capitalize() for a in AUTHORS], base_colors))
    author_palette['Overall'] = 'black'

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Grouped bar plot with bootstrap 95% CI
    # Use hue='condition' for grouping
    sns.barplot(
        data=plot_df,
        x='author',
        y='accuracy',
        hue='condition',
        order=author_order,
        hue_order=['Baseline', 'Content', 'Function', 'Pos'],
        errorbar='ci',  # Bootstrap 95% confidence intervals
        ax=ax,
        err_kws={'linewidth': 1.0},
        palette='Set2',  # Use neutral palette for conditions
        legend=False  # No legend (user will create manually)
    )

    # Apply custom alpha values per condition
    for i, bar_container in enumerate(ax.containers):
        condition_name = ['Baseline', 'Content', 'Function', 'Pos'][i]
        alpha_map = {'Baseline': 1.0, 'Content': 0.8, 'Function': 0.6, 'Pos': 0.4}
        alpha = alpha_map[condition_name]

        for bar in bar_container:
            bar.set_alpha(alpha)

    # Styling
    ax.set_xlabel('')  # Remove x-axis label
    ax.set_ylabel('Classification Accuracy', fontsize=12)
    ax.set_ylim(0, 1.0)
    sns.despine(ax=ax, top=True, right=True)

    # Increase tick font sizes
    ax.tick_params(axis='x', rotation=45, labelsize=14)
    ax.tick_params(axis='y', labelsize=11)

    plt.tight_layout()

    # Ensure output directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(output_path, format='pdf', bbox_inches='tight')

    return fig
