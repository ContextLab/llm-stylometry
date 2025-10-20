"""Generate classification accuracy bar chart with bootstrap CI."""

import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path

from llm_stylometry.core.constants import AUTHORS


def generate_classification_accuracy_figure(
    data_path: str = "data/classifier_results/baseline.pkl",
    output_path: str = None,
    figsize: tuple = (10, 6),
    font: str = 'Helvetica',
    variant: str = None
):
    """
    Generate Figure: Classification accuracy bar chart with bootstrap 95% CI.

    Args:
        data_path: Path to classifier results pkl file
        output_path: Path to save PDF (optional)
        figsize: Figure size
        font: Font family to use
        variant: Analysis variant ('content', 'function', 'pos') or None for baseline

    Returns:
        matplotlib figure object

    Examples:
        >>> fig = generate_classification_accuracy_figure(
        ...     data_path='data/classifier_results/baseline.pkl',
        ...     output_path='paper/figs/source/classification_accuracy_baseline.pdf'
        ... )
    """
    # Set font
    plt.rcParams['font.family'] = font
    plt.rcParams['font.sans-serif'] = [font]

    # Load results
    import pickle
    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    results_df = data['results'].copy()

    # Prepare data for seaborn
    # Results DF is in long format: one row per held-out book
    # Columns: split_id, author, accuracy, held_out_book_id, predicted_author, true_author, classifier

    # Capitalize author names for display
    results_df['author'] = results_df['author'].str.capitalize()

    # Add "Overall" category (all data points)
    overall_df = results_df.copy()
    overall_df['author'] = 'Overall'

    plot_df = pd.concat([results_df, overall_df], ignore_index=True)

    # Define author order (same as other figures)
    author_order = [a.capitalize() for a in AUTHORS] + ['Overall']

    # Define color palette (same as all_losses.py)
    # Tab10 palette with Baum and Thompson first
    base_colors = sns.color_palette("tab10", n_colors=len(AUTHORS))
    palette = dict(zip([a.capitalize() for a in AUTHORS], base_colors))
    palette['Overall'] = 'black'

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Bar plot with bootstrap 95% CI (seaborn's default: n_boot=1000)
    sns.barplot(
        data=plot_df,
        x='author',
        y='accuracy',
        order=author_order,
        palette=palette,
        errorbar='ci',  # Bootstrap 95% confidence intervals
        ax=ax,
        err_kws={'linewidth': 1.5}  # Make error bars visible
    )

    # Styling
    ax.set_xlabel('Author', fontsize=12)
    ax.set_ylabel('Classification Accuracy', fontsize=12)
    ax.set_ylim(0, 1.0)
    sns.despine(ax=ax, top=True, right=True)

    # Rotate x-axis labels if needed
    ax.tick_params(axis='x', rotation=45)

    plt.tight_layout()

    # Save if output path provided
    if output_path is None:
        if variant is None:
            output_path = "paper/figs/source/classification_accuracy_baseline.pdf"
        else:
            output_path = f"paper/figs/source/classification_accuracy_{variant}.pdf"

    # Ensure output directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(output_path, format='pdf', bbox_inches='tight')

    return fig
