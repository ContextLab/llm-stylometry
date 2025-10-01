"""Generate t-test figures from the paper."""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
import numpy as np
from pathlib import Path
from tqdm import tqdm


def calculate_t_statistics(df, max_epochs=500):
    """Calculate t-statistics comparing same vs other author losses."""

    # Define authors
    AUTHORS = ["baum", "thompson", "dickens", "melville", "wells", "austen", "fitzgerald", "twain"]

    # Filter and prepare data
    t_df = df[df["loss_dataset"].isin(AUTHORS)].copy()
    t_df = t_df[t_df["epochs_completed"] <= max_epochs]
    t_df["loss_dataset"] = t_df["loss_dataset"].str.capitalize()
    t_df["train_author"] = t_df["train_author"].str.capitalize()

    # Prepare authors and epochs
    authors = sorted(t_df["train_author"].unique())
    epochs = sorted(t_df["epochs_completed"].unique())
    t_raws = {author: [] for author in authors}

    # Compute Welch's t-statistic for each author/epoch
    for author in tqdm(authors, desc="Processing authors"):
        for epoch in epochs:
            true_losses = t_df[
                (t_df["train_author"] == author)
                & (t_df["loss_dataset"] == author)
                & (t_df["epochs_completed"] == epoch)
            ]["loss_value"].values

            other_losses = t_df[
                (t_df["train_author"] == author)
                & (t_df["loss_dataset"] != author)
                & (t_df["epochs_completed"] == epoch)
            ]["loss_value"].values

            if len(true_losses) > 0 and len(other_losses) > 0:
                result = ttest_ind(other_losses, true_losses, equal_var=False)
                t_raws[author].append(result.statistic)
            else:
                t_raws[author].append(np.nan)

    # Convert to long-form DataFrame
    t_raws_df = (
        pd.DataFrame(t_raws, index=epochs)
        .reset_index()
        .melt(id_vars="index", var_name="Author", value_name="t_raw")
        .rename(columns={"index": "Epoch"})
    )

    return t_raws_df, t_raws


def generate_t_test_figure(
    data_path="data/model_results.pkl",
    output_path=None,
    figsize=(6, 4),
    show_legend=False,
    font='Helvetica',
    variant=None
):
    """
    Generate Figure 2A: t-statistics for individual authors.

    Args:
        data_path: Path to model_results.pkl
        output_path: Path to save PDF (optional)
        figsize: Figure size
        show_legend: Whether to show legend (False for paper)
        font: Font family to use

        variant: Analysis variant ('content', 'function', 'pos') or None for baseline

    Returns:
        matplotlib figure object
    """
    # Set font
    plt.rcParams['font.family'] = font
    plt.rcParams['font.sans-serif'] = [font]

    # Load data and calculate t-statistics
    df = pd.read_pickle(data_path)

    # Filter by variant
    if variant is None:
        # Baseline: exclude variant models
        if 'variant' in df.columns:
            df = df[df['variant'].isna()].copy()
    else:
        # Specific variant
        if 'variant' not in df.columns:
            raise ValueError(f"No variant column in data")
        df = df[df['variant'] == variant].copy()

    t_raws_df, _ = calculate_t_statistics(df)

    # Define color palette
    unique_authors = sorted(t_raws_df["Author"].unique())
    fixed_first = ["Baum", "Thompson"]
    hue_order = fixed_first + [a for a in unique_authors if a not in fixed_first]
    palette = dict(zip(hue_order, sns.color_palette("tab10", n_colors=len(hue_order))))

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    sns.lineplot(
        data=t_raws_df,
        x="Epoch",
        y="t_raw",
        hue="Author",
        ax=ax,
        hue_order=hue_order,
        palette=palette,
        legend=show_legend,
    )

    sns.despine(ax=ax, top=True, right=True)
    # Remove title as requested
    # ax.set_title(
    #     "$t$-values: training author vs. other authors",
    #     fontsize=12,
    #     pad=10,
    # )
    ax.set_xlabel("Epochs completed", fontsize=12)
    ax.set_ylabel("$t$-value", fontsize=12)

    # Add threshold line
    threshold = 3.291
    ax.axhline(y=threshold, linestyle="--", color="black", label="p<0.001 threshold" if show_legend else "")
    ax.set_xlim(0, t_raws_df["Epoch"].max())
    ax.set_ylim(bottom=0)

    if show_legend:
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(
            handles=handles,
            labels=labels,
            title="Training author",
            fontsize=8,
            title_fontsize=9,
            loc="upper left",
        )

    plt.tight_layout()

    if output_path:
        # Add variant suffix to filename if variant specified
        if variant:
            from pathlib import Path
            output_path = Path(output_path)
            output_path = str(output_path.parent / f"{output_path.stem}_{variant}{output_path.suffix}")
        fig.savefig(output_path, format="pdf", bbox_inches="tight")

    return fig


def generate_t_test_avg_figure(
    data_path="data/model_results.pkl",
    output_path=None,
    figsize=(6, 4),
    show_legend=False,
    font='Helvetica',
    variant=None
):
    """
    Generate Figure 2B: Average t-statistic across all authors.

    Args:
        data_path: Path to model_results.pkl
        output_path: Path to save PDF (optional)
        figsize: Figure size
        show_legend: Whether to show legend (False for paper)
        font: Font family to use

        variant: Analysis variant ('content', 'function', 'pos') or None for baseline

    Returns:
        matplotlib figure object
    """
    # Set font
    plt.rcParams['font.family'] = font
    plt.rcParams['font.sans-serif'] = [font]

    # Load data and calculate t-statistics
    df = pd.read_pickle(data_path)

    # Filter by variant
    if variant is None:
        # Baseline: exclude variant models
        if 'variant' in df.columns:
            df = df[df['variant'].isna()].copy()
    else:
        # Specific variant
        if 'variant' not in df.columns:
            raise ValueError(f"No variant column in data")
        df = df[df['variant'] == variant].copy()

    t_raws_df, _ = calculate_t_statistics(df)

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    sns.lineplot(
        data=t_raws_df,
        x="Epoch",
        y="t_raw",
        ax=ax,
        legend=False,
        color="black",  # Set line color to black
    )

    sns.despine(ax=ax, top=True, right=True)
    # Remove title as requested
    # ax.set_title(
    #     "Average $t$-values: training author vs. other authors",
    #     fontsize=12,
    #     pad=10,
    # )
    ax.set_xlabel("Epochs completed", fontsize=12)
    ax.set_ylabel("$t$-value", fontsize=12)

    # Add threshold line
    threshold = 3.291
    ax.axhline(y=threshold, linestyle="--", color="black", label="p<0.001 threshold" if show_legend else "")
    ax.set_xlim(0, t_raws_df["Epoch"].max())
    ax.set_ylim(bottom=0)

    if show_legend:
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(
            handles=handles,
            labels=labels,
            fontsize=8,
            title_fontsize=9,
            loc="upper left",
        )

    plt.tight_layout()

    if output_path:
        # Add variant suffix to filename if variant specified
        if variant:
            from pathlib import Path
            output_path = Path(output_path)
            output_path = str(output_path.parent / f"{output_path.stem}_{variant}{output_path.suffix}")
        fig.savefig(output_path, format="pdf", bbox_inches="tight")

    return fig