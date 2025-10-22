"""Generate t-test figures from the paper."""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind, t as t_dist
import numpy as np
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)


def calculate_t_statistics(df, max_epochs=500):
    """
    Calculate t-statistics and df comparing same vs other author losses.

    Returns:
        tuple: (t_raws_df, t_raws, df_values, thresholds)
            - t_raws_df: Long-form DataFrame with columns [Epoch, Author, t_raw]
            - t_raws: Dict mapping author to list of t-values
            - df_values: Dict mapping author to list of degrees of freedom
            - thresholds: Dict mapping author to list of t-thresholds for p=0.001
    """

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
    df_values = {author: [] for author in authors}
    thresholds = {author: [] for author in authors}

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

            # T-test requires at least 2 samples per group for meaningful results
            if len(true_losses) >= 2 and len(other_losses) >= 2:
                result = ttest_ind(other_losses, true_losses, equal_var=False)
                if np.isnan(result.statistic):
                    logger.debug(f"NaN t-statistic for {author} at epoch {epoch}: "
                                f"n_true={len(true_losses)}, n_other={len(other_losses)}")
                t_raws[author].append(result.statistic)
                df_values[author].append(result.df)

                # Compute t-threshold for p=0.001 (one-tailed) given this df
                t_threshold = t_dist.ppf(1 - 0.001, result.df)
                thresholds[author].append(t_threshold)
            elif len(true_losses) > 0 or len(other_losses) > 0:
                # Have some data but insufficient for t-test
                logger.debug(f"Insufficient data for t-test for {author} at epoch {epoch}: "
                            f"n_true={len(true_losses)}, n_other={len(other_losses)} "
                            f"(need at least 2 samples per group)")
                t_raws[author].append(np.nan)
                df_values[author].append(np.nan)
                thresholds[author].append(np.nan)
            else:
                # No data at all
                logger.debug(f"No data for {author} at epoch {epoch}")
                t_raws[author].append(np.nan)
                df_values[author].append(np.nan)
                thresholds[author].append(np.nan)

    # Convert to long-form DataFrame
    t_raws_df = (
        pd.DataFrame(t_raws, index=epochs)
        .reset_index()
        .melt(id_vars="index", var_name="Author", value_name="t_raw")
        .rename(columns={"index": "Epoch"})
    )

    return t_raws_df, t_raws, df_values, thresholds


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
            raise ValueError("No variant column in data")
        df = df[df['variant'] == variant].copy()

    t_raws_df, _, df_values, thresholds = calculate_t_statistics(df)

    # Compute average threshold across authors at each epoch (for plotting)
    epochs = sorted(t_raws_df["Epoch"].unique())
    threshold_data = []
    for epoch in epochs:
        epoch_thresholds = []
        for author in thresholds.keys():
            epoch_idx = list(epochs).index(epoch)
            if epoch_idx < len(thresholds[author]):
                thresh = thresholds[author][epoch_idx]
                if not np.isnan(thresh):
                    epoch_thresholds.append(thresh)

        # For each epoch, add one row per author's threshold (for bootstrap CI calculation)
        for thresh in epoch_thresholds:
            threshold_data.append({'Epoch': epoch, 'threshold': thresh})

    threshold_df = pd.DataFrame(threshold_data)

    # Define color palette
    unique_authors = sorted(t_raws_df["Author"].unique())
    fixed_first = ["Baum", "Thompson"]
    hue_order = fixed_first + [a for a in unique_authors if a not in fixed_first]
    palette = dict(zip(hue_order, sns.color_palette("tab10", n_colors=len(hue_order))))

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Plot author t-statistics
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

    # Plot adaptive threshold with bootstrap 95% CI (solid black line)
    if not threshold_df.empty:
        sns.lineplot(
            data=threshold_df,
            x="Epoch",
            y="threshold",
            ax=ax,
            color="black",
            linewidth=2,
            linestyle="-",  # Solid line
            errorbar='ci',  # Bootstrap 95% CI
            label="p<0.001 threshold" if show_legend else ""
        )

    sns.despine(ax=ax, top=True, right=True)
    ax.set_xlabel("Epochs completed", fontsize=12)
    ax.set_ylabel("$t$-value", fontsize=12)

    # Calculate dynamic y-axis limits based on VALID data only
    valid_t_values = t_raws_df['t_raw'].replace([np.inf, -np.inf], np.nan).dropna()

    if len(valid_t_values) == 0:
        logger.warning("No valid t-statistics found. Using default axis limits.")
        y_min = -1.0
        y_max = 5.0
    else:
        y_min = valid_t_values.min()
        y_max = valid_t_values.max()

        # Add padding
        y_range = y_max - y_min
        padding = 0.05 * y_range if y_range > 0 else 0.5
        y_min = min(y_min, 0) - padding
        y_max = y_max + padding

    # Final validation
    if not (np.isfinite(y_min) and np.isfinite(y_max) and y_min < y_max):
        logger.error(f"Invalid axis limits computed: y_min={y_min}, y_max={y_max}. Using defaults.")
        y_min = -1.0
        y_max = 5.0
    ax.set_xlim(0, t_raws_df["Epoch"].max())
    ax.set_ylim(y_min, y_max)

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
            raise ValueError("No variant column in data")
        df = df[df['variant'] == variant].copy()

    t_raws_df, _, df_values, thresholds = calculate_t_statistics(df)

    # Compute average threshold across authors at each epoch
    epochs = sorted(t_raws_df["Epoch"].unique())
    threshold_data = []
    for epoch in epochs:
        epoch_thresholds = []
        for author in thresholds.keys():
            epoch_idx = list(epochs).index(epoch)
            if epoch_idx < len(thresholds[author]):
                thresh = thresholds[author][epoch_idx]
                if not np.isnan(thresh):
                    epoch_thresholds.append(thresh)

        for thresh in epoch_thresholds:
            threshold_data.append({'Epoch': epoch, 'threshold': thresh})

    threshold_df = pd.DataFrame(threshold_data)

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Plot average t-statistic (gray for consistency with individual figure)
    sns.lineplot(
        data=t_raws_df,
        x="Epoch",
        y="t_raw",
        ax=ax,
        legend=False,
        color="gray",  # Set line color to gray
    )

    # Plot adaptive threshold with bootstrap 95% CI (solid black line, consistent with 2a)
    if not threshold_df.empty:
        sns.lineplot(
            data=threshold_df,
            x="Epoch",
            y="threshold",
            ax=ax,
            color="black",
            linewidth=2,
            linestyle="-",  # Solid line
            errorbar='ci',  # Bootstrap 95% CI
            label="p<0.001 threshold" if show_legend else ""
        )

    sns.despine(ax=ax, top=True, right=True)
    ax.set_xlabel("Epochs completed", fontsize=12)
    ax.set_ylabel("$t$-value", fontsize=12)

    # Calculate dynamic y-axis limits
    valid_t_values = t_raws_df['t_raw'].replace([np.inf, -np.inf], np.nan).dropna()

    if len(valid_t_values) == 0:
        logger.warning("No valid t-statistics found for average figure. Using default axis limits.")
        y_min = -1.0
        y_max = 5.0
    else:
        y_min = valid_t_values.min()
        y_max = valid_t_values.max()

        # Add padding
        y_range = y_max - y_min
        padding = 0.05 * y_range if y_range > 0 else 0.5
        y_min = min(y_min, 0) - padding
        y_max = y_max + padding

    # Final validation
    if not (np.isfinite(y_min) and np.isfinite(y_max) and y_min < y_max):
        logger.error(f"Invalid axis limits computed for average figure: y_min={y_min}, y_max={y_max}. Using defaults.")
        y_min = -1.0
        y_max = 5.0

    ax.set_xlim(0, t_raws_df["Epoch"].max())
    ax.set_ylim(y_min, y_max)

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
