"""Generate stripplot figure from the paper."""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def generate_stripplot_figure(
    data_path="data/model_results.pkl",
    output_path=None,
    figsize=(8, 6),
    show_legend=False,
    font='Helvetica'
):
    """
    Generate Figure 1B: Strip plot showing loss distributions.

    Args:
        data_path: Path to model_results.pkl
        output_path: Path to save PDF (optional)
        figsize: Figure size
        show_legend: Whether to show legend (False for paper)
        font: Font family to use

    Returns:
        matplotlib figure object
    """
    # Set font
    plt.rcParams['font.family'] = font
    plt.rcParams['font.sans-serif'] = [font]

    # Load data
    df = pd.read_pickle(data_path)

    # Prepare data exactly as in stripplot.py
    strip_df = df.copy()
    strip_df["loss_dataset"] = strip_df["loss_dataset"].str.capitalize()
    strip_df["train_author"] = strip_df["train_author"].str.capitalize()

    # Get final epoch for each model/dataset combination
    strip_df = (
        strip_df[strip_df["loss_dataset"] != "Train"]
        .groupby(["train_author", "loss_dataset", "seed"])
        .tail(1)
        .rename(
            columns={
                "train_author": "Training Author",
                "loss_dataset": "Evaluated Author",
                "loss_value": "Loss",
            }
        )
    )

    # Add evaluation type column
    strip_df["EvalType"] = np.where(
        strip_df["Training Author"] == strip_df["Evaluated Author"], "Self", "Other"
    )
    eval_palette = {"Self": "C0", "Other": "C1"}

    # Create figure
    plt.figure(figsize=figsize)
    ax = sns.stripplot(
        data=strip_df,
        x="Training Author",
        y="Loss",
        hue="EvalType",
        palette=eval_palette,
        size=6,
        edgecolor=None,
        dodge=True,
        legend=show_legend,
    )

    # Set labels in sentence case
    plt.title(
        "Loss values: training author vs. other authors",
        fontsize=16,
        pad=10,
    )
    plt.xlabel("Training author", fontsize=14)
    plt.ylabel("Loss", fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    if show_legend:
        plt.legend(fontsize=12, title="Author type", title_fontsize=14)

    plt.tight_layout()

    # Save if path provided
    if output_path:
        plt.savefig(output_path, bbox_inches="tight", format="pdf")

    return plt.gcf()