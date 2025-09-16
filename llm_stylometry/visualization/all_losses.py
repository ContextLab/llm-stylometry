"""Generate all losses figure from the paper."""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path
import numpy as np


def generate_all_losses_figure(
    data_path="data/model_results.pkl",
    output_path=None,
    figsize=(8, 6),
    show_legend=False,
    font='Helvetica'
):
    """
    Generate Figure 1A: Training curves showing cross-entropy loss over epochs.

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

    # Define authors in requested order
    AUTHORS = ["baum", "thompson", "austen", "dickens", "fitzgerald", "melville", "twain", "wells"]

    # Prepare data exactly as in original all_losses.py
    plot_df = df[df["loss_dataset"].isin(AUTHORS + ["train"])].copy()
    plot_df = plot_df[plot_df["epochs_completed"] % 10 == 1]
    # Keep proper capitalization for authors
    plot_df["loss_dataset"] = plot_df["loss_dataset"].apply(lambda x: x.capitalize() if x != "train" else "Train")
    plot_df["train_author"] = plot_df["train_author"].str.capitalize()

    # Keep only rows up to min-epoch for each train_author
    min_epochs = (
        plot_df.groupby(["train_author", "seed"])["epochs_completed"]
        .max()
        .groupby("train_author")
        .min()
    )
    plot_df = plot_df[plot_df["epochs_completed"] <= plot_df["train_author"].map(min_epochs)]

    # Define fixed hue order and color palette for consistent mapping
    unique_authors = sorted(plot_df["train_author"].unique())
    fixed_first = ["Baum", "Thompson"]
    hue_order = fixed_first + [a for a in unique_authors if a not in fixed_first]
    palette = dict(zip(hue_order, sns.color_palette("tab10", n_colors=len(hue_order))))

    # Define explicit evaluation order for subplots
    eval_order = ["Train"] + [a.capitalize() for a in AUTHORS]
    loss_datasets = eval_order

    # Create figure
    fig, axes = plt.subplots(3, 3, figsize=figsize, sharex=True, sharey=True)

    # Plot all subplots without legends
    for ax, loss_dataset in zip(axes.flatten(), loss_datasets):
        subset = plot_df[plot_df["loss_dataset"] == loss_dataset]
        sns.lineplot(
            data=subset,
            x="epochs_completed",
            y="loss_value",
            hue="train_author",
            hue_order=hue_order,
            palette=palette,
            ax=ax,
            legend=False,
        )
        sns.despine(ax=ax, top=True, right=True)
        # Capitalize author names in titles
        ax.set_title(f"{loss_dataset}", fontsize=14)
        ax.set_xlabel("Epochs completed", fontsize=12)
        ax.set_ylabel("Loss", fontsize=12)
        ax.set_xlim(left=1, right=plot_df["epochs_completed"].max())

    if show_legend:
        # Use an invisible plot to extract legend handles
        dummy_ax = plt.figure().add_subplot()
        legend_plot = sns.lineplot(
            data=plot_df,
            x="epochs_completed",
            y="loss_value",
            hue="train_author",
            hue_order=hue_order,
            palette=palette,
            ax=dummy_ax,
            legend="auto",
        )
        handles, labels = dummy_ax.get_legend_handles_labels()
        plt.close()

        # Reorder legend entries
        ordered_handles = []
        ordered_labels = []
        for ds in eval_order:
            if ds in labels:
                idx = labels.index(ds)
                ordered_handles.append(handles[idx])
                ordered_labels.append(labels[idx])
            else:
                ordered_handles.append(Line2D([], [], color="white", alpha=0))
                ordered_labels.append("")

        # Transpose legend entries to fill columns first
        ncols = 3
        trans_handles = []
        trans_labels = []
        for i in range(ncols):
            for j in range(len(ordered_handles) // ncols):
                idx = j * ncols + i
                trans_handles.append(ordered_handles[idx])
                trans_labels.append(ordered_labels[idx])

        fig.legend(
            trans_handles,
            trans_labels,
            title="Training author",
            loc="upper center",
            ncol=3,
            fontsize=12,
            title_fontsize=13,
            bbox_to_anchor=(0.5, 1.15),
        )

    # Remove title as requested
    # fig.suptitle(
    #     "Losses on each comparison text",
    #     fontsize=16,
    #     y=1.20 if show_legend else 1.02,
    # )

    plt.tight_layout()

    # Save if path provided
    if output_path:
        fig.savefig(output_path, format="pdf", bbox_inches="tight")

    return fig