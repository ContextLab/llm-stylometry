"""Generate Oz losses figure from the paper."""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path


def generate_oz_losses_figure(
    data_path="data/model_results.pkl",
    output_path=None,
    figsize=(12, 8),
    show_legend=False,
    font='Helvetica'
):
    """
    Generate Figure 5: Oz losses analysis.

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

    # Filter for Baum and Thompson models and relevant datasets
    oz_datasets = ["baum", "thompson", "contested", "non_oz_baum", "non_oz_thompson", "train"]

    oz_df = df[
        (df["train_author"].isin(["baum", "thompson"])) &
        (df["loss_dataset"].isin(oz_datasets))
    ].copy()

    # Sample every 10 epochs for cleaner visualization
    oz_df = oz_df[oz_df["epochs_completed"] % 10 == 1]

    # Capitalize names for display
    oz_df["loss_dataset"] = oz_df["loss_dataset"].str.replace("_", " ").str.title()
    oz_df["loss_dataset"] = oz_df["loss_dataset"].str.replace("Non Oz", "Non-Oz")
    oz_df["train_author"] = oz_df["train_author"].str.capitalize()

    # Keep only rows up to min-epoch for each train_author
    min_epochs = (
        oz_df.groupby(["train_author", "seed"])["epochs_completed"]
        .max()
        .groupby("train_author")
        .min()
    )
    oz_df = oz_df[oz_df["epochs_completed"] <= oz_df["train_author"].map(min_epochs)]

    # Define colors
    palette = {"Baum": "#1f77b4", "Thompson": "#ff7f0e"}

    # Define subplot order
    subplot_order = ["Train", "Baum", "Thompson", "Contested", "Non-Oz Baum", "Non-Oz Thompson"]

    # Create figure with 2x3 subplots
    fig, axes = plt.subplots(2, 3, figsize=figsize, sharex=True, sharey=True)
    axes = axes.flatten()

    # Plot each dataset
    for idx, dataset in enumerate(subplot_order):
        ax = axes[idx]
        subset = oz_df[oz_df["loss_dataset"] == dataset]

        if not subset.empty:
            sns.lineplot(
                data=subset,
                x="epochs_completed",
                y="loss_value",
                hue="train_author",
                palette=palette,
                ax=ax,
                legend=False,
            )

        sns.despine(ax=ax, top=True, right=True)

        # Capitalize names and italicize 'Oz' in titles
        if dataset == "Train":
            title = "Training"
        elif dataset == "Contested":
            title = "Contested"
        elif dataset == "Non-Oz Baum":
            title = r"Non-$\mathit{Oz}$ Baum"
        elif dataset == "Non-Oz Thompson":
            title = r"Non-$\mathit{Oz}$ Thompson"
        else:
            title = dataset  # Already capitalized (Baum, Thompson)

        ax.set_title(title, fontsize=14)

        ax.set_xlabel("Epochs completed", fontsize=12)
        ax.set_ylabel("Loss", fontsize=12)
        ax.set_xlim(left=1, right=oz_df["epochs_completed"].max())

    if show_legend:
        # Add legend
        handles = [Line2D([0], [0], color=palette["Baum"], lw=2, label="Baum"),
                  Line2D([0], [0], color=palette["Thompson"], lw=2, label="Thompson")]
        fig.legend(handles=handles,
                  loc='upper center',
                  ncol=2,
                  fontsize=12,
                  bbox_to_anchor=(0.5, 1.02))

    # Remove title as requested
    # fig.suptitle("Cross-entropy loss across models and Oz authors",
    #             fontsize=16,
    #             y=1.05 if show_legend else 1.02)

    plt.tight_layout()

    # Save if path provided
    if output_path:
        fig.savefig(output_path, format="pdf", bbox_inches="tight")

    return fig