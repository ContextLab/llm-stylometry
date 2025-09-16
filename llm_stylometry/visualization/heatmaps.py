"""Generate heatmap figures from the paper."""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from tqdm import tqdm


def generate_loss_heatmap_figure(
    data_path="data/model_results.pkl",
    output_path=None,
    figsize=(8, 6),
    font='Helvetica'
):
    """
    Generate Figure 3: Loss heatmap (confusion matrix).

    Args:
        data_path: Path to model_results.pkl
        output_path: Path to save PDF (optional)
        figsize: Figure size
        font: Font family to use

    Returns:
        matplotlib figure object
    """
    # Set font
    plt.rcParams['font.family'] = font
    plt.rcParams['font.sans-serif'] = [font]

    # Load data
    df = pd.read_pickle(data_path)

    # Define authors
    AUTHORS = ["baum", "thompson", "dickens", "melville", "wells", "austen", "fitzgerald", "twain"]

    # Collect final epoch losses for each model
    all_losses = []

    for model_name in tqdm(df['model_name'].unique(), desc="Processing models"):
        model_df = df[df['model_name'] == model_name]

        # Get the last loss value for each evaluation dataset
        final_losses = model_df.groupby(['loss_dataset']).tail(1)
        final_losses = final_losses[final_losses['loss_dataset'].str.lower().isin(AUTHORS)]

        all_losses.append(final_losses[['train_author', 'loss_dataset', 'loss_value']])

    # Combine all data
    loss_df = pd.concat(all_losses, ignore_index=True)

    # Capitalize author names
    loss_df['training_author'] = loss_df['train_author'].str.capitalize()
    loss_df['evaluation_author'] = loss_df['loss_dataset'].str.capitalize()

    # Calculate average loss for each combination
    avg_loss = (
        loss_df.groupby(['training_author', 'evaluation_author'])['loss_value']
        .mean()
        .reset_index()
    )

    # Pivot to create the heatmap matrix
    heatmap_data = avg_loss.pivot(
        index='training_author',
        columns='evaluation_author',
        values='loss_value'
    )

    # Define the order from the paper
    new_order = ["Austen", "Baum", "Thompson", "Twain", "Melville", "Dickens", "Fitzgerald", "Wells"]

    # Reorder rows and columns
    heatmap_data = heatmap_data.reindex(index=new_order, columns=new_order)

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    sns.heatmap(
        heatmap_data,
        annot=True,
        fmt=".2f",
        ax=ax,
        cbar=False,  # No colorbar for paper
        cmap="Blues",
    )

    # Use sentence case for labels
    ax.set_xlabel("Comparison author", fontsize=15)
    ax.set_ylabel("Training author", fontsize=15)
    ax.set_title("Heatmap of average loss values", fontsize=16)

    plt.xticks(fontsize=14, rotation=45)
    plt.yticks(fontsize=14)
    plt.tight_layout()

    # Save if path provided
    if output_path:
        fig.savefig(output_path, format="pdf", bbox_inches="tight")

    return fig