"""3D MDS visualization for author stylometric distances."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.manifold import MDS
from pathlib import Path
from tqdm import tqdm


# Define author colors to match existing figures
AUTHOR_COLORS = {
    "Baum": "#1f77b4",      # Blue
    "Thompson": "#ff7f0e",   # Orange
    "Austen": "#2ca02c",     # Green
    "Dickens": "#d62728",    # Red
    "Fitzgerald": "#9467bd", # Purple
    "Melville": "#8c564b",   # Brown
    "Twain": "#e377c2",      # Pink
    "Wells": "#7f7f7f",      # Gray
}

# Standardized author order
AUTHOR_ORDER = ["baum", "thompson", "austen", "dickens", "fitzgerald", "melville", "twain", "wells"]


def create_loss_matrix(df):
    """Create a loss matrix from model results DataFrame."""

    # Collect final epoch losses for each model
    all_losses = []

    for model_name in tqdm(df['model_name'].unique(), desc="Processing models"):
        model_df = df[df['model_name'] == model_name]

        # Get the last loss value for each evaluation dataset
        final_losses = model_df.groupby(['loss_dataset']).tail(1)
        final_losses = final_losses[final_losses['loss_dataset'].str.lower().isin(AUTHOR_ORDER)]

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

    return heatmap_data.values, new_order


def generate_3d_mds_figure(
    data_path="data/model_results.pkl",
    output_path=None,
    figsize=(9, 7),
    font='Helvetica',
    zoom_factor=0.1,
    variant=None
):
    """
    Generate Figure 4: 3D MDS plot from loss matrix.

    Args:
        data_path: Path to model_results.pkl
        output_path: Path to save PDF (optional)
        figsize: Figure size (adjusted for single panel)
        font: Font family to use
        zoom_factor: Zoom factor for axis limits

        variant: Analysis variant ('content', 'function', 'pos') or None for baseline

    Returns:
        matplotlib figure object
    """
    # Set font
    plt.rcParams['font.family'] = font
    plt.rcParams['font.sans-serif'] = [font]

    # Load data and create loss matrix
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

    loss_matrix, author_names = create_loss_matrix(df)

    # Symmetrize the matrix for MDS
    symmetric_matrix = (loss_matrix + loss_matrix.T) / 2

    # Apply MDS with 3 components
    mds = MDS(n_components=3, dissimilarity='precomputed', random_state=1)
    coords = mds.fit_transform(symmetric_matrix)
    x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]

    # Create figure
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')

    # Plot points with author-specific colors and black outline
    for i, author in enumerate(author_names):
        ax.scatter(x[i], y[i], z[i],
                  s=300,  # Larger dots
                  color=AUTHOR_COLORS[author],
                  marker='o',
                  edgecolors='black',  # Black outline
                  linewidth=1.5,
                  depthshade=True,
                  alpha=0.9)

        # Add text labels above dots with bold font, with much more spacing
        ax.text(x[i], y[i], z[i] + 0.5,  # Move text even higher above dots
               author,
               fontsize=12,
               fontweight='bold',
               ha='center',
               va='bottom')

    # No title as requested
    # ax.set_title("MDS from loss matrix", fontsize=14)

    # Add background shading and grid lines
    ax.xaxis.pane.fill = True
    ax.yaxis.pane.fill = True
    ax.zaxis.pane.fill = True
    ax.xaxis.pane.set_edgecolor('gray')
    ax.yaxis.pane.set_edgecolor('gray')
    ax.zaxis.pane.set_edgecolor('gray')
    ax.xaxis.pane.set_alpha(0.75)  # Even darker background (75% opacity)
    ax.yaxis.pane.set_alpha(0.75)
    ax.zaxis.pane.set_alpha(0.75)

    # Make grid lines subtle but visible (15% opacity)
    # For 3D plots, we need to set grid properties differently
    ax.xaxis._axinfo['grid']['color'] = (0.5, 0.5, 0.5, 0.15)  # RGBA with 15% alpha
    ax.yaxis._axinfo['grid']['color'] = (0.5, 0.5, 0.5, 0.15)
    ax.zaxis._axinfo['grid']['color'] = (0.5, 0.5, 0.5, 0.15)
    ax.xaxis._axinfo['grid']['linewidth'] = 0.5
    ax.yaxis._axinfo['grid']['linewidth'] = 0.5
    ax.zaxis._axinfo['grid']['linewidth'] = 0.5
    ax.grid(True)

    # Remove tick labels but keep ticks for grid lines
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])

    # Set axis limits with zoom factor
    ax.set_xlim(x.min() - zoom_factor, x.max() + zoom_factor)
    ax.set_ylim(y.min() - zoom_factor, y.max() + zoom_factor)
    ax.set_zlim(z.min() - zoom_factor, z.max() + zoom_factor)

    # Set viewing angle
    ax.view_init(elev=20, azim=45)

    plt.tight_layout()

    # Save if path provided
    if output_path:
        # Add variant suffix to filename if variant specified
        if variant:
            from pathlib import Path
            output_path = Path(output_path)
            output_path = str(output_path.parent / f"{output_path.stem}_{variant}{output_path.suffix}")
        fig.savefig(output_path, format="pdf", bbox_inches="tight", dpi=300)

    return fig