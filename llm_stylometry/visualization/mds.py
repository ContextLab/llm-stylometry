"""3D MDS visualization for author stylometric distances."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from mpl_toolkits.mplot3d import Axes3D
from sklearn.manifold import MDS
import plotly.graph_objects as go
import logging

logger = logging.getLogger(__name__)

# Define author colors to match existing figures
AUTHOR_COLORS = {
    "baum": "#1f77b4",      # Blue
    "thompson": "#ff7f0e",   # Orange
    "austen": "#2ca02c",     # Green
    "dickens": "#d62728",    # Red
    "fitzgerald": "#9467bd", # Purple
    "melville": "#8c564b",   # Brown
    "twain": "#e377c2",      # Pink
    "wells": "#7f7f7f",      # Gray
}

# Standardized author order
AUTHOR_ORDER = ["baum", "thompson", "austen", "dickens", "fitzgerald", "melville", "twain", "wells"]


def create_loss_matrix(df, metric="mean"):
    """
    Create a loss matrix from model results DataFrame.

    Args:
        df: DataFrame with model results
        metric: Aggregation metric ('mean' or 'median')

    Returns:
        8x8 numpy array of cross-entropy losses
    """
    # Filter to final epoch for each model
    final_losses = []

    for model_name in df['model_name'].unique():
        model_df = df[df['model_name'] == model_name]
        # Get the maximum epoch for this model
        max_epoch = model_df['epochs_completed'].max()
        # Get losses at final epoch
        final_epoch_df = model_df[model_df['epochs_completed'] == max_epoch]
        final_losses.append(final_epoch_df)

    final_df = pd.concat(final_losses, ignore_index=True)

    # Create loss matrix
    loss_matrix = np.zeros((len(AUTHOR_ORDER), len(AUTHOR_ORDER)))

    for i, train_author in enumerate(AUTHOR_ORDER):
        for j, eval_author in enumerate(AUTHOR_ORDER):
            # Get losses where model trained on train_author evaluated on eval_author
            subset = final_df[
                (final_df['train_author'] == train_author) &
                (final_df['loss_dataset'] == eval_author)
            ]

            if len(subset) > 0:
                if metric == "mean":
                    loss_matrix[i, j] = subset['loss_value'].mean()
                elif metric == "median":
                    loss_matrix[i, j] = subset['loss_value'].median()
                else:
                    raise ValueError(f"Unknown metric: {metric}")

    return loss_matrix


def symmetrize_matrix(matrix):
    """
    Symmetrize a matrix by averaging with its transpose.

    Args:
        matrix: Input matrix

    Returns:
        Symmetrized matrix
    """
    return (matrix + matrix.T) / 2


def create_3d_mds_plot(loss_matrix, output_path=None, interactive=False):
    """
    Generate 3D MDS plot from loss matrix.

    Args:
        loss_matrix: 8x8 matrix of cross-entropy losses
        output_path: Path to save static PDF (optional)
        interactive: Whether to create interactive Plotly plot

    Returns:
        If interactive=True, returns Plotly figure object
    """
    # Symmetrize the matrix for MDS
    symmetric_matrix = symmetrize_matrix(loss_matrix)

    # Apply MDS with 3 components
    mds = MDS(n_components=3, dissimilarity='precomputed', random_state=42)
    coords = mds.fit_transform(symmetric_matrix)

    if interactive:
        # Create interactive Plotly 3D plot
        fig = go.Figure()

        for i, author in enumerate(AUTHOR_ORDER):
            fig.add_trace(go.Scatter3d(
                x=[coords[i, 0]],
                y=[coords[i, 1]],
                z=[coords[i, 2]],
                mode='markers+text',
                marker=dict(
                    size=12,
                    color=AUTHOR_COLORS[author],
                    line=dict(color='black', width=1)
                ),
                text=author.capitalize(),
                textposition='top center',
                name=author.capitalize(),
                hovertemplate=f"{author.capitalize()}<br>X: %{{x:.3f}}<br>Y: %{{y:.3f}}<br>Z: %{{z:.3f}}"
            ))

        fig.update_layout(
            title="3D MDS Projection of Author Stylometric Distances",
            scene=dict(
                xaxis_title="MDS Dimension 1",
                yaxis_title="MDS Dimension 2",
                zaxis_title="MDS Dimension 3",
                bgcolor="white",
                xaxis=dict(gridcolor='lightgray', zerolinecolor='gray'),
                yaxis=dict(gridcolor='lightgray', zerolinecolor='gray'),
                zaxis=dict(gridcolor='lightgray', zerolinecolor='gray'),
            ),
            showlegend=True,
            width=800,
            height=600,
            margin=dict(l=0, r=0, t=40, b=0)
        )

        if output_path:
            # Save interactive HTML
            import plotly.io as pio
            html_path = str(output_path).replace('.pdf', '.html')
            fig.write_html(html_path)
            logger.info(f"Saved interactive plot to {html_path}")

        return fig

    else:
        # Create static matplotlib 3D plot
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Plot each author
        for i, author in enumerate(AUTHOR_ORDER):
            ax.scatter(coords[i, 0], coords[i, 1], coords[i, 2],
                      color=AUTHOR_COLORS[author],
                      s=200,
                      label=author.capitalize(),
                      edgecolors='black',
                      linewidth=1,
                      alpha=0.8)

            # Add text labels
            ax.text(coords[i, 0], coords[i, 1], coords[i, 2],
                   author.capitalize(),
                   fontsize=10,
                   ha='center',
                   va='bottom')

        ax.set_xlabel('MDS Dimension 1', fontsize=12, labelpad=10)
        ax.set_ylabel('MDS Dimension 2', fontsize=12, labelpad=10)
        ax.set_zlabel('MDS Dimension 3', fontsize=12, labelpad=10)
        ax.set_title('3D MDS Projection of Author Stylometric Distances', fontsize=14, pad=20)

        # Add grid
        ax.grid(True, alpha=0.3)

        # Set viewing angle for better visibility
        ax.view_init(elev=20, azim=45)

        # Add legend
        ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1), ncol=1)

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved static plot to {output_path}")

        return fig


def plot_mds_from_dataframe(df_path="data/model_results.pkl", output_path=None, interactive=False):
    """
    Convenience function to create MDS plot directly from model results.

    Args:
        df_path: Path to model_results.pkl
        output_path: Path to save plot
        interactive: Whether to create interactive plot

    Returns:
        Figure object
    """
    # Load data
    df = pd.read_pickle(df_path)

    # Create loss matrix
    loss_matrix = create_loss_matrix(df)

    # Generate plot
    return create_3d_mds_plot(loss_matrix, output_path, interactive)


if __name__ == "__main__":
    # Test the implementation
    import sys
    from pathlib import Path

    # Check if model results exist
    results_path = Path("data/model_results.pkl")
    if not results_path.exists():
        print("Error: data/model_results.pkl not found. Run consolidate_model_results.py first.")
        sys.exit(1)

    # Generate both static and interactive plots
    print("Generating 3D MDS plots...")

    # Static plot for paper
    static_fig = plot_mds_from_dataframe(
        output_path="paper/figs/source/3d_MDS_plot_new.pdf",
        interactive=False
    )
    plt.show()

    # Interactive plot for exploration
    interactive_fig = plot_mds_from_dataframe(
        output_path="paper/figs/source/3d_MDS_plot_interactive.html",
        interactive=True
    )

    print("Plots generated successfully!")