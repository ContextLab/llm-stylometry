"""Test script for 3D MDS visualization."""

from pathlib import Path
import sys
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

# Add package to path
sys.path.insert(0, str(Path(__file__).parent))

from llm_stylometry.visualization.mds import plot_mds_from_dataframe

# Check if model results exist
results_path = Path("data/model_results.pkl")
if not results_path.exists():
    print("Error: data/model_results.pkl not found.")
    sys.exit(1)

print("Generating 3D MDS plots...")

# Generate static plot
try:
    static_fig = plot_mds_from_dataframe(
        output_path="paper/figs/source/3d_MDS_plot_new.pdf",
        interactive=False
    )
    print("✓ Static PDF generated: paper/figs/source/3d_MDS_plot_new.pdf")
except Exception as e:
    print(f"Error generating static plot: {e}")

# Generate interactive plot
try:
    interactive_fig = plot_mds_from_dataframe(
        output_path="paper/figs/source/3d_MDS_plot_interactive.html",
        interactive=True
    )
    print("✓ Interactive HTML generated: paper/figs/source/3d_MDS_plot_interactive.html")
except Exception as e:
    print(f"Error generating interactive plot: {e}")

print("\nPlots generated successfully!")