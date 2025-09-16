#!/usr/bin/env python
"""
Direct figure generation script for LLM Stylometry paper.
Runs the visualization functions directly without notebooks.
"""

import sys
import os
from pathlib import Path

# Add package to path
sys.path.insert(0, str(Path(__file__).parent))

# Set matplotlib to non-interactive backend
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from llm_stylometry.visualization import (
    generate_all_losses_figure,
    generate_stripplot_figure,
    generate_t_test_figure,
    generate_t_test_avg_figure,
    generate_loss_heatmap_figure,
    generate_3d_mds_figure,
    generate_oz_losses_figure
)


def main():
    """Generate all figures for the paper."""

    print("\n" + "╔" + "═" * 58 + "╗")
    print("║" + " " * 10 + "LLM Stylometry Direct Figure Generator" + " " * 10 + "║")
    print("╚" + "═" * 58 + "╝")

    # Check for data file
    data_file = Path('data/model_results.pkl')
    if not data_file.exists():
        print("\nERROR: Required data file not found: data/model_results.pkl")
        print("Please ensure you have the consolidated model results.")
        sys.exit(1)

    print("\n✓ Found model results data")

    # Ensure output directory exists
    output_dir = Path('paper/figs/source')
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("Generating Figures")
    print("=" * 60)

    figures = [
        ('Figure 1A: Training curves',
         lambda: generate_all_losses_figure(
             data_path='data/model_results.pkl',
             output_path='paper/figs/source/all_losses.pdf',
             show_legend=False
         )),
        ('Figure 1B: Strip plot',
         lambda: generate_stripplot_figure(
             data_path='data/model_results.pkl',
             output_path='paper/figs/source/stripplot.pdf',
             show_legend=False
         )),
        ('Figure 2A: Individual t-tests',
         lambda: generate_t_test_figure(
             data_path='data/model_results.pkl',
             output_path='paper/figs/source/t_test.pdf',
             show_legend=False
         )),
        ('Figure 2B: Average t-test',
         lambda: generate_t_test_avg_figure(
             data_path='data/model_results.pkl',
             output_path='paper/figs/source/t_test_avg.pdf',
             show_legend=False
         )),
        ('Figure 3: Confusion matrix',
         lambda: generate_loss_heatmap_figure(
             data_path='data/model_results.pkl',
             output_path='paper/figs/source/average_loss_heatmap.pdf'
         )),
        ('Figure 4: 3D MDS plot',
         lambda: generate_3d_mds_figure(
             data_path='data/model_results.pkl',
             output_path='paper/figs/source/3d_MDS_plot.pdf'
         )),
        ('Figure 5: Oz losses',
         lambda: generate_oz_losses_figure(
             data_path='data/model_results.pkl',
             output_path='paper/figs/source/oz_losses.pdf',
             show_legend=False
         )),
    ]

    success_count = 0
    failed_figures = []

    for description, generate_func in figures:
        print(f"\nGenerating {description}...")
        try:
            fig = generate_func()
            plt.close(fig)
            print(f"  ✓ Generated successfully")
            success_count += 1
        except Exception as e:
            print(f"  ✗ Error: {str(e)[:100]}")
            failed_figures.append(description)

    # Verify outputs
    print("\n" + "=" * 60)
    print("Verifying Output Files")
    print("=" * 60)

    expected_files = [
        ('paper/figs/source/all_losses.pdf', 'Figure 1A'),
        ('paper/figs/source/stripplot.pdf', 'Figure 1B'),
        ('paper/figs/source/t_test.pdf', 'Figure 2A'),
        ('paper/figs/source/t_test_avg.pdf', 'Figure 2B'),
        ('paper/figs/source/average_loss_heatmap.pdf', 'Figure 3'),
        ('paper/figs/source/3d_MDS_plot.pdf', 'Figure 4'),
        ('paper/figs/source/oz_losses.pdf', 'Figure 5'),
    ]

    for file_path, name in expected_files:
        path = Path(file_path)
        if path.exists():
            size_kb = path.stat().st_size / 1024
            print(f"  ✓ {name}: {size_kb:.1f} KB")
        else:
            print(f"  ✗ {name}: NOT FOUND")

    # Summary
    print("\n" + "=" * 60)
    if success_count == len(figures):
        print("✓ All figures generated successfully!")
        print("=" * 60)
        print("\nFigures are saved in: paper/figs/source/")
    else:
        print(f"⚠ Generated {success_count}/{len(figures)} figures")
        if failed_figures:
            print("\nFailed figures:")
            for fig in failed_figures:
                print(f"  - {fig}")
        print("\nPlease check the error messages above.")

    return success_count == len(figures)


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)