#!/usr/bin/env python
"""
Generate figures for the dataset-size (ntokens) analysis.

Usage:
    python code/generate_ntokens_figures.py

Generates:
    paper/figs/source/t_test_ntokens.pdf
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib
matplotlib.use('Agg')

from llm_stylometry.visualization.t_tests import generate_t_test_ntokens_figure


def main():
    print("Generating t_test_ntokens.pdf...")
    fig = generate_t_test_ntokens_figure(
        data_path="data/model_results_ntokens.parquet",
        output_path="paper/figs/source/t_test_ntokens.pdf",
    )
    import matplotlib.pyplot as plt
    plt.close(fig)
    print("Done.")


if __name__ == "__main__":
    main()
