"""
Visualization functions for embedding-based authorship attribution comparison.

Generates:
  - Main paper: bar chart comparing book-level accuracy across models
  - Appendix: purity distribution and confusion heatmap
"""

import matplotlib

matplotlib.use("Agg")
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "code"))
from constants import AUTHORS


def _short_model_name(model_name):
    """Extract a short display name from a HuggingFace model name."""
    parts = model_name.split("/")
    return parts[-1] if len(parts) > 1 else model_name


def generate_embedding_comparison_figure(
    summaries,
    output_path=None,
    figsize=(6, 4),
    font="Helvetica",
):
    """
    Generate bar chart comparing book-level accuracy across embedding models
    and our cross-entropy approach (100% baseline).

    Args:
        summaries: List of summary dicts (from compute_summary)
        output_path: Path to save PDF
        figsize: Figure size
        font: Font family
    """
    plt.rcParams["font.family"] = font
    plt.rcParams["font.sans-serif"] = [font]

    fig, ax = plt.subplots(figsize=figsize)

    # Prepare data: models + our method
    model_names = [_short_model_name(s["model"]) for s in summaries]
    accuracies = [s["overall_accuracy"] for s in summaries]

    # Add our method
    model_names.append("Predictive\ncomparison")
    accuracies.append(100.0)

    # Colors: embedding models in graded blue, our method in red
    n_emb = len(summaries)
    emb_colors = sns.color_palette("Blues_d", n_colors=n_emb)
    colors = list(emb_colors) + ["#b2182b"]

    x = np.arange(len(model_names))
    bars = ax.bar(x, accuracies, color=colors, edgecolor="white", linewidth=0.5)

    # Add accuracy labels on bars
    for bar, acc in zip(bars, accuracies):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{acc:.1f}%",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(model_names, fontsize=9)
    ax.set_ylabel("Book-level accuracy (%)", fontsize=12)
    ax.set_ylim(0, 110)

    sns.despine(ax=ax, top=True, right=True)
    plt.tight_layout()

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, format="pdf", bbox_inches="tight")

    return fig


def generate_embedding_purity_figure(
    all_book_results,
    output_path=None,
    figsize=(8, 4),
    font="Helvetica",
):
    """
    Generate purity distribution figure (appendix).

    Shows box/strip plot of purity scores per model per author.

    Args:
        all_book_results: Dict mapping model_name -> list of result dicts
        output_path: Path to save PDF
    """
    plt.rcParams["font.family"] = font
    plt.rcParams["font.sans-serif"] = [font]

    rows = []
    for model_name, results in all_book_results.items():
        for r in results:
            rows.append(
                {
                    "Model": _short_model_name(model_name),
                    "Author": r["true_author"].capitalize(),
                    "Purity": r["purity"],
                }
            )
    df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=figsize)

    sns.boxplot(
        data=df,
        x="Author",
        y="Purity",
        hue="Model",
        ax=ax,
        fliersize=3,
        linewidth=0.8,
    )
    sns.stripplot(
        data=df,
        x="Author",
        y="Purity",
        hue="Model",
        ax=ax,
        dodge=True,
        alpha=0.4,
        size=3,
        legend=False,
    )

    ax.set_ylabel("Purity (fraction of chunks voting for modal author)", fontsize=11)
    ax.set_xlabel("")
    ax.set_ylim(0, 1.05)
    ax.axhline(y=1.0, color="gray", linestyle="--", linewidth=0.5, alpha=0.5)

    handles, labels = ax.get_legend_handles_labels()
    n_models = len(all_book_results)
    ax.legend(
        handles[:n_models],
        labels[:n_models],
        title="Model",
        fontsize=8,
        title_fontsize=9,
        loc="lower left",
    )

    sns.despine(ax=ax, top=True, right=True)
    plt.tight_layout()

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, format="pdf", bbox_inches="tight")

    return fig


def generate_embedding_confusion_figure(
    all_book_results,
    output_path=None,
    figsize=(12, 4),
    font="Helvetica",
):
    """
    Generate confusion heatmap figure (appendix).

    One subplot per model showing true author vs predicted author.

    Args:
        all_book_results: Dict mapping model_name -> list of result dicts
        output_path: Path to save PDF
    """
    plt.rcParams["font.family"] = font
    plt.rcParams["font.sans-serif"] = [font]

    n_models = len(all_book_results)
    fig, axes = plt.subplots(1, n_models, figsize=figsize, sharey=True)
    if n_models == 1:
        axes = [axes]

    author_labels = [a.capitalize() for a in AUTHORS]

    for idx, (model_name, results) in enumerate(all_book_results.items()):
        ax = axes[idx]

        # Build confusion matrix
        confusion = np.zeros((len(AUTHORS), len(AUTHORS)))
        for r in results:
            true_idx = AUTHORS.index(r["true_author"])
            pred_idx = AUTHORS.index(r["modal_author"])
            confusion[true_idx, pred_idx] += 1

        # Normalize by row (true author)
        row_sums = confusion.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        confusion_norm = confusion / row_sums * 100

        sns.heatmap(
            confusion_norm,
            ax=ax,
            annot=True,
            fmt=".0f",
            xticklabels=author_labels,
            yticklabels=author_labels if idx == 0 else False,
            cmap="Blues",
            vmin=0,
            vmax=100,
            cbar=idx == n_models - 1,
            linewidths=0.5,
            linecolor="white",
            annot_kws={"fontsize": 8},
        )

        ax.set_title(_short_model_name(model_name), fontsize=11)
        ax.set_xlabel("Predicted author", fontsize=10)
        if idx == 0:
            ax.set_ylabel("True author", fontsize=10)
        ax.tick_params(labelsize=8)

    plt.tight_layout()

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, format="pdf", bbox_inches="tight")

    return fig
