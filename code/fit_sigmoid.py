#!/usr/bin/env python
"""
Fit a sigmoid to classification accuracy as a function of training tokens.

Loads per-author, per-seed accuracy from the dataset-size sweep results
(data/model_results_ntokens.pkl.gz) and produces a single-panel figure:
  - Per-author accuracy dots (colored by author)
  - Black sigmoid curve fit to mean accuracy across authors
  - 95% bootstrap confidence interval ribbon
  - Labeled vertical line at the minimum tokens for >=95% expected accuracy

Usage:
    python code/fit_sigmoid.py

Generates: paper/figs/source/accuracy_vs_tokens_sigmoid.pdf
"""

import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, str(Path(__file__).parent))
from constants import AUTHORS


def load_per_author_accuracy(data_path="data/model_results_ntokens.pkl.gz"):
    """
    Load per-author, per-n_train_tokens accuracy from the sweep results.

    Returns:
        per_author: DataFrame with columns: n_tokens, author, accuracy (%)
        mean_acc: DataFrame with columns: n_tokens, accuracy (%)
    """
    data_path = Path(data_path)
    if data_path.suffix == ".parquet":
        df = pd.read_parquet(data_path)
    else:
        df = pd.read_pickle(data_path)

    if "variant" in df.columns:
        df = df[df["variant"].isna()].copy()

    # Get final epoch per model per eval author
    final_df = (
        df[df["loss_dataset"].isin(AUTHORS)]
        .groupby(["model_name", "loss_dataset"])
        .tail(1)
        .copy()
    )

    # Predict: author with minimum loss for each model
    predictions = final_df.loc[
        final_df.groupby("model_name")["loss_value"].idxmin(),
        ["model_name", "train_author", "loss_dataset", "n_train_tokens"],
    ].copy()
    predictions = predictions.rename(columns={"loss_dataset": "predicted_author"})
    predictions["correct"] = (
        predictions["predicted_author"] == predictions["train_author"]
    )

    # Per author, per n_train_tokens accuracy (averaged over seeds)
    per_author = (
        predictions.groupby(["n_train_tokens", "train_author"])["correct"]
        .mean()
        .reset_index()
    )
    per_author.columns = ["n_tokens", "author", "accuracy"]
    per_author["accuracy"] *= 100

    # Mean accuracy per token level
    mean_acc = predictions.groupby("n_train_tokens")["correct"].mean().reset_index()
    mean_acc.columns = ["n_tokens", "accuracy"]
    mean_acc["accuracy"] *= 100

    return per_author, mean_acc


def sigmoid(x, L, K, b, m):
    """Sigmoid: y = L + K / (1 + exp(-b * (x - m)))"""
    return L + K / (1.0 + np.exp(-b * (x - m)))


def fit_sigmoid(log_x, y, p0=None):
    """Fit sigmoid to data, return parameters and covariance."""
    if p0 is None:
        p0 = [65.0, 35.0, 5.0, 4.4]

    popt, pcov = curve_fit(
        sigmoid,
        log_x,
        y,
        p0=p0,
        bounds=([0, 0, 0, 3], [100, 100, 50, 6]),
        maxfev=10000,
    )
    return popt, pcov


def find_threshold_tokens(popt, target_accuracy=95.0):
    """Find minimum tokens for target accuracy via inverse sigmoid."""
    L, K, b, m = popt
    ratio = K / (target_accuracy - L)
    if ratio <= 1:
        return None
    log_x = m - (1.0 / b) * np.log(ratio - 1.0)
    return 10**log_x


def bootstrap_sigmoid_ci(per_author_df, n_bootstrap=1000, seed=42):
    """
    Bootstrap the sigmoid fit by resampling authors.

    For each iteration, resample 8 authors with replacement,
    compute mean accuracy per token level, and fit sigmoid.

    Returns:
        popt_main: Parameters from the main fit
        bootstrap_curves: array (n_bootstrap, n_x_points)
        bootstrap_thresholds: list of threshold values
        x_smooth: x values for the curves
    """
    rng = np.random.default_rng(seed)
    authors = sorted(per_author_df["author"].unique())
    token_levels = sorted(per_author_df["n_tokens"].unique())
    log_tokens = np.log10(np.array(token_levels, dtype=float))

    # Main fit on mean accuracy
    mean_acc = per_author_df.groupby("n_tokens")["accuracy"].mean().values
    popt_main, _ = fit_sigmoid(log_tokens, mean_acc)

    x_smooth = np.linspace(log_tokens.min() - 0.1, log_tokens.max() + 0.1, 500)

    bootstrap_curves = []
    bootstrap_thresholds = []

    for _ in range(n_bootstrap):
        sampled_authors = rng.choice(authors, size=len(authors), replace=True)
        sampled_rows = [
            per_author_df[per_author_df["author"] == a] for a in sampled_authors
        ]
        sampled_df = pd.concat(sampled_rows)
        sampled_mean = sampled_df.groupby("n_tokens")["accuracy"].mean().values

        try:
            bp, _ = fit_sigmoid(log_tokens, sampled_mean, p0=popt_main)
            bootstrap_curves.append(sigmoid(x_smooth, *bp))
            bt = find_threshold_tokens(bp, 95.0)
            if bt is not None and 100 < bt < 1e7:
                bootstrap_thresholds.append(bt)
        except (RuntimeError, ValueError):
            continue

    return popt_main, np.array(bootstrap_curves), bootstrap_thresholds, x_smooth


def generate_accuracy_sigmoid_figure(
    data_path="data/model_results_ntokens.pkl.gz",
    output_path=None,
    figsize=(5, 3.5),
    font="Helvetica",
    n_bootstrap=1000,
):
    """Generate single-panel figure with per-author dots, sigmoid fit, and CI ribbon."""
    plt.rcParams["font.family"] = font
    plt.rcParams["font.sans-serif"] = [font]

    # Load data
    per_author, mean_acc = load_per_author_accuracy(data_path)
    tokens_arr = np.array(sorted(mean_acc["n_tokens"].unique()), dtype=float)
    log_tokens = np.log10(tokens_arr)
    acc_arr = mean_acc.sort_values("n_tokens")["accuracy"].values

    # Fit and bootstrap
    print("Fitting sigmoid and bootstrapping CI...")
    popt, bootstrap_curves, bootstrap_thresholds, x_smooth = bootstrap_sigmoid_ci(
        per_author, n_bootstrap=n_bootstrap
    )
    L, K, b, m = popt

    # R-squared
    y_pred = sigmoid(log_tokens, *popt)
    ss_res = np.sum((acc_arr - y_pred) ** 2)
    ss_tot = np.sum((acc_arr - np.mean(acc_arr)) ** 2)
    r_squared = 1 - ss_res / ss_tot

    # Residual diagnostics
    residuals = acc_arr - y_pred
    rmse = np.sqrt(np.mean(residuals**2))
    max_residual = np.max(np.abs(residuals))

    # Threshold
    threshold_tokens_95 = find_threshold_tokens(popt, 95.0)

    # Print results
    print(
        f"Sigmoid fit: y = {L:.1f} + {K:.1f} / (1 + exp(-{b:.2f} * (log10(x) - {m:.2f})))"
    )
    print(f"  L (lower asymptote) = {L:.2f}")
    print(f"  K (range)           = {K:.2f}")
    print(f"  b (steepness)       = {b:.2f}")
    print(f"  m (midpoint log10)  = {m:.2f}")
    print(f"  Midpoint            = {10**m:,.0f} tokens")
    print(f"  R²                  = {r_squared:.4f}")
    print(f"  Residuals: RMSE={rmse:.2f}%, max={max_residual:.2f}%")
    if threshold_tokens_95:
        print(f"  Tokens for ≥95%     = {threshold_tokens_95:,.0f}")
    if bootstrap_thresholds:
        bt_arr = np.array(bootstrap_thresholds)
        ci_lo, ci_hi = np.percentile(bt_arr, [2.5, 97.5])
        print(f"  95% CI for threshold: [{ci_lo:,.0f}, {ci_hi:,.0f}] tokens")

    # --- Figure ---
    fig, ax = plt.subplots(figsize=figsize)

    # Bootstrap CI ribbon (no legend entry)
    if len(bootstrap_curves) > 0:
        ci_lower = np.percentile(bootstrap_curves, 2.5, axis=0)
        ci_upper = np.percentile(bootstrap_curves, 97.5, axis=0)
        ax.fill_between(
            10**x_smooth,
            ci_lower,
            ci_upper,
            alpha=0.15,
            color="black",
        )

    # Sigmoid fit curve (black, no legend entry)
    y_smooth = sigmoid(x_smooth, *popt)
    ax.plot(10**x_smooth, y_smooth, "-", color="black", linewidth=2, zorder=3)

    # Per-author dots with jitter (display only), small and transparent
    author_order = [
        "baum",
        "thompson",
        "austen",
        "dickens",
        "fitzgerald",
        "melville",
        "twain",
        "wells",
    ]
    palette = dict(
        zip(author_order, sns.color_palette("tab10", n_colors=len(author_order)))
    )
    jitter_rng = np.random.default_rng(123)

    for author in author_order:
        author_data = per_author[per_author["author"] == author]
        jitter_factor = jitter_rng.uniform(0.92, 1.08, size=len(author_data))
        jittered_tokens = author_data["n_tokens"].values * jitter_factor
        ax.scatter(
            jittered_tokens,
            author_data["accuracy"].values,
            color=palette[author],
            s=15,
            alpha=0.4,
            zorder=4,
            label=author.capitalize(),
            edgecolors="none",
        )

    # Mean accuracy markers (squares, prominent)
    mean_sorted = mean_acc.sort_values("n_tokens")
    ax.scatter(
        mean_sorted["n_tokens"].values,
        mean_sorted["accuracy"].values,
        color="black",
        s=40,
        zorder=5,
        marker="s",
        alpha=0.6,
        edgecolors="white",
        linewidth=0.5,
        label="Mean",
    )

    # 95% threshold
    ax.axhline(y=95, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    if threshold_tokens_95:
        ax.axvline(
            x=threshold_tokens_95,
            color="gray",
            linestyle=":",
            linewidth=1,
            alpha=0.7,
        )
        ax.annotate(
            f"{threshold_tokens_95:,.0f}\ntokens",
            xy=(threshold_tokens_95, 95),
            xytext=(threshold_tokens_95 * 4, 82),
            fontsize=8,
            arrowprops=dict(
                arrowstyle="->",
                color="black",
                lw=1.5,
                mutation_scale=15,
            ),
            color="black",
            ha="center",
            zorder=10,
        )

    ax.set_xscale("log")
    ax.set_xlabel("Training tokens per author", fontsize=12)
    ax.set_ylabel("Attribution accuracy (%)", fontsize=12)
    ax.set_ylim(20, 105)
    ax.set_xlim(1500, 800000)

    ax.legend(
        fontsize=8,
        title_fontsize=9,
        loc="lower right",
        framealpha=0.9,
        ncol=2,
    )

    sns.despine(ax=ax, top=True, right=True)
    plt.tight_layout()

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, format="pdf", bbox_inches="tight")
        print(f"\nFigure saved to: {output_path}")

    # Save fit results for use by other figures (e.g., t-test ntokens)
    import json

    results_path = Path("data/sigmoid_fit_results.json")
    results_path.parent.mkdir(parents=True, exist_ok=True)
    fit_results = {
        "L": float(L),
        "K": float(K),
        "b": float(b),
        "m": float(m),
        "r_squared": float(r_squared),
        "threshold_tokens_95": (
            float(threshold_tokens_95) if threshold_tokens_95 else None
        ),
        "bootstrap_ci_lo": float(ci_lo) if bootstrap_thresholds else None,
        "bootstrap_ci_hi": float(ci_hi) if bootstrap_thresholds else None,
    }
    with open(results_path, "w") as f:
        json.dump(fit_results, f, indent=2)

    return fig, popt


if __name__ == "__main__":
    fig, popt = generate_accuracy_sigmoid_figure(
        output_path="paper/figs/source/accuracy_vs_tokens_sigmoid.pdf",
    )
    plt.close(fig)
