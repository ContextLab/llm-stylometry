"""Generate t-test figures from the paper."""

import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import t as t_dist
from scipy.stats import ttest_ind
from tqdm import tqdm

logger = logging.getLogger(__name__)


def calculate_t_statistics(df, max_epochs=500):
    """
    Calculate t-statistics and df comparing same vs other author losses.

    Returns:
        tuple: (t_raws_df, t_raws, df_values, thresholds)
            - t_raws_df: Long-form DataFrame with columns [Epoch, Author, t_raw]
            - t_raws: Dict mapping author to list of t-values
            - df_values: Dict mapping author to list of degrees of freedom
            - thresholds: Dict mapping author to list of t-thresholds for p=0.001
    """

    # Define authors
    AUTHORS = [
        "baum",
        "thompson",
        "dickens",
        "melville",
        "wells",
        "austen",
        "fitzgerald",
        "twain",
    ]

    # Filter and prepare data
    t_df = df[df["loss_dataset"].isin(AUTHORS)].copy()
    t_df = t_df[t_df["epochs_completed"] <= max_epochs]
    t_df["loss_dataset"] = t_df["loss_dataset"].str.capitalize()
    t_df["train_author"] = t_df["train_author"].str.capitalize()

    # Prepare authors and epochs
    authors = sorted(t_df["train_author"].unique())
    epochs = sorted(t_df["epochs_completed"].unique())
    t_raws = {author: [] for author in authors}
    df_values = {author: [] for author in authors}
    thresholds = {author: [] for author in authors}

    # Compute Welch's t-statistic for each author/epoch
    for author in tqdm(authors, desc="Processing authors"):
        for epoch in epochs:
            true_losses = t_df[
                (t_df["train_author"] == author)
                & (t_df["loss_dataset"] == author)
                & (t_df["epochs_completed"] == epoch)
            ]["loss_value"].values

            other_losses = t_df[
                (t_df["train_author"] == author)
                & (t_df["loss_dataset"] != author)
                & (t_df["epochs_completed"] == epoch)
            ]["loss_value"].values

            # T-test requires at least 2 samples per group for meaningful results
            if len(true_losses) >= 2 and len(other_losses) >= 2:
                result = ttest_ind(other_losses, true_losses, equal_var=False)
                if np.isnan(result.statistic):
                    logger.debug(
                        f"NaN t-statistic for {author} at epoch {epoch}: "
                        f"n_true={len(true_losses)}, n_other={len(other_losses)}"
                    )
                t_raws[author].append(result.statistic)
                df_values[author].append(result.df)

                # Compute t-threshold for p=0.001 (one-tailed) given this df
                t_threshold = t_dist.ppf(1 - 0.001, result.df)
                thresholds[author].append(t_threshold)
            elif len(true_losses) > 0 or len(other_losses) > 0:
                # Have some data but insufficient for t-test
                logger.debug(
                    f"Insufficient data for t-test for {author} at epoch {epoch}: "
                    f"n_true={len(true_losses)}, n_other={len(other_losses)} "
                    f"(need at least 2 samples per group)"
                )
                t_raws[author].append(np.nan)
                df_values[author].append(np.nan)
                thresholds[author].append(np.nan)
            else:
                # No data at all
                logger.debug(f"No data for {author} at epoch {epoch}")
                t_raws[author].append(np.nan)
                df_values[author].append(np.nan)
                thresholds[author].append(np.nan)

    # Convert to long-form DataFrame
    t_raws_df = (
        pd.DataFrame(t_raws, index=epochs)
        .reset_index()
        .melt(id_vars="index", var_name="Author", value_name="t_raw")
        .rename(columns={"index": "Epoch"})
    )

    return t_raws_df, t_raws, df_values, thresholds


def generate_t_test_figure(
    data_path="data/model_results.pkl",
    output_path=None,
    figsize=(6, 4),
    show_legend=False,
    font="Helvetica",
    variant=None,
):
    """
    Generate Figure 2A: t-statistics for individual authors.

    Args:
        data_path: Path to model_results.pkl
        output_path: Path to save PDF (optional)
        figsize: Figure size
        show_legend: Whether to show legend (False for paper)
        font: Font family to use

        variant: Analysis variant ('content', 'function', 'pos') or None for baseline

    Returns:
        matplotlib figure object
    """
    # Set font
    plt.rcParams["font.family"] = font
    plt.rcParams["font.sans-serif"] = [font]

    # Load data and calculate t-statistics
    df = pd.read_pickle(data_path)

    # Filter by variant
    if variant is None:
        # Baseline: exclude variant models
        if "variant" in df.columns:
            df = df[df["variant"].isna()].copy()
    else:
        # Specific variant
        if "variant" not in df.columns:
            raise ValueError("No variant column in data")
        df = df[df["variant"] == variant].copy()

    t_raws_df, _, df_values, thresholds = calculate_t_statistics(df)

    # Compute average threshold across authors at each epoch (for plotting)
    epochs = sorted(t_raws_df["Epoch"].unique())
    threshold_data = []
    for epoch in epochs:
        epoch_thresholds = []
        for author in thresholds.keys():
            epoch_idx = list(epochs).index(epoch)
            if epoch_idx < len(thresholds[author]):
                thresh = thresholds[author][epoch_idx]
                if not np.isnan(thresh):
                    epoch_thresholds.append(thresh)

        # For each epoch, add one row per author's threshold (for bootstrap CI calculation)
        for thresh in epoch_thresholds:
            threshold_data.append({"Epoch": epoch, "threshold": thresh})

    threshold_df = pd.DataFrame(threshold_data)

    # Define color palette
    unique_authors = sorted(t_raws_df["Author"].unique())
    fixed_first = ["Baum", "Thompson"]
    hue_order = fixed_first + [a for a in unique_authors if a not in fixed_first]
    palette = dict(zip(hue_order, sns.color_palette("tab10", n_colors=len(hue_order))))

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Plot author t-statistics
    sns.lineplot(
        data=t_raws_df,
        x="Epoch",
        y="t_raw",
        hue="Author",
        ax=ax,
        hue_order=hue_order,
        palette=palette,
        legend=show_legend,
    )

    # Plot adaptive threshold with bootstrap 95% CI (solid black line)
    if not threshold_df.empty:
        sns.lineplot(
            data=threshold_df,
            x="Epoch",
            y="threshold",
            ax=ax,
            color="black",
            linewidth=2,
            linestyle="-",  # Solid line
            errorbar="ci",  # Bootstrap 95% CI
            label="p<0.001 threshold" if show_legend else "",
        )

    sns.despine(ax=ax, top=True, right=True)
    ax.set_xlabel("Epochs completed", fontsize=12)
    ax.set_ylabel("$t$-value", fontsize=12)

    # Calculate dynamic y-axis limits based on VALID data only
    valid_t_values = t_raws_df["t_raw"].replace([np.inf, -np.inf], np.nan).dropna()

    if len(valid_t_values) == 0:
        logger.warning("No valid t-statistics found. Using default axis limits.")
        y_min = -1.0
        y_max = 5.0
    else:
        y_min = valid_t_values.min()
        y_max = valid_t_values.max()

        # Add padding
        y_range = y_max - y_min
        padding = 0.05 * y_range if y_range > 0 else 0.5
        y_min = min(y_min, 0) - padding
        y_max = y_max + padding

    # Final validation
    if not (np.isfinite(y_min) and np.isfinite(y_max) and y_min < y_max):
        logger.error(
            f"Invalid axis limits computed: y_min={y_min}, y_max={y_max}. Using defaults."
        )
        y_min = -1.0
        y_max = 5.0
    ax.set_xlim(0, t_raws_df["Epoch"].max())
    ax.set_ylim(y_min, y_max)

    if show_legend:
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(
            handles=handles,
            labels=labels,
            title="Training author",
            fontsize=8,
            title_fontsize=9,
            loc="upper left",
        )

    plt.tight_layout()

    if output_path:
        # Add variant suffix to filename if variant specified
        if variant:
            from pathlib import Path

            output_path = Path(output_path)
            output_path = str(
                output_path.parent / f"{output_path.stem}_{variant}{output_path.suffix}"
            )
        fig.savefig(output_path, format="pdf", bbox_inches="tight")

    return fig


def generate_t_test_avg_figure(
    data_path="data/model_results.pkl",
    output_path=None,
    figsize=(6, 4),
    show_legend=False,
    font="Helvetica",
    variant=None,
):
    """
    Generate Figure 2B: Average t-statistic across all authors.

    Args:
        data_path: Path to model_results.pkl
        output_path: Path to save PDF (optional)
        figsize: Figure size
        show_legend: Whether to show legend (False for paper)
        font: Font family to use

        variant: Analysis variant ('content', 'function', 'pos') or None for baseline

    Returns:
        matplotlib figure object
    """
    # Set font
    plt.rcParams["font.family"] = font
    plt.rcParams["font.sans-serif"] = [font]

    # Load data and calculate t-statistics
    df = pd.read_pickle(data_path)

    # Filter by variant
    if variant is None:
        # Baseline: exclude variant models
        if "variant" in df.columns:
            df = df[df["variant"].isna()].copy()
    else:
        # Specific variant
        if "variant" not in df.columns:
            raise ValueError("No variant column in data")
        df = df[df["variant"] == variant].copy()

    t_raws_df, _, df_values, thresholds = calculate_t_statistics(df)

    # Compute average threshold across authors at each epoch
    epochs = sorted(t_raws_df["Epoch"].unique())
    threshold_data = []
    for epoch in epochs:
        epoch_thresholds = []
        for author in thresholds.keys():
            epoch_idx = list(epochs).index(epoch)
            if epoch_idx < len(thresholds[author]):
                thresh = thresholds[author][epoch_idx]
                if not np.isnan(thresh):
                    epoch_thresholds.append(thresh)

        for thresh in epoch_thresholds:
            threshold_data.append({"Epoch": epoch, "threshold": thresh})

    threshold_df = pd.DataFrame(threshold_data)

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Plot average t-statistic (gray for consistency with individual figure)
    sns.lineplot(
        data=t_raws_df,
        x="Epoch",
        y="t_raw",
        ax=ax,
        legend=False,
        color="gray",  # Set line color to gray
    )

    # Plot adaptive threshold with bootstrap 95% CI (solid black line, consistent with 2a)
    if not threshold_df.empty:
        sns.lineplot(
            data=threshold_df,
            x="Epoch",
            y="threshold",
            ax=ax,
            color="black",
            linewidth=2,
            linestyle="-",  # Solid line
            errorbar="ci",  # Bootstrap 95% CI
            label="p<0.001 threshold" if show_legend else "",
        )

    sns.despine(ax=ax, top=True, right=True)
    ax.set_xlabel("Epochs completed", fontsize=12)
    ax.set_ylabel("$t$-value", fontsize=12)

    # Calculate dynamic y-axis limits
    valid_t_values = t_raws_df["t_raw"].replace([np.inf, -np.inf], np.nan).dropna()

    if len(valid_t_values) == 0:
        logger.warning(
            "No valid t-statistics found for average figure. Using default axis limits."
        )
        y_min = -1.0
        y_max = 5.0
    else:
        y_min = valid_t_values.min()
        y_max = valid_t_values.max()

        # Add padding
        y_range = y_max - y_min
        padding = 0.05 * y_range if y_range > 0 else 0.5
        y_min = min(y_min, 0) - padding
        y_max = y_max + padding

    # Final validation
    if not (np.isfinite(y_min) and np.isfinite(y_max) and y_min < y_max):
        logger.error(
            f"Invalid axis limits computed for average figure: y_min={y_min}, y_max={y_max}. Using defaults."
        )
        y_min = -1.0
        y_max = 5.0

    ax.set_xlim(0, t_raws_df["Epoch"].max())
    ax.set_ylim(y_min, y_max)

    if show_legend:
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(
            handles=handles,
            labels=labels,
            fontsize=8,
            title_fontsize=9,
            loc="upper left",
        )

    plt.tight_layout()

    if output_path:
        # Add variant suffix to filename if variant specified
        if variant:
            from pathlib import Path

            output_path = Path(output_path)
            output_path = str(
                output_path.parent / f"{output_path.stem}_{variant}{output_path.suffix}"
            )
        fig.savefig(output_path, format="pdf", bbox_inches="tight")

    return fig


def _load_ntokens_t_test_panel_data(
    data_path="data/model_results_ntokens.parquet",
    cache_path="data/t_test_ntokens_cache",
):
    """Load baseline ntokens results and prepare per-size t-test data."""
    import json
    from pathlib import Path

    data_path = Path(data_path)
    cache_path = Path(cache_path)

    # Load from Parquet-based cache if available
    if cache_path.is_dir() and any(cache_path.glob("panel_*_meta.json")):
        panel_data = []
        meta_files = sorted(cache_path.glob("panel_*_meta.json"))
        for meta_file in meta_files:
            idx = meta_file.stem.split("_")[1]
            with open(meta_file) as f:
                meta = json.load(f)
            t_raws_df = pd.read_parquet(cache_path / f"panel_{idx}_t_raws.parquet")
            threshold_df = pd.read_parquet(
                cache_path / f"panel_{idx}_threshold.parquet"
            )
            panel_data.append(
                {
                    "n_train_tokens": meta["n_train_tokens"],
                    "label": meta["label"],
                    "t_raws_df": t_raws_df,
                    "threshold_df": threshold_df,
                }
            )
        return panel_data

    if data_path.suffix == ".parquet":
        df = pd.read_parquet(data_path)
    else:
        df = pd.read_pickle(data_path)
    if "variant" in df.columns:
        df = df[df["variant"].isna()].copy()
    if "n_train_tokens" not in df.columns:
        raise ValueError("No n_train_tokens column in data")

    ntokens_values = sorted(df["n_train_tokens"].dropna().unique())
    panel_data = []

    for n_train_tokens in ntokens_values:
        n_df = df[df["n_train_tokens"] == n_train_tokens].copy()
        if n_df.empty:
            raise ValueError(f"No rows found for n_train_tokens={n_train_tokens}")

        t_raws_df, _, _, thresholds = calculate_t_statistics(n_df)

        epochs = sorted(t_raws_df["Epoch"].unique())
        threshold_data = []
        for epoch in epochs:
            epoch_thresholds = []
            for author in thresholds.keys():
                epoch_idx = list(epochs).index(epoch)
                if epoch_idx < len(thresholds[author]):
                    thresh = thresholds[author][epoch_idx]
                    if not np.isnan(thresh):
                        epoch_thresholds.append(thresh)

            for thresh in epoch_thresholds:
                threshold_data.append({"Epoch": epoch, "threshold": thresh})

        panel_data.append(
            {
                "n_train_tokens": n_train_tokens,
                "label": f"{n_train_tokens:,}",
                "t_raws_df": t_raws_df,
                "threshold_df": pd.DataFrame(threshold_data),
            }
        )

    # Save cache as Parquet files (format-stable across numpy/pandas versions)
    cache_path.mkdir(parents=True, exist_ok=True)
    for i, panel in enumerate(panel_data):
        panel["t_raws_df"].to_parquet(
            cache_path / f"panel_{i}_t_raws.parquet", index=False
        )
        panel["threshold_df"].to_parquet(
            cache_path / f"panel_{i}_threshold.parquet", index=False
        )
        import json

        with open(cache_path / f"panel_{i}_meta.json", "w") as f:
            json.dump(
                {
                    "n_train_tokens": int(panel["n_train_tokens"]),
                    "label": panel["label"],
                },
                f,
            )

    return panel_data


def _compute_bootstrap_t_values(data_path, final_epoch=500, n_bootstrap=200, seed=42):
    """
    Compute t-values at the final epoch for each author and token level,
    with bootstrap resampling over seeds to produce CI data.

    For each (author, n_tokens), the pooled t-statistic uses all 10 seeds.
    Bootstrap: resample 10 seeds with replacement, recompute t-statistic.

    Returns:
        DataFrame with columns: n_tokens, Author, bootstrap_iter, t_value
        (iter=0 is the original, iter=1..n_bootstrap are bootstrap samples)
    """
    from pathlib import Path

    from scipy.stats import ttest_ind

    rng = np.random.default_rng(seed)
    data_path = Path(data_path)
    if data_path.suffix == ".parquet":
        df = pd.read_parquet(data_path)
    else:
        df = pd.read_pickle(data_path)

    if "variant" in df.columns:
        df = df[df["variant"].isna()].copy()

    authors = sorted(df["train_author"].unique())
    eval_df = df[
        (df["loss_dataset"].isin(authors)) & (df["epochs_completed"] == final_epoch)
    ].copy()

    all_seeds = sorted(eval_df["seed"].unique())
    rows = []

    for n_tokens in sorted(eval_df["n_train_tokens"].dropna().unique()):
        nt_df = eval_df[eval_df["n_train_tokens"] == n_tokens]
        for author in authors:
            author_df = nt_df[nt_df["train_author"] == author]

            for boot_iter in range(n_bootstrap + 1):
                if boot_iter == 0:
                    sampled_seeds = all_seeds
                else:
                    sampled_seeds = rng.choice(
                        all_seeds, size=len(all_seeds), replace=True
                    )

                self_losses = []
                other_losses = []
                for s in sampled_seeds:
                    seed_df = author_df[author_df["seed"] == s]
                    self_losses.extend(
                        seed_df[seed_df["loss_dataset"] == author]["loss_value"].values
                    )
                    other_losses.extend(
                        seed_df[
                            (seed_df["loss_dataset"] != author)
                            & (seed_df["loss_dataset"] != "train")
                        ]["loss_value"].values
                    )

                if len(self_losses) >= 2 and len(other_losses) >= 2:
                    result = ttest_ind(other_losses, self_losses, equal_var=False)
                    if np.isfinite(result.statistic):
                        rows.append(
                            {
                                "n_tokens": int(n_tokens),
                                "Author": author.capitalize(),
                                "bootstrap_iter": boot_iter,
                                "t_value": result.statistic,
                            }
                        )

    return pd.DataFrame(rows)


def generate_t_test_ntokens_figure(
    data_path="data/model_results_ntokens.parquet",
    output_path=None,
    figsize=(5, 3.5),
    font="Helvetica",
    final_epoch=500,
    show_legend=False,
    **kwargs,
):
    """
    Generate figure: final-epoch t-value vs training tokens, one curve per author,
    with 95% CI ribbons across seeds.

    Designed as panel B alongside the sigmoid figure (panel A).

    Args:
        data_path: Path to ntokens results
        output_path: Path to save PDF
        figsize: Figure size (smaller for 2-panel layout)
        font: Font family
        final_epoch: Epoch to extract t-values from (default 500)
        show_legend: Whether to show legend (default False for panel use)
    """
    plt.rcParams["font.family"] = font
    plt.rcParams["font.sans-serif"] = [font]

    # Compute bootstrap t-values from raw data
    logger.info("Computing bootstrap t-values at epoch %d...", final_epoch)
    df = _compute_bootstrap_t_values(data_path, final_epoch, n_bootstrap=200)

    fig, ax = plt.subplots(figsize=figsize)

    # Author colors matching other figures
    fixed_first = ["Baum", "Thompson"]
    unique_authors = sorted(df["Author"].unique())
    hue_order = fixed_first + [a for a in unique_authors if a not in fixed_first]
    palette = dict(zip(hue_order, sns.color_palette("tab10", n_colors=len(hue_order))))

    # Plot mean line + 95% CI ribbon per author from bootstrap iterations
    for author in hue_order:
        author_df = df[df["Author"] == author]
        if author_df.empty:
            continue

        # Original values (bootstrap_iter == 0) for the line
        orig = author_df[author_df["bootstrap_iter"] == 0].sort_values("n_tokens")
        boot = author_df[author_df["bootstrap_iter"] > 0]

        # Compute CI from bootstrap
        ci = (
            boot.groupby("n_tokens")["t_value"]
            .agg(
                ci_lo=lambda x: np.percentile(x, 2.5),
                ci_hi=lambda x: np.percentile(x, 97.5),
            )
            .reindex(orig["n_tokens"].values)
        )

        color = palette[author]
        ax.plot(
            orig["n_tokens"].values,
            orig["t_value"].values,
            marker="o",
            markersize=3,
            linewidth=1.2,
            color=color,
        )
        ax.fill_between(
            orig["n_tokens"].values,
            ci["ci_lo"].values,
            ci["ci_hi"].values,
            alpha=0.15,
            color=color,
        )

    # p<0.001 threshold line
    from scipy.stats import t as t_dist

    threshold = t_dist.ppf(1 - 0.001, 14)
    ax.axhline(
        y=threshold,
        color="black",
        linestyle="--",
        linewidth=1.5,
        alpha=0.7,
    )

    # Vertical line at minimum tokens for >=95% accuracy (from sigmoid fit)
    import json
    from pathlib import Path

    sigmoid_results_path = Path("data/sigmoid_fit_results.json")
    if sigmoid_results_path.exists():
        with open(sigmoid_results_path) as f:
            sigmoid_results = json.load(f)
        threshold_tokens = sigmoid_results.get("threshold_tokens_95")
        if threshold_tokens:
            ax.axvline(
                x=threshold_tokens,
                color="gray",
                linestyle=":",
                linewidth=1,
                alpha=0.7,
            )

    ax.set_xscale("log")
    ax.set_xlabel("Training tokens per author", fontsize=12)
    ax.set_ylabel(f"$t$-value (epoch {final_epoch})", fontsize=12)

    sns.despine(ax=ax, top=True, right=True)
    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, format="pdf", bbox_inches="tight")

    return fig


# Keep old functions for backward compatibility but mark as deprecated
def generate_t_test_ntokens_grid_figure(
    data_path="data/model_results_ntokens.parquet",
    output_path=None,
    figsize=(12, 16),
    font="Helvetica",
    panel_data=None,
):
    """Generate a Figure 2A-style grid across training-token counts."""
    plt.rcParams["font.family"] = font
    plt.rcParams["font.sans-serif"] = [font]

    if panel_data is None:
        panel_data = _load_ntokens_t_test_panel_data(data_path)

    ncols = 3
    nrows = int(np.ceil(len(panel_data) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, sharex=True, sharey=True)
    axes = np.atleast_1d(axes).flatten()
    combined_t_raws_df = pd.concat(
        [panel["t_raws_df"] for panel in panel_data],
        ignore_index=True,
    )

    valid_t_values = (
        combined_t_raws_df["t_raw"].replace([np.inf, -np.inf], np.nan).dropna()
    )
    if len(valid_t_values) == 0:
        logger.warning(
            "No valid t-statistics found for ntokens grid. Using default axis limits."
        )
        y_min = -1.0
        y_max = 5.0
    else:
        y_min = valid_t_values.min()
        y_max = valid_t_values.max()
        y_range = y_max - y_min
        padding = 0.05 * y_range if y_range > 0 else 0.5
        y_min = min(y_min, 0) - padding
        y_max = y_max + padding

    for i, panel in enumerate(panel_data):
        ax = axes[i]
        t_raws_df = panel["t_raws_df"]
        threshold_df = panel["threshold_df"]

        unique_authors = sorted(t_raws_df["Author"].unique())
        fixed_first = ["Baum", "Thompson"]
        hue_order = fixed_first + [a for a in unique_authors if a not in fixed_first]
        palette = dict(
            zip(hue_order, sns.color_palette("tab10", n_colors=len(hue_order)))
        )

        sns.lineplot(
            data=t_raws_df,
            x="Epoch",
            y="t_raw",
            hue="Author",
            ax=ax,
            hue_order=hue_order,
            palette=palette,
            legend=(i == ncols - 1),
        )

        if not threshold_df.empty:
            sns.lineplot(
                data=threshold_df,
                x="Epoch",
                y="threshold",
                ax=ax,
                color="black",
                linewidth=2,
                linestyle="-",
                errorbar="ci",
            )

        sns.despine(ax=ax, top=True, right=True)
        ax.set_title(f'{panel["label"]} tokens', fontsize=12)
        ax.set_xlim(0, t_raws_df["Epoch"].max())
        ax.set_ylim(y_min, y_max)

        if i % ncols == 0:
            ax.set_ylabel("$t$-value", fontsize=12)
        else:
            ax.set_ylabel("")

        legend = ax.get_legend()
        if i == ncols - 1:
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(
                handles=handles,
                labels=labels,
                title="Training author",
                fontsize=8,
                title_fontsize=9,
                loc="upper right",
            )
        elif legend is not None:
            legend.remove()

    for i, ax in enumerate(axes):
        if i >= len(panel_data):
            ax.set_visible(False)
            continue
        if i < len(panel_data) - ncols:
            ax.set_xlabel("")
        else:
            ax.set_xlabel("Epochs completed", fontsize=12)

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, format="pdf", bbox_inches="tight")

    return fig


def generate_t_test_avg_ntokens_figure(
    data_path="data/model_results_ntokens.parquet",
    output_path=None,
    figsize=(6, 4),
    show_legend=True,
    font="Helvetica",
    panel_data=None,
):
    """Generate a Figure 2B-style plot with one average curve per token count."""
    plt.rcParams["font.family"] = font
    plt.rcParams["font.sans-serif"] = [font]

    if panel_data is None:
        panel_data = _load_ntokens_t_test_panel_data(data_path)

    combined_t_raws_df = []
    combined_threshold_df = []
    for panel in panel_data:
        t_raws_df = panel["t_raws_df"].copy()
        t_raws_df["Training tokens"] = panel["label"]
        combined_t_raws_df.append(t_raws_df)
        combined_threshold_df.append(panel["threshold_df"])

    combined_t_raws_df = pd.concat(combined_t_raws_df, ignore_index=True)
    combined_threshold_df = pd.concat(combined_threshold_df, ignore_index=True)

    fig, ax = plt.subplots(figsize=figsize)

    sns.lineplot(
        data=combined_t_raws_df,
        x="Epoch",
        y="t_raw",
        hue="Training tokens",
        hue_order=[panel["label"] for panel in panel_data],
        palette=sns.color_palette("viridis", n_colors=len(panel_data)),
        ax=ax,
    )

    if not combined_threshold_df.empty:
        sns.lineplot(
            data=combined_threshold_df,
            x="Epoch",
            y="threshold",
            ax=ax,
            color="black",
            linewidth=2,
            linestyle="-",
            errorbar="ci",
            label="p<0.001 threshold" if show_legend else "",
        )

    sns.despine(ax=ax, top=True, right=True)
    ax.set_xlabel("Epochs completed", fontsize=12)
    ax.set_ylabel("$t$-value", fontsize=12)

    valid_t_values = (
        combined_t_raws_df["t_raw"].replace([np.inf, -np.inf], np.nan).dropna()
    )
    if len(valid_t_values) == 0:
        logger.warning(
            "No valid t-statistics found for ntokens average figure. Using default axis limits."
        )
        y_min = -1.0
        y_max = 5.0
    else:
        y_min = valid_t_values.min()
        y_max = valid_t_values.max()
        y_range = y_max - y_min
        padding = 0.05 * y_range if y_range > 0 else 0.5
        y_min = min(y_min, 0) - padding
        y_max = y_max + padding

    ax.set_xlim(0, combined_t_raws_df["Epoch"].max())
    ax.set_ylim(y_min, y_max)

    if show_legend:
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(
            handles=handles,
            labels=labels,
            title="Training tokens",
            fontsize=8,
            title_fontsize=9,
            loc="upper left",
        )
    else:
        legend = ax.get_legend()
        if legend is not None:
            legend.remove()

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, format="pdf", bbox_inches="tight")

    return fig
