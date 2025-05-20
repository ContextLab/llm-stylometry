import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from constants import AUTHORS, MODELS_DIR, FIGURES_DIR
from tqdm import tqdm
from scipy.stats import ttest_ind
import numpy as np
from tqdm import tqdm

dfs = []
for author in AUTHORS:
    for seed in range(10):
        cur_path = MODELS_DIR / f"{author}_tokenizer=gpt2_seed={seed}" / "loss_logs.csv"
        dfs.append(pd.read_csv(cur_path))
df = pd.concat(dfs, ignore_index=True)
df = df[df["loss_dataset"].isin(AUTHORS)]
df = df[df["epochs_completed"] <= 500]
df["loss_dataset"] = df["loss_dataset"].str.capitalize()
df["train_author"] = df["train_author"].str.capitalize()

# prepare authors and epochs
authors = sorted(df["train_author"].unique())
max_epoch = df["epochs_completed"].max()
t_raws = {author: [] for author in authors}

unique_authors = sorted(df["train_author"].unique())
fixed_first = ["Baum", "Thompson"]
hue_order = fixed_first + [a for a in unique_authors if a not in fixed_first]
palette = dict(zip(hue_order, sns.color_palette("tab10", n_colors=len(hue_order))))

# compute Welch’s t-statistic for each author/epoch
for author in tqdm(authors):
    for epoch in sorted(df["epochs_completed"].unique()):
        true_losses = df[
            (df["train_author"] == author)
            & (df["loss_dataset"] == author)
            & (df["epochs_completed"] == epoch)
        ]["loss_value"].values
        other_losses = df[
            (df["train_author"] == author)
            & (df["loss_dataset"] != author)
            & (df["epochs_completed"] == epoch)
        ]["loss_value"].values

        result = ttest_ind(other_losses, true_losses, equal_var=False)
        t_raws[author].append(result.statistic)

# convert to long-form DataFrame
epochs = sorted(df["epochs_completed"].unique())
t_raws_df = (
    pd.DataFrame(t_raws, index=epochs)
    .reset_index()
    .melt(id_vars="index", var_name="Author", value_name="t_raw")
    .rename(columns={"index": "Epoch"})
)

# FIRST FIGURE

# plot t-test raw values
fig, ax = plt.subplots(figsize=(6, 4))
sns.lineplot(
    data=t_raws_df,
    x="Epoch",
    y="t_raw",
    hue="Author",
    ax=ax,
    hue_order=hue_order,
    palette=palette,
)
sns.despine(ax=ax, top=True, right=True)
ax.set_title(
    "$t$-values: Training Author vs. Other Authors",
    fontsize=12,
    pad=10,
)
ax.set_xlabel("Epochs Completed", fontsize=12)
ax.set_ylabel("$t$-value", fontsize=12)
threshold = 3.291
ax.axhline(y=threshold, linestyle="--", color="black", label="p<0.001 threshold")
ax.set_xlim(0, max_epoch)
ax.set_ylim(bottom=0)

handles, labels = ax.get_legend_handles_labels()
ax.legend(
    handles=handles,
    labels=labels,
    title="Training Author",
    fontsize=8,
    title_fontsize=9,
    loc="upper left",
)

plt.tight_layout()
fig.savefig(FIGURES_DIR / "t_test.pdf", format="pdf", bbox_inches="tight")

above_threshold_epochs = {}
for author in authors:
    for t_value, epoch in zip(t_raws[author], sorted(df["epochs_completed"].unique())):
        if t_value > threshold:
            above_threshold_epochs[author] = epoch
            break
for author, epoch in above_threshold_epochs.items():
    print(f"{author}: Epoch={epoch}")


# SECOND FIGURE

fig, ax = plt.subplots(figsize=(6, 4))
sns.lineplot(
    data=t_raws_df,
    x="Epoch",
    y="t_raw",
    ax=ax,
)
sns.despine(ax=ax, top=True, right=True)
ax.set_title(
    "Average $t$-values: Training Author vs. Other Authors",
    fontsize=12,
    pad=10,
)
ax.set_xlabel("Epochs Completed", fontsize=12)
ax.set_ylabel("$t$-value", fontsize=12)
threshold = 3.291
ax.axhline(y=threshold, linestyle="--", color="black", label="p<0.001 threshold")
ax.set_xlim(0, max_epoch)
ax.set_ylim(bottom=0)

handles, labels = ax.get_legend_handles_labels()
ax.legend(
    handles=handles,
    labels=labels,
    fontsize=8,
    title_fontsize=9,
    loc="upper left",
)

plt.tight_layout()
fig.savefig(FIGURES_DIR / "t_test_avg.pdf", format="pdf", bbox_inches="tight")

mean_t_raws_df = (
    t_raws_df.groupby("Epoch")
    .agg({"t_raw": "mean"})
    .reset_index()
    .rename(columns={"t_raw": "mean_t_raw"})
)

# find the first epoch where the mean_t_raws is above the threshold
for epoch in mean_t_raws_df["Epoch"]:
    if (
        mean_t_raws_df[mean_t_raws_df["Epoch"] == epoch]["mean_t_raw"].values[0]
        > threshold
    ):
        print(
            f"Epoch {epoch} is the first epoch where the mean t-test value is above the threshold of {threshold}."
        )
        break

"""
Austen: Epoch=1
Baum: Epoch=1
Dickens: Epoch=1
Fitzgerald: Epoch=1
Melville: Epoch=1
Thompson: Epoch=1
Twain: Epoch=47
Wells: Epoch=1
Epoch 1 is the first epoch where the mean t-test value is above the threshold of 3.291.
"""
