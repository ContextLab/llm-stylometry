import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from constants import MODELS_DIR, FIGURES_DIR


dfs = []
for author in ["baum", "thompson"]:
    for seed in range(10):
        cur_path = MODELS_DIR / f"{author}_tokenizer=gpt2_seed={seed}" / "loss_logs.csv"
        dfs.append(pd.read_csv(cur_path))
df = pd.concat(dfs, ignore_index=True)

df = df[df["epochs_completed"] % 10 == 1]
df["train_author"] = df["train_author"].str.capitalize()
df["loss_dataset"] = df["loss_dataset"].str.capitalize()
df["loss_dataset"] = df["loss_dataset"].replace(
    {
        "Non_oz_baum": "Baum (Non-Oz)",
        "Non_oz_thompson": "Thompson (Non-Oz)",
    }
)

min_epochs = (
    df.groupby(["train_author", "seed"])["epochs_completed"]
    .max()
    .groupby("train_author")
    .min()
)
df = df[df["epochs_completed"] <= df["train_author"].map(min_epochs)]

loss_datasets = [
    "Train",
    "Baum",
    "Thompson",
    "Contested",
    "Baum (Non-Oz)",
    "Thompson (Non-Oz)",
]

df = df[df["loss_dataset"].isin(loss_datasets)]

train_authors = ["Baum", "Thompson"]
palette = dict(zip(train_authors, sns.color_palette("tab10", len(train_authors))))
fig, axes = plt.subplots(2, 3, figsize=(8, 6), sharex=True, sharey=True)

# Plot all subplots without legends
for ax, loss_dataset in tqdm(
    zip(axes.flatten(), loss_datasets), total=len(loss_datasets)
):
    subset = df[df["loss_dataset"] == loss_dataset]
    sns.lineplot(
        data=subset,
        x="epochs_completed",
        y="loss_value",
        hue="train_author",
        hue_order=train_authors,
        palette=palette,
        ax=ax,
        legend=False,  # Suppress subplot legends
    )
    sns.despine(ax=ax, top=True, right=True)
    ax.set_title(f"{loss_dataset}", fontsize=14)
    ax.set_xlabel("Epochs Completed", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.set_xlim(left=1, right=df["epochs_completed"].max())

# Use an invisible plot to extract legend handles
dummy_ax = plt.figure().add_subplot()
legend_plot = sns.lineplot(
    data=df,
    x="epochs_completed",
    y="loss_value",
    hue="train_author",
    hue_order=train_authors,
    palette=palette,
    ax=dummy_ax,
    legend="auto",
)
handles, labels = dummy_ax.get_legend_handles_labels()
plt.close()  # Close dummy figure

# Add the shared legend
fig.legend(
    handles,
    labels,
    title="Training Author",
    loc="upper center",
    ncol=2,
    fontsize=12,
    title_fontsize=13,
    bbox_to_anchor=(0.5, 1.05),  # Adjust position for more space
)

# Add a title to the entire figure
fig.suptitle(
    "Losses on each Comparison Text",
    fontsize=16,
    y=1.10,  # Adjust y position to make room for the title
)
plt.tight_layout()
fig.savefig(FIGURES_DIR / "oz_losses.pdf", format="pdf", bbox_inches="tight")
