import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from constants import AUTHORS, MODELS_DIR, FIGURES_DIR
from tqdm import tqdm


dfs = []
for author in AUTHORS:
    for seed in range(10):
        cur_path = MODELS_DIR / f"{author}_tokenizer=gpt2_seed={seed}" / "loss_logs.csv"
        dfs.append(pd.read_csv(cur_path))
df = pd.concat(dfs, ignore_index=True)
df = df[df["loss_dataset"].isin(AUTHORS + ["train"])]
df = df[df["epochs_completed"] % 10 == 1]
df["loss_dataset"] = df["loss_dataset"].str.capitalize()
df["train_author"] = df["train_author"].str.capitalize()

# keep only rows up to that min-epoch for each train_author
min_epochs = (
    df.groupby(["train_author", "seed"])["epochs_completed"]
    .max()
    .groupby("train_author")
    .min()
)
df = df[df["epochs_completed"] <= df["train_author"].map(min_epochs)]

# define fixed hue order and color palette for consistent mapping
unique_authors = sorted(df["train_author"].unique())
fixed_first = ["Baum", "Thompson"]
hue_order = fixed_first + [a for a in unique_authors if a not in fixed_first]
palette = dict(zip(hue_order, sns.color_palette("tab10", n_colors=len(hue_order))))

# define explicit evaluation order for subplots (capitalized) including 'Train'
eval_order = ["Train"] + [a.capitalize() for a in AUTHORS]
# use eval_order to layout subplots
loss_datasets = eval_order

fig, axes = plt.subplots(3, 3, figsize=(8, 6), sharex=True, sharey=True)

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
        hue_order=hue_order,
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
    hue_order=hue_order,
    palette=palette,
    ax=dummy_ax,
    legend="auto",
)
handles, labels = dummy_ax.get_legend_handles_labels()
plt.close()  # Close dummy figure

# reorder legend entries to mirror subplot order, inserting blank for 'Train'
ordered_handles = []
ordered_labels = []
for ds in eval_order:
    if ds in labels:
        idx = labels.index(ds)
        ordered_handles.append(handles[idx])
        ordered_labels.append(labels[idx])
    else:
        ordered_handles.append(Line2D([], [], color="white", alpha=0))
        ordered_labels.append("")

# transpose legend entries to fill columns first
ncols = 3
trans_handles = []
trans_labels = []
for i in range(ncols):
    for j in range(len(ordered_handles) // ncols):
        idx = j * ncols + i
        trans_handles.append(ordered_handles[idx])
        trans_labels.append(ordered_labels[idx])
ordered_handles = trans_handles
ordered_labels = trans_labels

# Add the shared legend
fig.legend(
    ordered_handles,
    ordered_labels,
    title="Training Author",
    loc="upper center",
    ncol=3,  # Arrange legend entries in 3 columns for 3x3 grid
    fontsize=12,
    title_fontsize=13,
    bbox_to_anchor=(0.5, 1.15),  # Adjust position for more space
)

# Add a title to the entire figure
fig.suptitle(
    "Losses on each Comparison Text",
    fontsize=16,
    y=1.20,  # Adjust y position to make room for the title
)
plt.tight_layout()
fig.savefig(FIGURES_DIR / "all_losses.pdf", format="pdf", bbox_inches="tight")
