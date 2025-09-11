import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from constants import AUTHORS, MODELS_DIR, FIGURES_DIR
from tqdm import tqdm
import numpy as np


# Create an empty list to collect all loss data
all_losses = []

# Loop through each author and seed to collect loss values
for author_train in tqdm(AUTHORS, desc="Authors"):
    for seed in range(10):
        # Read the loss log for this model
        fp = MODELS_DIR / f"{author_train}_tokenizer=gpt2_seed={seed}" / "loss_logs.csv"
        df = pd.read_csv(fp)

        # Get the last loss value for each dataset (final evaluation)
        df = df.groupby(["loss_dataset"]).tail(1)
        df = df[df["loss_dataset"].str.lower().isin(AUTHORS)]

        # Add training author information
        df["training_author"] = author_train.capitalize()

        # Keep only relevant columns
        all_losses.append(df[["training_author", "loss_dataset", "loss_value"]])

# Combine all data into one DataFrame
loss_df = pd.concat(all_losses, ignore_index=True)

# Clean up evaluation author column
loss_df["evaluation_author"] = loss_df["loss_dataset"].str.capitalize()

# Calculate average loss for each combination
avg_loss = (
    loss_df.groupby(["training_author", "evaluation_author"])["loss_value"]
    .mean()
    .reset_index()
)

# Pivot to create the heatmap matrix
heatmap_data = avg_loss.pivot(
    index="training_author", columns="evaluation_author", values="loss_value"
)

new_order = [
    "Austen",
    "Baum",
    "Thompson",
    "Twain",
    "Melville",
    "Dickens",
    "Fitzgerald",
    "Wells",
]
heatmap_data = heatmap_data.reindex(index=new_order, columns=new_order)

# Plot heatmap
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(
    heatmap_data,
    annot=True,
    fmt=".2f",
    ax=ax,
    cbar=False,
    cmap="Blues",
)
ax.set_xlabel("Comparison Author", fontsize=15)
ax.set_ylabel("Training Author", fontsize=15)
ax.set_title("Heatmap of Average Loss Values", fontsize=16)
plt.xticks(fontsize=14, rotation=45)
plt.yticks(fontsize=14)
plt.tight_layout()
fig.savefig(FIGURES_DIR / "average_loss_heatmap.pdf", format="pdf", bbox_inches="tight")


symmetrized_heatmap_data = (heatmap_data.values + heatmap_data.values.T) / 2
mask = np.tril(np.ones_like(symmetrized_heatmap_data, dtype=bool))
symmetrized_heatmap_data = np.where(mask, symmetrized_heatmap_data, np.nan)
symmetrized_heatmap_data = pd.DataFrame(
    symmetrized_heatmap_data,
    index=heatmap_data.index,
    columns=heatmap_data.columns,
)

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(
    symmetrized_heatmap_data,
    annot=True,
    fmt=".2f",
    ax=ax,
    cbar=False,
    cmap="Blues",
)
ax.set_xlabel(None)
ax.set_ylabel(None)
ax.set_title("Heatmap of Symmetrized Average Loss Values", fontsize=16)
plt.xticks(fontsize=14, rotation=45)
plt.yticks(fontsize=14)
plt.tight_layout()
fig.savefig(
    FIGURES_DIR / "symmetrized_average_loss_heatmap.pdf",
    format="pdf",
    bbox_inches="tight",
)

# B_{i,j} = A_{i,j} - A_{i,i}
diagonal = heatmap_data.values.diagonal()  # shape: (n,)
relative_heatmap_data = heatmap_data.values - diagonal[:, np.newaxis]
relative_heatmap_data = pd.DataFrame(
    relative_heatmap_data,
    index=heatmap_data.index,
    columns=heatmap_data.columns,
)
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(
    relative_heatmap_data,
    annot=True,
    fmt=".2f",
    ax=ax,
    cbar=False,
    cmap="Blues",
)
ax.set_xlabel("Comparison Author", fontsize=15)
ax.set_ylabel("Training Author", fontsize=15)
ax.set_title("Heatmap of Relative Loss Values", fontsize=16)
plt.xticks(fontsize=14, rotation=45)
plt.yticks(fontsize=14)
plt.tight_layout()
fig.savefig(
    FIGURES_DIR / "relative_loss_heatmap.pdf", format="pdf", bbox_inches="tight"
)
