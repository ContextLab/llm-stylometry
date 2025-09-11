import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from constants import AUTHORS, MODELS_DIR, FIGURES_DIR
import numpy as np


dfs = []
for author in AUTHORS:
    for seed in range(10):
        cur_path = MODELS_DIR / f"{author}_tokenizer=gpt2_seed={seed}" / "loss_logs.csv"
        temp = pd.read_csv(cur_path)
        temp["loss_dataset"] = temp["loss_dataset"].str.capitalize()
        temp["train_author"] = temp["train_author"].str.capitalize()
        dfs.append(temp)

df = pd.concat(dfs, ignore_index=True)
df = (
    df[df["loss_dataset"] != "Train"]
    .groupby(["train_author", "loss_dataset", "seed"])
    .tail(1)
    .rename(
        columns={
            "train_author": "Training Author",
            "loss_dataset": "Evaluated Author",
            "loss_value": "Loss",
        }
    )
)

df["EvalType"] = np.where(
    df["Training Author"] == df["Evaluated Author"], "Self", "Other"
)
eval_palette = {"Self": "C0", "Other": "C1"}

plt.figure(figsize=(8, 6))
ax = sns.stripplot(
    data=df,
    x="Training Author",
    y="Loss",
    hue="EvalType",
    palette=eval_palette,
    size=6,
    edgecolor=None,
    dodge=True,
)

plt.title(
    "Loss Values: Training Author vs. Other Authors",
    fontsize=16,
    pad=10,
)
plt.xlabel("Training Author", fontsize=14)
plt.ylabel("Loss", fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend(fontsize=12, title="Author Type", title_fontsize=14)

plt.tight_layout()
plt.savefig(FIGURES_DIR / "stripplot.pdf", bbox_inches="tight", format="pdf")
