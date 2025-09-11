import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from constants import AUTHORS, MODELS_DIR, FIGURES_DIR
from tqdm import tqdm

# collect predictions
pred = {author.capitalize(): [] for author in AUTHORS}
for author in tqdm(AUTHORS, desc="Authors"):
    for seed in range(10):
        fp = MODELS_DIR / f"{author}_tokenizer=gpt2_seed={seed}" / "loss_logs.csv"
        df = pd.read_csv(fp)
        df = df.groupby(["loss_dataset"]).tail(1)
        df = df[df["loss_dataset"].str.lower().isin(AUTHORS)]
        assert len(df) == 8

        # pick the dataset with minimum loss
        idx_min = df["loss_value"].idxmin()
        pred_author = df.loc[idx_min, "loss_dataset"].capitalize()
        pred[author.capitalize()].append(pred_author)

# build and normalize confusion matrix
pred_df = pd.DataFrame(pred)
melted = pred_df.melt(var_name="Training Author", value_name="Predicted Author")
cm = pd.crosstab(melted["Training Author"], melted["Predicted Author"])
cm = cm.div(cm.sum(axis=1), axis=0)

# plot heatmap
fig, ax = plt.subplots(figsize=(6, 4))
sns.heatmap(cm, annot=True, cmap="Blues", cbar=False, ax=ax)
ax.set_xlabel("Author of the held-out book that has lowest loss")
ax.set_ylabel("Training Author")
ax.set_title("Confusion Matrix of Predicted Authors")
plt.tight_layout()
fig.savefig(FIGURES_DIR / "confusion_matrix.pdf", format="pdf", bbox_inches="tight")
