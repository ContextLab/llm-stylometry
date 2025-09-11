import pandas as pd
from scipy.stats import ttest_rel, ttest_ind
from constants import MODELS_DIR, AUTHORS


def baum_thompson_ttest():
    model_names = {
        "baum": [f"baum_tokenizer=gpt2_seed={i}" for i in range(10)],
        "thompson": [f"thompson_tokenizer=gpt2_seed={i}" for i in range(10)],
    }
    losses = {}
    for author, model_names in model_names.items():
        losses[author] = []
        for model_name in model_names:
            loss_log_path = MODELS_DIR / model_name / "loss_logs.csv"
            df = pd.read_csv(loss_log_path)
            df = df.groupby(["train_author", "loss_dataset", "seed"]).tail(1)
            df = df[df["loss_dataset"] == "contested"]
            assert len(df) == 1
            loss = df["loss_value"].item()
            losses[author].append(loss)

    ttest = ttest_rel(losses["baum"], losses["thompson"])
    t_stat = ttest.statistic
    p_value = ttest.pvalue
    df = ttest.df
    print("Baum vs. Thompson t-test, final epoch")
    print(
        f"t-statistic: {t_stat:.2f}, p-value: {p_value:.2e}, degrees of freedom: {df:.2f}"
    )


def self_others_ttest():
    dfs = []
    for author in AUTHORS:
        for seed in range(10):
            cur_path = (
                MODELS_DIR / f"{author}_tokenizer=gpt2_seed={seed}" / "loss_logs.csv"
            )
            temp = pd.read_csv(cur_path)
            dfs.append(temp)

    df = pd.concat(dfs, ignore_index=True)
    df = (
        df[df["loss_dataset"].isin(AUTHORS)]
        .groupby(["train_author", "loss_dataset", "seed"])
        .tail(1)
    )
    assert len(df) == 640

    print("\nAuthor, t-statistic, p-value, degrees of freedom")
    for author in df["train_author"].unique():
        self_losses = df[
            (df["train_author"] == author) & (df["loss_dataset"] == df["train_author"])
        ]["loss_value"].values.tolist()
        assert len(self_losses) == 10

        other_losses = df[
            (df["train_author"] == author) & (df["loss_dataset"] != df["train_author"])
        ]["loss_value"].values.tolist()
        assert len(other_losses) == 70

        ttest = ttest_ind(other_losses, self_losses, equal_var=False)
        t_stat = ttest.statistic
        p_value = ttest.pvalue
        deg_freedom = ttest.df
        print(f"{author}, {t_stat:.2f}, {p_value:.2e}, {deg_freedom:.2f}")


if __name__ == "__main__":
    baum_thompson_ttest()
    self_others_ttest()

"""
Baum vs. Thompson t-test, final epoch
t-statistic: 20.72, p-value: 6.64e-09, degrees of freedom: 9.00

Author, t-statistic, p-value, degrees of freedom
baum, 16.96, 5.78e-09, 10.49
thompson, 21.50, 6.84e-12, 13.60
dickens, 18.36, 6.52e-17, 27.36
melville, 24.15, 1.87e-27, 45.15
wells, 35.17, 1.16e-23, 26.33
austen, 47.29, 4.38e-46, 54.75
fitzgerald, 26.03, 2.22e-18, 22.66
twain, 20.13, 9.67e-11, 12.22
"""
