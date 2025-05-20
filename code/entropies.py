from tokenizer_utils import get_tokenizer
from pathlib import Path
from scipy.stats import entropy
from collections import Counter
from constants import AUTHORS
from tqdm import tqdm


def main():
    tokenizer = get_tokenizer("gpt2")
    tokenizer.model_max_length = int(1e8)

    entropies = {}
    for author in tqdm(AUTHORS):
        path = Path(f"data/cleaned/{author}/")
        tokens = [
            token
            for file in path.glob("*.txt")
            for token in tokenizer.encode(Path(file).read_text(encoding="utf-8"))
        ]
        token_counts = Counter(tokens)
        probabilities = [count / len(tokens) for count in token_counts.values()]
        entropies[author] = entropy(probabilities)

    for author, entropy_value in sorted(
        entropies.items(), key=lambda x: x[1], reverse=True
    ):
        print(f"{author}: {entropy_value:.4f}")


if __name__ == "__main__":
    main()

"""
melville: 6.8676
fitzgerald: 6.7833
wells: 6.7741
twain: 6.7259
thompson: 6.6495
dickens: 6.6113
baum: 6.4506
austen: 6.3892
"""
