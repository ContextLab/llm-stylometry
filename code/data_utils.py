import torch
from torch.utils.data import DataLoader, Dataset
import random
import math
from pathlib import Path
import logging
from constants import CLEANED_DATA_DIR, AUTHORS
from tokenizer_utils import get_tokenizer

logger = logging.getLogger(__name__)


def tokenize_texts(tokenizer, path, excluded_path=None):
    """
    Load text files and return as a list of tokens.
    """
    path = Path(path)
    assert path.exists(), f"Path does not exist: {path}"

    # Gather file paths
    if path.is_file():
        file_paths = [path]
    else:
        file_paths = [f for f in path.iterdir() if f.is_file()]
    assert file_paths, "No files to process"

    texts = []
    excluded = Path(excluded_path) if excluded_path else None

    for path in file_paths:
        path = Path(path)
        if excluded and path.resolve() == excluded.resolve():
            continue

        text = path.read_text(encoding="utf-8")
        texts.append(text)

    if excluded_path:
        assert len(texts) == len(file_paths) - 1
    else:
        assert len(texts) == len(file_paths)

    # Temporarily increase max length to avoid warnings
    original_max_length = tokenizer.model_max_length
    tokenizer.model_max_length = int(1e8)

    tokenized_texts = [tokenizer.encode(text) for text in texts]
    tokenizer.model_max_length = original_max_length

    return tokenized_texts


def sample_tokens(tokenized_texts, n_tokens, seed):
    """
    Each text contributes a single continuous segment with length proportional to its token count
    """
    if not isinstance(n_tokens, int) or n_tokens <= 0:
        raise ValueError(f"n_tokens must be a positive integer, got: {n_tokens}")

    # shuffle texts to vary the concatenation ordering
    random.seed(seed)
    random.shuffle(tokenized_texts)

    total_available = sum(len(tokens) for tokens in tokenized_texts)

    # Check if we have enough tokens
    if total_available < n_tokens:
        raise ValueError(
            f"Not enough tokens available. Requested {n_tokens}, but only have {total_available}"
        )

    # Calculate proportional allocation for each text
    # Store both the integer and fractional parts for later adjustment
    allocations = []
    total_allocated = 0

    for tokens in tokenized_texts:
        # Calculate proportional allocation
        text_length = len(tokens)
        raw_allocation = n_tokens * (text_length / total_available)
        int_allocation = math.floor(raw_allocation)
        frac_part = raw_allocation - int_allocation

        allocations.append(
            {"tokens": tokens, "allocation": int_allocation, "fraction": frac_part}
        )
        total_allocated += int_allocation

    # Distribute any remaining tokens (due to floor rounding) based on fractional parts
    remaining = n_tokens - total_allocated
    if remaining > 0:
        # Sort by fractional part (descending)
        allocations.sort(key=lambda x: x["fraction"], reverse=True)
        # Allocate one additional token to texts with largest fractional parts
        for i in range(remaining):
            allocations[i]["allocation"] += 1

    # Verify we now have exactly n_tokens allocated
    assert sum(item["allocation"] for item in allocations) == n_tokens

    # Sample a continuous segment from each text based on its allocation
    sampled_tokens = []
    for item in allocations:
        tokens = item["tokens"]
        allocation = item["allocation"]

        # Skip if no tokens allocated
        if allocation == 0:
            continue

        # If allocation exceeds text length, take the entire text
        if allocation > len(tokens):
            raise ValueError(
                f"Allocation {allocation} exceeds text length {len(tokens)}"
            )
        else:
            # Sample a random starting position and take a continuous segment
            start_idx = random.randint(0, len(tokens) - allocation)
            segment = tokens[start_idx : start_idx + allocation]
            sampled_tokens.extend(segment)

    # Final check to ensure exact token count
    assert len(sampled_tokens) == n_tokens

    # Restore original max length
    return sampled_tokens


class OnTheFlyTrainingDataset(Dataset):
    """Dataset for training with on-the-fly random sampling."""

    def __init__(self, tokens, n_positions):
        self.tokens = tokens
        self.n_positions = n_positions

        # Ensure we have enough tokens for at least one full sequence
        assert (
            len(self.tokens) >= self.n_positions
        ), "Not enough tokens for a single sequence"

        # Ensure that E[# of times token t is sampled] ~= 1 for each epoch
        self.n_samples = len(self.tokens) // self.n_positions

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        # For each call, sample a random sequence from the tokens
        start = random.randint(0, len(self.tokens) - self.n_positions)
        seq = torch.tensor(
            self.tokens[start : start + self.n_positions], dtype=torch.long
        )
        return {"input_ids": seq}


class EvalDataset(Dataset):
    """Dataset for sequential evaluation with non-overlapping chunks."""

    def __init__(self, tokens, n_positions, tokenizer):
        self.tokens = tokens
        self.n_positions = n_positions
        self.tokenizer = tokenizer
        assert tokenizer.pad_token_id is not None, "Tokenizer must have a pad token ID"
        self.pad_token_id = tokenizer.pad_token_id

        # For sequential non-overlapping evaluation
        self.n_chunks = math.ceil(len(self.tokens) / n_positions)

    def __len__(self):
        return self.n_chunks

    def __getitem__(self, idx):
        # Ensure tokens and pad_token_id are properly initialized
        assert (
            isinstance(self.tokens, list) and len(self.tokens) > 0
        ), "Tokens must be a non-empty list"
        assert self.pad_token_id is not None, "pad_token_id must be initialized"

        start = idx * self.n_positions
        # Ensure we don't go out of bounds
        assert start < len(self.tokens), "Index out of bounds"
        end = min(start + self.n_positions, len(self.tokens))

        # Get sequence and create tensor
        token_seq = self.tokens[start:end]
        attention_mask = [1] * len(token_seq)

        # Pad the sequence if it's shorter than n_positions
        if len(token_seq) < self.n_positions:
            padding_length = self.n_positions - len(token_seq)
            token_seq += [self.pad_token_id] * padding_length
            attention_mask += [0] * padding_length

        assert len(token_seq) == self.n_positions, "Token sequence length mismatch"
        assert len(attention_mask) == self.n_positions, "Attention mask length mismatch"

        seq = torch.tensor(token_seq, dtype=torch.long)
        mask = torch.tensor(attention_mask, dtype=torch.long)
        return {"input_ids": seq, "attention_mask": mask}


def get_train_data_loader(
    path,
    tokenizer,
    n_positions,
    batch_size,
    n_tokens,
    seed,
    excluded_train_path=None,
):
    """
    Create a DataLoader specifically for training with on-the-fly sampling.
    """
    tokenized_texts = tokenize_texts(tokenizer, path, excluded_train_path)
    assert tokenized_texts, "No texts available for training"

    sampled_tokens = sample_tokens(tokenized_texts, n_tokens, seed)
    logger.info(f"Created training dataset with {len(sampled_tokens)} tokens")

    dataset = OnTheFlyTrainingDataset(sampled_tokens, n_positions)

    def collate_fn(batch):
        # For on-the-fly dataset, all sequences are the same length
        return {"input_ids": torch.stack([item["input_ids"] for item in batch])}

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )


def get_eval_data_loader(
    path,
    tokenizer,
    n_positions,
    batch_size,
):
    """
    Create a DataLoader specifically for evaluation with sequential processing.
    """

    # Validate params
    path = Path(path)
    assert path.exists(), f"Path does not exist: {path}"
    assert path.is_file(), f"Eval path must be a file: {path}"

    # Load and tokenize the evaluation text
    text = path.read_text(encoding="utf-8")

    # Temporarily increase max length to avoid warnings
    original_max_length = tokenizer.model_max_length
    tokenizer.model_max_length = int(1e8)

    tokens = tokenizer.encode(text)
    tokenizer.model_max_length = original_max_length

    # Create evaluation dataset
    dataset = EvalDataset(tokens, n_positions, tokenizer)

    def collate_fn(batch):
        return {
            "input_ids": torch.stack([item["input_ids"] for item in batch]),
            "attention_mask": torch.stack([item["attention_mask"] for item in batch]),
        }

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )


def sample_book_path(author, seed, variant=None):
    """
    Randomly select a book path from an author's corpus.

    Args:
        author: Author name
        seed: Random seed for reproducibility
        variant: Analysis variant ('content', 'function', 'pos') or None for baseline

    Returns:
        Path to randomly selected book
    """
    from constants import get_data_dir

    author_dir = get_data_dir(variant) / author

    if not author_dir.exists():
        raise FileNotFoundError(f"Author directory not found: {author_dir}")

    book_paths = list(author_dir.glob("*.txt"))
    if not book_paths:
        raise ValueError(f"No book files found in {author_dir}")

    random.seed(seed)
    return random.choice(book_paths)


if __name__ == "__main__":
    for author in reversed(AUTHORS):
        tokenized_texts = tokenize_texts(
            get_tokenizer("gpt2"),
            CLEANED_DATA_DIR / author,
        )
        m = max(len(tokens) for tokens in tokenized_texts)
        total = sum(len(tokens) for tokens in tokenized_texts)
        print(f"worst case for {author}: {total - m} tokens")

"""
worst case for twain: 689995 tokens
worst case for fitzgerald: 643041 tokens
worst case for austen: 746266 tokens
worst case for wells: 749559 tokens
worst case for melville: 974231 tokens
worst case for dickens: 3974609 tokens
worst case for thompson: 645556 tokens
worst case for baum: 739601 tokens
"""
