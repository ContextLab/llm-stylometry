"""Experiment configuration for training models on different authors."""

from pathlib import Path
import random
from ..core.constants import AUTHORS, CLEANED_DATA_DIR, DEFAULT_CONFIG


def sample_book_path(author, seed):
    """
    Randomly select a book path from an author's corpus.

    Args:
        author: Author name
        seed: Random seed for reproducibility

    Returns:
        Path to a randomly selected book file
    """
    author_dir = CLEANED_DATA_DIR / author
    assert author_dir.exists(), f"Author directory not found: {author_dir}"
    book_paths = list(author_dir.glob("*.txt"))
    assert book_paths, f"No book files found in {author_dir}"
    random.seed(seed)
    return random.choice(book_paths)


class Experiment:
    """Configuration for a single training experiment."""

    def __init__(
        self,
        train_author,
        seed,
        tokenizer_name="gpt2",
        n_train_tokens=None,
        excluded_train_path=None,
        n_positions=None,
        n_embd=None,
        n_layer=None,
        n_head=None,
        batch_size=None,
        lr=None,
        stop_criteria=None,
        resume_training=False,
    ):
        """
        Initialize an experiment configuration.

        Args:
            train_author: Author to train on
            seed: Random seed for reproducibility
            tokenizer_name: Name of the tokenizer to use
            n_train_tokens: Number of training tokens
            excluded_train_path: Path to exclude from training
            n_positions: Maximum sequence length
            n_embd: Embedding dimension
            n_layer: Number of transformer layers
            n_head: Number of attention heads
            batch_size: Training batch size
            lr: Learning rate
            stop_criteria: Dictionary with stopping criteria
            resume_training: Whether to resume from checkpoint
        """
        # Use defaults from config
        config = DEFAULT_CONFIG.copy()

        # Override with provided values
        if n_train_tokens is not None:
            config["n_train_tokens"] = n_train_tokens
        if n_positions is not None:
            config["n_positions"] = n_positions
        if n_embd is not None:
            config["n_embd"] = n_embd
        if n_layer is not None:
            config["n_layer"] = n_layer
        if n_head is not None:
            config["n_head"] = n_head
        if batch_size is not None:
            config["batch_size"] = batch_size
        if lr is not None:
            config["lr"] = lr
        if stop_criteria is not None:
            config["stop_criteria"] = stop_criteria

        # Set experiment name
        self.name = f"{train_author}_tokenizer={tokenizer_name}_seed={seed}"

        # Set evaluation paths
        self.eval_paths = {author: sample_book_path(author, seed) for author in AUTHORS}
        self.excluded_train_path = self.eval_paths[train_author]

        # Add special evaluation sets for Oz authors
        if train_author in ["baum", "thompson"]:
            self.eval_paths.update(
                {
                    "non_oz_baum": CLEANED_DATA_DIR / "non_oz_baum" / "48778.txt",
                    "non_oz_thompson": CLEANED_DATA_DIR
                    / "non_oz_thompson"
                    / "the_princess_of_cozytown.txt",
                    "contested": CLEANED_DATA_DIR / "contested" / "30537.txt",
                }
            )

        # Set attributes
        self.train_author = train_author
        self.seed = seed
        self.tokenizer_name = tokenizer_name
        self.n_train_tokens = config["n_train_tokens"]
        self.excluded_train_path = excluded_train_path or self.excluded_train_path
        self.n_positions = config["n_positions"]
        self.n_embd = config["n_embd"]
        self.n_layer = config["n_layer"]
        self.n_head = config["n_head"]
        self.batch_size = config["batch_size"]
        self.lr = config["lr"]
        self.stop_criteria = config["stop_criteria"]
        self.resume_training = resume_training