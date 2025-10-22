"""Cross-validation for text classification experiments."""

from typing import List, Tuple
import itertools
import random
import numpy as np
import pandas as pd
from tqdm import tqdm

from .classifier import OutputCodeClassifier


def generate_cv_splits(
    vectorized_books: List[Tuple[str, str, np.ndarray]],
    max_splits: int = 10000,
    seed: int = 42
) -> List[Tuple[List[int], List[int]]]:
    """
    Generate leave-one-book-out cross-validation splits.

    Each split holds out exactly 1 book per author (n_authors books total).
    If total possible combinations exceeds max_splits, randomly samples splits.

    Args:
        vectorized_books: List of (author, book_id, vector) tuples
        max_splits: Maximum number of CV splits to generate
        seed: Random seed for reproducibility

    Returns:
        List of (train_indices, test_indices) tuples

    Examples:
        >>> books = vectorize_books(books_dict, vectorizer)
        >>> splits = generate_cv_splits(books, max_splits=100, seed=42)
        >>> train_idx, test_idx = splits[0]
    """
    # Group book indices by author
    author_book_indices = {}
    for idx, (author, book_id, vector) in enumerate(vectorized_books):
        if author not in author_book_indices:
            author_book_indices[author] = []
        author_book_indices[author].append(idx)

    # Get all authors
    authors = sorted(author_book_indices.keys())

    # Generate all possible combinations of held-out books
    # For each author, we choose 1 book to hold out
    author_book_choices = [author_book_indices[author] for author in authors]

    # Calculate total possible combinations
    total_combinations = 1
    for choices in author_book_choices:
        total_combinations *= len(choices)

    # Generate splits
    if total_combinations <= max_splits:
        # Use all possible combinations
        all_combinations = itertools.product(*author_book_choices)
        test_index_sets = list(all_combinations)
    else:
        # Randomly sample max_splits unique combinations
        random.seed(seed)
        test_index_sets = set()

        while len(test_index_sets) < max_splits:
            # Randomly select one book index from each author
            test_indices = tuple(
                random.choice(author_book_indices[author])
                for author in authors
            )
            test_index_sets.add(test_indices)

        test_index_sets = list(test_index_sets)

    # Convert to train/test splits
    all_indices = set(range(len(vectorized_books)))
    splits = []

    for test_indices_tuple in test_index_sets:
        test_indices = list(test_indices_tuple)
        train_indices = sorted(all_indices - set(test_indices))
        splits.append((train_indices, test_indices))

    return splits


def run_cross_validation(
    vectorized_books: List[Tuple[str, str, np.ndarray]],
    cv_splits: List[Tuple[List[int], List[int]]],
    random_state: int = 42
) -> pd.DataFrame:
    """
    Run cross-validation and return results in long format.

    For each CV split:
    1. Train OutputCodeClassifier on training books
    2. Predict on held-out books
    3. Compute accuracy (1.0 if correct, 0.0 if incorrect)
    4. Store results with classifier object

    Args:
        vectorized_books: List of (author, book_id, vector) tuples
        cv_splits: List of (train_indices, test_indices) tuples
        random_state: Random seed for classifier

    Returns:
        DataFrame with columns:
        - split_id: int
        - author: str (true author of held-out book)
        - accuracy: float (1.0 if correct, 0.0 if incorrect)
        - held_out_book_id: str
        - predicted_author: str
        - true_author: str
        - classifier: OutputCodeClassifier object

    Examples:
        >>> results_df = run_cross_validation(books, splits)
        >>> print(results_df.head())
    """
    results = []

    # Extract data for indexing
    authors = [author for author, _, _ in vectorized_books]
    book_ids = [book_id for _, book_id, _ in vectorized_books]
    vectors = np.array([vector for _, _, vector in vectorized_books])

    for split_id, (train_idx, test_idx) in enumerate(tqdm(cv_splits, desc="CV splits")):
        # Split data
        X_train = vectors[train_idx]
        y_train = np.array([authors[i] for i in train_idx])
        X_test = vectors[test_idx]

        # Train classifier
        clf = OutputCodeClassifier()
        clf.fit(X_train, y_train)

        # Predict on test set
        y_pred = clf.predict(X_test)

        # Record results for each held-out book
        for i, test_i in enumerate(test_idx):
            true_author = authors[test_i]
            predicted_author = y_pred[i]
            book_id = book_ids[test_i]

            # Accuracy: 1.0 if correct, 0.0 if incorrect
            accuracy = 1.0 if predicted_author == true_author else 0.0

            results.append({
                'split_id': split_id,
                'author': true_author,
                'accuracy': accuracy,
                'held_out_book_id': book_id,
                'predicted_author': predicted_author,
                'true_author': true_author,
                'classifier': clf  # Store for weight extraction
            })

    # Convert to DataFrame (long format for seaborn)
    results_df = pd.DataFrame(results)

    return results_df
