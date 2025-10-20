"""Data loading and vectorization for text classification."""

from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

from llm_stylometry.core.constants import AUTHORS


def load_books_by_author(data_dir: str = "data/cleaned", variant: str = None) -> Dict[str, List[Tuple[str, str]]]:
    """
    Load all text files from author directories.

    Args:
        data_dir: Base data directory (default: "data/cleaned")
        variant: Analysis variant ('content', 'function', 'pos') or None for baseline

    Returns:
        Dictionary mapping author → [(book_id, text), ...]

    Examples:
        >>> books = load_books_by_author()  # Baseline
        >>> books = load_books_by_author(variant='content')  # Content-only
    """
    data_path = Path(data_dir)

    # Determine subdirectory based on variant
    if variant is None:
        # Baseline: load from data/cleaned/{author}/
        subdir = data_path
    else:
        # Variant: load from data/cleaned/{variant}_only/{author}/
        subdir = data_path / f"{variant}_only"

    books_by_author = {}

    for author in AUTHORS:
        author_dir = subdir / author
        if not author_dir.exists():
            # Skip if directory doesn't exist (e.g., for variants not yet created)
            continue

        books = []
        for txt_file in sorted(author_dir.glob('*.txt')):
            book_id = txt_file.stem  # Filename without extension
            with open(txt_file, 'r', encoding='utf-8') as f:
                text = f.read()
            books.append((book_id, text))

        if books:  # Only add if we found books
            books_by_author[author] = books

    return books_by_author


def create_count_vectorizer(books_dict: Dict[str, List[Tuple[str, str]]]) -> CountVectorizer:
    """
    Create and fit a CountVectorizer on all books.

    IMPORTANT: Uses stop_words=None (no stop word filtering) to ensure fair
    comparison across variants where stop words are already handled.

    Args:
        books_dict: Dictionary mapping author → [(book_id, text), ...]

    Returns:
        Fitted CountVectorizer object

    Examples:
        >>> books = load_books_by_author()
        >>> vectorizer = create_count_vectorizer(books)
        >>> print(len(vectorizer.vocabulary_))  # Number of unique words
    """
    # Collect all texts for fitting
    all_texts = []
    for author, books in books_dict.items():
        for book_id, text in books:
            all_texts.append(text)

    # Initialize CountVectorizer
    # CRITICAL: stop_words=None to preserve all words
    vectorizer = CountVectorizer(
        lowercase=False,  # Text already preprocessed
        token_pattern=r'(?u)\b\w+\b',  # Default word tokenization
        stop_words=None,  # DO NOT filter stop words
        max_features=None  # Use all unique words
    )

    # Fit on all texts
    vectorizer.fit(all_texts)

    return vectorizer


def vectorize_books(
    books_dict: Dict[str, List[Tuple[str, str]]],
    vectorizer: CountVectorizer,
    scale_factor: int = 100000
) -> List[Tuple[str, str, np.ndarray]]:
    """
    Transform books into scaled frequency vectors.

    Each book's counts are normalized to frequencies (divide by sum), then
    scaled up by scale_factor and rounded to integers. This removes book length
    as a feature while giving Naive Bayes the integer count data it expects.

    Args:
        books_dict: Dictionary mapping author → [(book_id, text), ...]
        vectorizer: Fitted CountVectorizer
        scale_factor: Scaling factor for frequencies (default: 100000)

    Returns:
        List of (author, book_id, scaled_vector) tuples where each vector
        contains scaled integer counts that all sum to approximately scale_factor

    Examples:
        >>> books = load_books_by_author()
        >>> vectorizer = create_count_vectorizer(books)
        >>> vectors = vectorize_books(books, vectorizer)
        >>> author, book_id, vec = vectors[0]
        >>> print(vec.shape)  # (vocab_size,)
        >>> print(vec.sum())  # Should be ~100000
        >>> print(vec.dtype)  # Should be int
    """
    vectorized_books = []

    for author, books in books_dict.items():
        for book_id, text in books:
            # Transform text to feature vector (counts)
            vector = vectorizer.transform([text]).toarray()[0].astype(float)

            # Normalize to frequencies (divide by sum)
            # This prevents classifiers from using book length
            total_count = vector.sum()
            if total_count > 0:
                frequencies = vector / total_count

                # Scale up and convert to integers for Naive Bayes
                # All books will have same total count (scale_factor)
                scaled_vector = np.round(frequencies * scale_factor).astype(int)
            else:
                scaled_vector = vector.astype(int)

            vectorized_books.append((author, book_id, scaled_vector))

    return vectorized_books
