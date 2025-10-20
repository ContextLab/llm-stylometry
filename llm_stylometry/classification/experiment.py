"""High-level experiment runner for text classification."""

from pathlib import Path
from typing import Optional
import pickle
import pandas as pd

from .vectorizer import load_books_by_author, create_count_vectorizer, vectorize_books
from .cross_validation import generate_cv_splits, run_cross_validation


def run_classification_experiment(
    variant: Optional[str] = None,
    max_splits: int = 10000,
    seed: int = 42,
    data_dir: str = "data/cleaned",
    output_dir: str = "data/classifier_results"
) -> str:
    """
    Run complete text classification experiment.

    Steps:
    1. Load books for specified variant
    2. Create and fit CountVectorizer
    3. Vectorize all books
    4. Generate CV splits
    5. Run cross-validation
    6. Save results and fitted objects

    Args:
        variant: Analysis variant ('content', 'function', 'pos') or None for baseline
        max_splits: Maximum number of CV splits
        seed: Random seed for reproducibility
        data_dir: Base data directory
        output_dir: Directory to save results

    Returns:
        Path to saved results file

    Examples:
        >>> path = run_classification_experiment()  # Baseline
        >>> path = run_classification_experiment(variant='content')
    """
    print(f"Running classification experiment: {variant or 'baseline'}")

    # Step 1: Load books
    print("Loading books...")
    books_dict = load_books_by_author(data_dir=data_dir, variant=variant)

    if not books_dict:
        raise ValueError(f"No books found for variant: {variant}")

    total_books = sum(len(books) for books in books_dict.values())
    print(f"Loaded {total_books} books from {len(books_dict)} authors")

    # Step 2: Create and fit vectorizer
    print("Creating CountVectorizer...")
    vectorizer = create_count_vectorizer(books_dict)
    vocab_size = len(vectorizer.vocabulary_)
    print(f"Vocabulary size: {vocab_size} unique words")

    # Step 3: Vectorize books
    print("Vectorizing books...")
    vectorized_books = vectorize_books(books_dict, vectorizer)

    # Step 4: Generate CV splits
    print(f"Generating CV splits (max_splits={max_splits})...")
    cv_splits = generate_cv_splits(vectorized_books, max_splits=max_splits, seed=seed)
    print(f"Generated {len(cv_splits)} CV splits")

    # Step 5: Run cross-validation
    print("Running cross-validation...")
    results_df = run_cross_validation(vectorized_books, cv_splits, random_state=seed)

    # Step 6: Save results
    print("Saving results...")
    output_path = save_classification_results(
        results_df=results_df,
        vectorizer=vectorizer,
        variant=variant,
        n_splits=len(cv_splits),
        seed=seed,
        output_dir=output_dir
    )

    print(f"Results saved to: {output_path}")
    print(f"Overall accuracy: {results_df['accuracy'].mean():.4f}")

    return output_path


def save_classification_results(
    results_df: pd.DataFrame,
    vectorizer,
    variant: Optional[str] = None,
    n_splits: int = None,
    seed: int = None,
    output_dir: str = "data/classifier_results"
) -> str:
    """
    Save classification results and fitted objects to pickle file.

    Args:
        results_df: Results DataFrame from cross-validation
        vectorizer: Fitted CountVectorizer
        variant: Analysis variant or None for baseline
        n_splits: Number of CV splits
        seed: Random seed used
        output_dir: Directory to save results

    Returns:
        Path to saved pickle file

    Examples:
        >>> save_classification_results(results_df, vectorizer)
    """
    # Create output directory if needed
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Determine filename
    if variant is None:
        filename = "baseline.pkl"
    else:
        filename = f"{variant}.pkl"

    filepath = output_path / filename

    # Extract feature names
    feature_names = vectorizer.get_feature_names_out().tolist()

    # Package data
    data = {
        'results': results_df,
        'vectorizer': vectorizer,
        'feature_names': feature_names,
        'variant': variant,
        'n_splits': n_splits,
        'seed': seed
    }

    # Save to pickle
    with open(filepath, 'wb') as f:
        pickle.dump(data, f, protocol=4)

    return str(filepath)


def load_classification_results(filepath: str) -> dict:
    """
    Load classification results from pickle file.

    Args:
        filepath: Path to pickle file

    Returns:
        Dictionary with keys:
        - results: pd.DataFrame
        - vectorizer: CountVectorizer
        - feature_names: List[str]
        - variant: str or None
        - n_splits: int
        - seed: int

    Examples:
        >>> data = load_classification_results('data/classifier_results/baseline.pkl')
        >>> results_df = data['results']
    """
    with open(filepath, 'rb') as f:
        data = pickle.load(f)

    return data
