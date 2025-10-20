"""Classification module for word count-based text classification."""

from .vectorizer import load_books_by_author, create_count_vectorizer, vectorize_books
from .classifier import OutputCodeClassifier
from .cross_validation import generate_cv_splits, run_cross_validation
from .experiment import run_classification_experiment, save_classification_results

__all__ = [
    'load_books_by_author',
    'create_count_vectorizer',
    'vectorize_books',
    'OutputCodeClassifier',
    'generate_cv_splits',
    'run_cross_validation',
    'run_classification_experiment',
    'save_classification_results',
]
