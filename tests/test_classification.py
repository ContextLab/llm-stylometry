"""Tests for text classification module (NO MOCKS - Real integration tests)."""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
import shutil

from llm_stylometry.classification import (
    load_books_by_author,
    create_count_vectorizer,
    vectorize_books,
    OutputCodeClassifier,
    generate_cv_splits,
    run_cross_validation,
    run_classification_experiment,
    save_classification_results
)


# Test fixtures path
FIXTURE_DIR = Path(__file__).parent / "fixtures" / "classification" / "cleaned"


class TestLoadBooksByAuthor:
    """Test book loading functionality."""

    def test_load_books_baseline(self):
        """Test loading books from baseline (real text files)."""
        books = load_books_by_author(data_dir=str(FIXTURE_DIR), variant=None)

        # Should have 3 authors
        assert len(books) == 3
        assert set(books.keys()) == {'baum', 'austen', 'dickens'}

        # Each author should have 3 books
        for author, book_list in books.items():
            assert len(book_list) == 3

            # Verify book structure
            for book_id, text in book_list:
                assert isinstance(book_id, str)
                assert isinstance(text, str)
                assert len(text) > 0  # Non-empty text


class TestCountVectorizer:
    """Test CountVectorizer creation and fitting."""

    def test_create_count_vectorizer(self):
        """Test creating CountVectorizer with real books."""
        books = load_books_by_author(data_dir=str(FIXTURE_DIR))
        vectorizer = create_count_vectorizer(books)

        # Verify vocabulary exists
        vocab = vectorizer.vocabulary_
        assert len(vocab) > 0

        # Verify stop_words=None (no filtering)
        assert vectorizer.stop_words is None

        # Verify common words are in vocabulary
        # (These should exist in real books)
        vocab_words = set(vectorizer.get_feature_names_out())
        assert len(vocab_words) > 100  # Reasonable vocabulary size

    def test_vectorizer_transform_shape(self):
        """Test that vectorizer produces correct output shape."""
        books = load_books_by_author(data_dir=str(FIXTURE_DIR))
        vectorizer = create_count_vectorizer(books)

        # Transform should work
        sample_text = list(books.values())[0][0][1]  # First book text
        vector = vectorizer.transform([sample_text])

        # Should be sparse matrix
        assert vector.shape[0] == 1
        assert vector.shape[1] == len(vectorizer.vocabulary_)


class TestVectorizeBooks:
    """Test book vectorization."""

    def test_vectorize_books(self):
        """Test vectorizing books produces correct structure."""
        books = load_books_by_author(data_dir=str(FIXTURE_DIR))
        vectorizer = create_count_vectorizer(books)
        vectorized = vectorize_books(books, vectorizer)

        # Should have 9 vectors (3 authors × 3 books)
        assert len(vectorized) == 9

        # Verify structure
        for author, book_id, vector in vectorized:
            assert author in {'baum', 'austen', 'dickens'}
            assert isinstance(book_id, str)
            assert isinstance(vector, np.ndarray)
            assert vector.shape == (len(vectorizer.vocabulary_),)
            # Vectors should be normalized to frequencies (sum to 1.0)
            assert np.isclose(vector.sum(), 1.0, atol=1e-6)

    def test_frequency_normalization(self):
        """Test that vectors are normalized to frequencies (sum to 1.0)."""
        books = load_books_by_author(data_dir=str(FIXTURE_DIR))
        vectorizer = create_count_vectorizer(books)
        vectorized = vectorize_books(books, vectorizer)

        # Check all books
        for author, book_id, vector in vectorized:
            # Each vector should sum to 1.0 (frequencies, not counts)
            assert np.isclose(vector.sum(), 1.0, atol=1e-6), \
                f"Vector for {author}/{book_id} sums to {vector.sum()}, expected 1.0"

            # All values should be non-negative frequencies
            assert (vector >= 0).all()
            assert (vector <= 1).all()


class TestOutputCodeClassifier:
    """Test OutputCodeClassifier training and prediction."""

    def test_classifier_training(self):
        """Test classifier trains without errors on real data."""
        books = load_books_by_author(data_dir=str(FIXTURE_DIR))
        vectorizer = create_count_vectorizer(books)
        vectorized = vectorize_books(books, vectorizer)

        # Extract features and labels
        X = np.array([vec for _, _, vec in vectorized])
        y = np.array([author for author, _, _ in vectorized])

        # Train classifier
        clf = OutputCodeClassifier(random_state=42)
        clf.fit(X, y)

        # Verify fitted attributes
        assert clf.classes_ is not None
        assert clf.n_classes_ == 3
        assert set(clf.classes_) == {'baum', 'austen', 'dickens'}

    def test_classifier_predictions(self):
        """Test classifier makes predictions."""
        books = load_books_by_author(data_dir=str(FIXTURE_DIR))
        vectorizer = create_count_vectorizer(books)
        vectorized = vectorize_books(books, vectorizer)

        X = np.array([vec for _, _, vec in vectorized])
        y = np.array([author for author, _, _ in vectorized])

        # Train on first 2 books per author (6 total)
        train_mask = np.array([i % 3 < 2 for i in range(9)])
        X_train, y_train = X[train_mask], y[train_mask]
        X_test, y_test = X[~train_mask], y[~train_mask]

        clf = OutputCodeClassifier(random_state=42)
        clf.fit(X_train, y_train)

        # Make predictions
        predictions = clf.predict(X_test)

        # Verify predictions
        assert len(predictions) == 3
        assert all(p in {'baum', 'austen', 'dickens'} for p in predictions)

    def test_feature_weights_extraction(self):
        """Test extracting author-specific feature weights via back-solving."""
        books = load_books_by_author(data_dir=str(FIXTURE_DIR))
        vectorizer = create_count_vectorizer(books)
        vectorized = vectorize_books(books, vectorizer)

        X = np.array([vec for _, _, vec in vectorized])
        y = np.array([author for author, _, _ in vectorized])

        clf = OutputCodeClassifier(random_state=42)
        clf.fit(X, y)

        # Extract weights
        feature_names = vectorizer.get_feature_names_out().tolist()
        weights = clf.get_feature_weights(feature_names)

        # Verify structure
        assert 'overall' in weights
        assert 'baum' in weights
        assert 'austen' in weights
        assert 'dickens' in weights

        # Verify all authors have same words but different weights
        for author in ['baum', 'austen', 'dickens']:
            assert len(weights[author]) == len(feature_names)

        # Verify author weights are different
        baum_weights = list(weights['baum'].values())
        austen_weights = list(weights['austen'].values())

        # At least some weights should differ
        assert not np.allclose(baum_weights, austen_weights)


class TestCrossValidation:
    """Test cross-validation split generation and execution."""

    def test_cv_split_generation(self):
        """Test generating CV splits."""
        books = load_books_by_author(data_dir=str(FIXTURE_DIR))
        vectorizer = create_count_vectorizer(books)
        vectorized = vectorize_books(books, vectorizer)

        splits = generate_cv_splits(vectorized, max_splits=10, seed=42)

        # Should have 10 splits (or fewer if not enough combinations)
        assert len(splits) > 0
        assert len(splits) <= 10

        # Verify split structure
        for train_idx, test_idx in splits:
            # Should hold out 3 books (1 per author)
            assert len(test_idx) == 3

            # Train and test should be disjoint
            assert set(train_idx).isdisjoint(set(test_idx))

            # Together should cover all books
            assert set(train_idx) | set(test_idx) == set(range(9))

    def test_run_cross_validation(self):
        """Test running full cross-validation."""
        books = load_books_by_author(data_dir=str(FIXTURE_DIR))
        vectorizer = create_count_vectorizer(books)
        vectorized = vectorize_books(books, vectorizer)

        splits = generate_cv_splits(vectorized, max_splits=5, seed=42)
        results_df = run_cross_validation(vectorized, splits, random_state=42)

        # Verify DataFrame structure (long format)
        assert isinstance(results_df, pd.DataFrame)
        assert len(results_df) == 5 * 3  # 5 splits × 3 held-out books

        # Verify columns
        required_cols = ['split_id', 'author', 'accuracy', 'held_out_book_id',
                        'predicted_author', 'true_author', 'classifier']
        assert all(col in results_df.columns for col in required_cols)

        # Verify accuracy values
        assert all(results_df['accuracy'].isin([0.0, 1.0]))

        # Verify classifier objects stored
        assert all(results_df['classifier'].notna())


class TestExperiment:
    """Test end-to-end experiment running."""

    def test_classification_experiment_end_to_end(self):
        """Test running complete classification experiment."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = run_classification_experiment(
                variant=None,
                max_splits=5,
                seed=42,
                data_dir=str(FIXTURE_DIR),
                output_dir=tmpdir
            )

            # Verify output file exists
            assert Path(output_path).exists()

            # Load and verify results
            import pickle
            with open(output_path, 'rb') as f:
                data = pickle.load(f)

            # Verify structure
            assert 'results' in data
            assert 'vectorizer' in data
            assert 'feature_names' in data
            assert 'variant' in data
            assert 'n_splits' in data
            assert 'seed' in data

            # Verify results DataFrame
            results_df = data['results']
            assert isinstance(results_df, pd.DataFrame)
            assert len(results_df) > 0

            # Verify vectorizer works
            vectorizer = data['vectorizer']
            test_text = "This is a test sentence."
            vec = vectorizer.transform([test_text])
            assert vec.shape[1] == len(data['feature_names'])

    def test_save_and_load_results(self):
        """Test saving and loading results."""
        books = load_books_by_author(data_dir=str(FIXTURE_DIR))
        vectorizer = create_count_vectorizer(books)
        vectorized = vectorize_books(books, vectorizer)
        splits = generate_cv_splits(vectorized, max_splits=3, seed=42)
        results_df = run_cross_validation(vectorized, splits, random_state=42)

        with tempfile.TemporaryDirectory() as tmpdir:
            # Save results
            output_path = save_classification_results(
                results_df=results_df,
                vectorizer=vectorizer,
                variant=None,
                n_splits=3,
                seed=42,
                output_dir=tmpdir
            )

            # Load back
            import pickle
            with open(output_path, 'rb') as f:
                data = pickle.load(f)

            # Verify all fields match
            assert data['variant'] is None
            assert data['n_splits'] == 3
            assert data['seed'] == 42
            assert len(data['feature_names']) == len(vectorizer.vocabulary_)


class TestReproducibility:
    """Test reproducibility with seeds."""

    def test_reproducibility_with_seed(self):
        """Test that same seed produces identical results."""
        books = load_books_by_author(data_dir=str(FIXTURE_DIR))
        vectorizer = create_count_vectorizer(books)
        vectorized = vectorize_books(books, vectorizer)

        # Run twice with same seed
        splits1 = generate_cv_splits(vectorized, max_splits=5, seed=42)
        splits2 = generate_cv_splits(vectorized, max_splits=5, seed=42)

        # Should produce identical splits
        assert len(splits1) == len(splits2)
        for (train1, test1), (train2, test2) in zip(splits1, splits2):
            assert train1 == train2
            assert test1 == test2

    def test_different_seeds_differ(self):
        """Test that different seeds produce different results."""
        books = load_books_by_author(data_dir=str(FIXTURE_DIR))
        vectorizer = create_count_vectorizer(books)
        vectorized = vectorize_books(books, vectorizer)

        # Run with different seeds
        splits1 = generate_cv_splits(vectorized, max_splits=10, seed=42)
        splits2 = generate_cv_splits(vectorized, max_splits=10, seed=123)

        # Should produce different splits
        assert splits1 != splits2


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_unbalanced_books_handling(self):
        """Test with unbalanced book counts."""
        # This test uses real data where authors already have equal books
        # In production, authors may have different book counts
        books = load_books_by_author(data_dir=str(FIXTURE_DIR))

        # Remove one book from one author to create imbalance
        books['baum'] = books['baum'][:2]  # Only 2 books for Baum

        vectorizer = create_count_vectorizer(books)
        vectorized = vectorize_books(books, vectorizer)

        # Should still generate splits
        splits = generate_cv_splits(vectorized, max_splits=5, seed=42)
        assert len(splits) > 0

        # Each split should still hold out 1 book per author
        for train_idx, test_idx in splits:
            assert len(test_idx) == 3  # 1 per author
