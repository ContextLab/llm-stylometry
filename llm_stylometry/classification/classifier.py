"""Naive Bayes classifier for author attribution."""

from typing import Dict, List
import numpy as np
from sklearn.naive_bayes import MultinomialNB


class OutputCodeClassifier:
    """
    Multinomial Naive Bayes classifier for author attribution.

    Designed for text classification with word count/frequency features.
    Models P(word|author) distributions for each author.

    Attributes:
        classifier: Fitted sklearn MultinomialNB
        classes_: Array of class labels (author names)
        n_classes_: Number of classes

    Examples:
        >>> clf = OutputCodeClassifier()
        >>> clf.fit(X_train, y_train)
        >>> predictions = clf.predict(X_test)
        >>> weights = clf.get_feature_weights(feature_names)
    """

    def __init__(self, alpha: float = 1.0):
        """
        Initialize Multinomial Naive Bayes classifier.

        Args:
            alpha: Additive smoothing parameter (default: 1.0 for Laplace smoothing)
        """
        self.classifier = MultinomialNB(alpha=alpha)
        self.classes_ = None
        self.n_classes_ = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Train classifier on feature matrix and labels.

        Args:
            X: Feature matrix (n_samples × n_features)
            y: Labels array (n_samples,)

        Returns:
            self
        """
        self.classifier.fit(X, y)
        self.classes_ = self.classifier.classes_
        self.n_classes_ = len(self.classes_)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict author labels for feature matrix.

        Args:
            X: Feature matrix (n_samples × n_features)

        Returns:
            Predicted labels array (n_samples,)
        """
        return self.classifier.predict(X)

    def get_feature_weights(self, feature_names: List[str]) -> Dict[str, Dict[str, float]]:
        """
        Extract author-specific feature weights from Naive Bayes.

        Naive Bayes directly models P(word|author) for each author. We extract
        these probabilities as feature weights. Higher probability means the word
        is more characteristic of that author.

        Args:
            feature_names: List of feature names (vocabulary)

        Returns:
            Dictionary with keys:
            - author names: {word: weight, ...} for each author (P(word|author))
            - 'overall': {word: avg_weight, ...} averaged across all authors

        Raises:
            ValueError: If classifier hasn't been fitted yet
        """
        if self.classes_ is None:
            raise ValueError("Classifier must be fitted before extracting weights")

        # Get feature log probabilities: shape (n_classes, n_features)
        # feature_log_prob_[i, j] = log P(word_j | class_i)
        feature_log_probs = self.classifier.feature_log_prob_

        # Convert to actual probabilities
        feature_probs = np.exp(feature_log_probs)

        # Extract per-author weights
        author_weights = {}
        for class_idx, author in enumerate(self.classes_):
            # Get probability distribution for this author
            author_probs = feature_probs[class_idx, :]

            # Map to feature names
            author_weights[author] = {
                feature_name: float(prob)
                for feature_name, prob in zip(feature_names, author_probs)
            }

        # Compute overall weights as average across all authors
        overall_weights = {}
        for feature_idx, feature_name in enumerate(feature_names):
            avg_weight = np.mean(feature_probs[:, feature_idx])
            overall_weights[feature_name] = float(avg_weight)

        return {**author_weights, 'overall': overall_weights}
