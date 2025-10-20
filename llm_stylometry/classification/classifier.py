"""Output-code multi-class classifier for author attribution."""

from typing import Dict, List
import numpy as np
from sklearn.multiclass import OutputCodeClassifier as SKLearnOutputCodeClassifier
from sklearn.linear_model import LogisticRegression


class OutputCodeClassifier:
    """
    Wrapper around sklearn's OutputCodeClassifier for author attribution.

    Uses logistic regression as the base estimator with output-code multi-class
    strategy for robust predictions across multiple authors.

    Attributes:
        classifier: Fitted sklearn OutputCodeClassifier
        classes_: Array of class labels (author names)
        n_classes_: Number of classes

    Examples:
        >>> clf = OutputCodeClassifier()
        >>> clf.fit(X_train, y_train)
        >>> predictions = clf.predict(X_test)
        >>> weights = clf.get_feature_weights(feature_names)
    """

    def __init__(self, max_iter: int = 5000, random_state: int = 42):
        """
        Initialize OutputCodeClassifier.

        Args:
            max_iter: Maximum iterations for logistic regression (default: 5000 for high-dimensional data)
            random_state: Random seed for reproducibility
        """
        # Base estimator: Logistic Regression
        base_estimator = LogisticRegression(
            max_iter=max_iter,
            random_state=random_state,
            solver='lbfgs'
        )

        # Output-code multi-class classifier
        self.classifier = SKLearnOutputCodeClassifier(
            estimator=base_estimator,
            random_state=random_state
        )

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
        Extract author-specific feature weights by back-solving from outputs.

        For each author, creates an indicator output vector and back-solves to find
        the input (word count) pattern that would produce that output. This reveals
        which words are most characteristic of each author.

        Algorithm:
        1. For each author, create target output (indicator vector)
        2. Back-solve: input = pseudo_inverse(weights) @ (output - bias)
        3. Map feature names to back-solved weights

        Args:
            feature_names: List of feature names (vocabulary)

        Returns:
            Dictionary with keys:
            - author names: {word: weight, ...} for each author (author-specific)
            - 'overall': {word: avg_weight, ...} averaged across all authors

        Raises:
            ValueError: If classifier hasn't been fitted yet
        """
        if self.classes_ is None:
            raise ValueError("Classifier must be fitted before extracting weights")

        # Get decision function scores for a dummy input to understand dimensions
        # We'll extract the code_book and estimators to back-solve

        # Extract binary classifiers and code book from OutputCodeClassifier
        binary_classifiers = self.classifier.estimators_
        code_book = self.classifier.code_book_

        # code_book shape: (n_classes, n_binary_classifiers)
        # Each row is the target output for one class across all binary classifiers

        n_features = len(feature_names)
        n_binary = len(binary_classifiers)

        # Build weight matrix: (n_binary × n_features)
        # Each row is the coefficient vector from one binary classifier
        W = np.zeros((n_binary, n_features))
        b = np.zeros(n_binary)

        for i, binary_clf in enumerate(binary_classifiers):
            if hasattr(binary_clf, 'coef_') and hasattr(binary_clf, 'intercept_'):
                W[i, :] = binary_clf.coef_[0]
                b[i] = binary_clf.intercept_[0]

        # For each author, back-solve from their code to input features
        author_weights = {}

        for class_idx, author in enumerate(self.classes_):
            # Get target output code for this author
            target_output = code_book[class_idx, :]  # Shape: (n_binary,)

            # Back-solve: X = W^+ @ (Y - b)
            # where W^+ is the pseudo-inverse of W
            # W is (n_binary × n_features), so W^+ is (n_features × n_binary)

            W_pinv = np.linalg.pinv(W)  # Pseudo-inverse

            # Solve for input that produces this output
            input_weights = W_pinv @ (target_output - b)  # Shape: (n_features,)

            # Map to feature names
            author_weights[author] = {
                feature_name: float(weight)
                for feature_name, weight in zip(feature_names, input_weights)
            }

        # Compute overall weights as average across all authors
        overall_weights = {}
        for feature_name in feature_names:
            avg_weight = np.mean([
                author_weights[author][feature_name]
                for author in self.classes_
            ])
            overall_weights[feature_name] = float(avg_weight)

        return {**author_weights, 'overall': overall_weights}
