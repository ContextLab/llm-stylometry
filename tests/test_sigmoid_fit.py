#!/usr/bin/env python
"""Tests for sigmoid fit to accuracy vs tokens data."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "code"))

import numpy as np
from fit_sigmoid import (
    bootstrap_sigmoid_ci,
    find_threshold_tokens,
    fit_sigmoid,
    load_per_author_accuracy,
    sigmoid,
)


def _get_mean_data():
    """Load mean accuracy data for fitting tests."""
    _, mean_acc = load_per_author_accuracy()
    mean_acc = mean_acc.sort_values("n_tokens")
    tokens = np.array(mean_acc["n_tokens"].values, dtype=float)
    accuracy = mean_acc["accuracy"].values
    log_tokens = np.log10(tokens)
    return log_tokens, accuracy, tokens


def test_sigmoid_fit_converges():
    """T020: Verify sigmoid fit produces R² > 0.95 with valid parameters."""
    log_tokens, accuracy, _ = _get_mean_data()
    popt, pcov = fit_sigmoid(log_tokens, accuracy)
    L, K, b, m = popt

    y_pred = sigmoid(log_tokens, *popt)
    ss_res = np.sum((accuracy - y_pred) ** 2)
    ss_tot = np.sum((accuracy - np.mean(accuracy)) ** 2)
    r_squared = 1 - ss_res / ss_tot

    assert r_squared > 0.95, f"R² = {r_squared:.4f}, expected > 0.95"
    assert 50 < L < 80, f"L = {L:.2f}, expected 50-80"
    assert 20 < K < 50, f"K = {K:.2f}, expected 20-50"
    assert 1 < b < 20, f"b = {b:.2f}, expected 1-20"
    assert 3.5 < m < 5.5, f"m = {m:.2f}, expected 3.5-5.5"
    assert 95 < L + K <= 101, f"Upper asymptote = {L + K:.2f}, expected ~100"

    perr = np.sqrt(np.diag(pcov))
    assert all(np.isfinite(perr)), f"Parameter errors not finite: {perr}"


def test_sigmoid_fit_residuals():
    """T020: Verify residual diagnostics are reasonable."""
    log_tokens, accuracy, _ = _get_mean_data()
    popt, _ = fit_sigmoid(log_tokens, accuracy)
    y_pred = sigmoid(log_tokens, *popt)
    residuals = accuracy - y_pred
    rmse = np.sqrt(np.mean(residuals**2))
    max_residual = np.max(np.abs(residuals))

    assert rmse < 5.0, f"RMSE = {rmse:.2f}%, expected < 5%"
    assert max_residual < 10.0, f"Max residual = {max_residual:.2f}%, expected < 10%"


def test_bootstrap_ci_produces_valid_range():
    """T020: Verify bootstrap CI from author resampling is valid."""
    per_author, _ = load_per_author_accuracy()
    popt, curves, thresholds, x_smooth = bootstrap_sigmoid_ci(
        per_author, n_bootstrap=200, seed=42
    )

    assert len(curves) > 100, f"Only {len(curves)} valid bootstrap fits out of 200"
    assert len(thresholds) > 50, f"Only {len(thresholds)} valid thresholds"

    ci_lo, ci_hi = np.percentile(thresholds, [2.5, 97.5])
    assert ci_lo > 1000, f"CI lower = {ci_lo:,.0f}, expected > 1,000"
    assert ci_hi < 200000, f"CI upper = {ci_hi:,.0f}, expected < 200,000"


def test_find_threshold_tokens_95():
    """T021: Verify threshold for 95% accuracy is in expected range."""
    log_tokens, accuracy, _ = _get_mean_data()
    popt, _ = fit_sigmoid(log_tokens, accuracy)
    threshold = find_threshold_tokens(popt, 95.0)

    assert threshold is not None, "Threshold computation returned None"
    assert 30000 < threshold < 70000, f"Threshold = {threshold:,.0f}, expected 30K-70K"

    predicted = sigmoid(np.log10(threshold), *popt)
    assert (
        abs(predicted - 95.0) < 0.1
    ), f"Sigmoid at threshold = {predicted:.2f}%, expected ~95%"


def test_find_threshold_tokens_above_asymptote():
    """T021: Verify threshold returns None when target exceeds asymptote."""
    log_tokens, accuracy, _ = _get_mean_data()
    popt, _ = fit_sigmoid(log_tokens, accuracy)
    L, K, _, _ = popt
    upper = L + K

    result = find_threshold_tokens(popt, upper + 5)
    assert result is None, "Expected None for target > asymptote"


def test_sigmoid_function_properties():
    """Verify sigmoid function has correct mathematical properties."""
    L, K, b, m = 70.0, 30.0, 5.0, 4.5

    assert abs(sigmoid(m, L, K, b, m) - (L + K / 2)) < 0.01
    assert abs(sigmoid(10.0, L, K, b, m) - (L + K)) < 0.01
    assert abs(sigmoid(0.0, L, K, b, m) - L) < 0.01

    x = np.linspace(3, 6, 100)
    y = sigmoid(x, L, K, b, m)
    assert all(np.diff(y) > 0), "Sigmoid should be monotonically increasing"


def test_per_author_accuracy_loads():
    """Verify per-author accuracy data loads correctly from Parquet."""
    per_author, mean_acc = load_per_author_accuracy()

    assert len(per_author) > 0, "No per-author data loaded"
    assert len(mean_acc) > 0, "No mean accuracy data loaded"
    assert set(per_author["author"].unique()) == set(
        [
            "baum",
            "thompson",
            "austen",
            "dickens",
            "fitzgerald",
            "melville",
            "twain",
            "wells",
        ]
    ), "Missing authors"
    assert len(mean_acc["n_tokens"].unique()) == 19, "Expected 19 token levels"
    assert all(
        0 <= a <= 100 for a in per_author["accuracy"]
    ), "Accuracy should be 0-100%"
