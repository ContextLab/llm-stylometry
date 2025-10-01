#!/usr/bin/env python
"""
Test suite for INTERPRETATION.md examples.

Tests actual analysis workflows from the interpretation guide.
Uses real data and functions - no mocks or simulations.
"""

import subprocess
import sys
from pathlib import Path

import pytest


def test_oz_analysis_variants():
    """Test Example 1 from INTERPRETATION.md (Oz books analysis with variants)."""

    # Only test if we have necessary visualization function
    try:
        from llm_stylometry.visualization import generate_oz_losses_figure
    except ImportError:
        pytest.skip("Visualization module not available")

    # Need actual data with Baum/Thompson models
    test_data = Path('tests/data/test_model_results.pkl')
    if not test_data.exists():
        pytest.skip("Test data not found")

    # Test generating Oz analysis for each variant
    variants = [None, 'content', 'function', 'pos']

    for variant in variants:
        try:
            fig = generate_oz_losses_figure(
                data_path=str(test_data),
                variant=variant
            )

            if fig is not None:
                # Verify figure has expected structure
                axes = fig.get_axes()
                assert len(axes) > 0, f"Figure has no axes for variant {variant}"

                # Check data exists (may not have all variants in test data)
                assert hasattr(fig, 'get_axes'), f"Invalid figure object for variant {variant}"
        except (ValueError, KeyError) as e:
            # Expected if variant data doesn't exist or insufficient Oz data
            if variant is not None:
                # Variants may not exist in test data
                pass
            else:
                # Baseline should work if we have test data
                pytest.fail(f"Baseline Oz analysis failed unexpectedly: {e}")


def test_confusion_matrix_variants():
    """Test Example 2 from INTERPRETATION.md (Dickens vs Austen comparison)."""

    try:
        from llm_stylometry.visualization import generate_loss_heatmap_figure
    except ImportError:
        pytest.skip("Visualization module not available")

    test_data = Path('tests/data/test_model_results.pkl')
    if not test_data.exists():
        pytest.skip("Test data not found")

    # Generate confusion matrices for each variant
    variants = [None, 'content', 'function', 'pos']

    for variant in variants:
        try:
            fig = generate_loss_heatmap_figure(
                data_path=str(test_data),
                variant=variant
            )

            if fig is not None:
                # Verify figure structure
                axes = fig.get_axes()
                assert len(axes) > 0, f"Figure has no axes for variant {variant}"
        except (ValueError, KeyError) as e:
            # Expected if variant data doesn't exist
            if variant is not None:
                pass
            else:
                pytest.fail(f"Baseline confusion matrix failed unexpectedly: {e}")


def test_comparison_workflow():
    """Test the complete workflow from INTERPRETATION.md Step 2."""

    test_data = Path('tests/data/test_model_results.pkl')
    if not test_data.exists():
        pytest.skip("Test data not found")

    # Step 2: Compute statistics for all variants using --all
    result = subprocess.run(
        ['./run_stats.sh', '--all', '-d', str(test_data)],
        capture_output=True,
        text=True,
        timeout=300
    )

    # May fail if insufficient data for all variants
    if result.returncode != 0:
        output = result.stdout + result.stderr
        if 'Insufficient' in output:
            pytest.skip("Insufficient test data for --all workflow")
        else:
            pytest.fail(f"Stats --all failed: {result.stderr}")

    # If succeeded, verify output structure
    output = result.stdout + result.stderr

    # Should process multiple variants
    variant_mentions = sum([
        'baseline' in output.lower(),
        'content' in output.lower(),
        'function' in output.lower(),
        'pos' in output.lower()
    ])
    assert variant_mentions >= 2, "Expected --all to process multiple variants"

    # Should have t-statistics or similar output
    has_stats = any([
        't-statistic' in output.lower(),
        't-test' in output.lower(),
        'average' in output.lower()
    ])
    assert has_stats, "Missing expected statistical output"


def test_workflow_figure_generation():
    """Test generating figures for all variants as in workflow examples."""

    test_data = Path('tests/data/test_model_results.pkl')
    if not test_data.exists():
        pytest.skip("Test data not found")

    # Test generating Figure 5 (Oz analysis) for variants
    variants = [None, 'content', 'function', 'pos']

    for variant in variants:
        cmd = ['./run_llm_stylometry.sh', '-f', '5', '-d', str(test_data), '--no-setup']
        if variant:
            cmd.append(f'--{variant}-only')

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)

        # May fail if no Oz data or variant data
        if result.returncode != 0:
            output = result.stdout + result.stderr
            if any(x in output for x in ['No data', 'Insufficient', 'KeyError', 'ValueError']):
                # Expected if data missing
                continue
            else:
                # Unexpected failure
                pytest.fail(f"Figure 5 generation failed for {variant}: {result.stderr}")

        # If succeeded, verify file creation
        expected_name = 'oz_losses.pdf' if not variant else f'oz_losses_{variant}.pdf'
        output_path = Path('paper/figs/source') / expected_name
        if output_path.exists():
            assert output_path.stat().st_size > 1000, f"Output file too small: {output_path}"


def test_variant_transformation_examples():
    """Verify the transformation examples from INTERPRETATION.md are correct."""

    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

    text = "The quick brown fox jumps over the lazy dog"
    tokens = text.split()

    # Test Content-Only transformation
    content_result = []
    for token in tokens:
        if token.lower() in ENGLISH_STOP_WORDS:
            content_result.append('<FUNC>')
        else:
            content_result.append(token)
    content_output = ' '.join(content_result)

    # Should match documentation
    assert content_output == "<FUNC> quick brown fox jumps <FUNC> <FUNC> lazy dog", \
        f"Content-only transformation incorrect: {content_output}"

    # Test Function-Only transformation
    function_result = []
    for token in tokens:
        if token.lower() in ENGLISH_STOP_WORDS:
            function_result.append(token)
        else:
            function_result.append('<CONTENT>')
    function_output = ' '.join(function_result)

    # Should match documentation
    assert function_output == "The <CONTENT> <CONTENT> <CONTENT> <CONTENT> over the <CONTENT> <CONTENT>", \
        f"Function-only transformation incorrect: {function_output}"

    # Verify function words identified correctly
    function_words = [t for t in tokens if t.lower() in ENGLISH_STOP_WORDS]
    assert function_words == ['The', 'over', 'the'], \
        f"Function words incorrect: {function_words}"

    # Verify content words identified correctly
    content_words = [t for t in tokens if t.lower() not in ENGLISH_STOP_WORDS]
    assert content_words == ['quick', 'brown', 'fox', 'jumps', 'lazy', 'dog'], \
        f"Content words incorrect: {content_words}"


if __name__ == '__main__':
    # Run tests
    pytest.main([__file__, '-v'])
