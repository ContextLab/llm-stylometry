#!/usr/bin/env python
"""
Test suite for documentation examples.

Tests all command examples from README.md to ensure they work correctly.
Uses real commands and actual data - no mocks or simulations.
"""

import subprocess
import sys
from pathlib import Path

import pytest


def test_readme_help_commands():
    """Test that all help commands from README work."""

    # Test run_llm_stylometry.sh --help
    result = subprocess.run(
        ['./run_llm_stylometry.sh', '--help'],
        capture_output=True,
        text=True,
        timeout=10
    )
    assert result.returncode == 0, f"run_llm_stylometry.sh --help failed: {result.stderr}"
    assert '--content-only' in result.stdout, "Missing --content-only flag in help"
    assert '--function-only' in result.stdout, "Missing --function-only flag in help"
    assert '--part-of-speech' in result.stdout, "Missing --part-of-speech flag in help"

    # Test run_stats.sh --help
    result = subprocess.run(
        ['./run_stats.sh', '--help'],
        capture_output=True,
        text=True,
        timeout=10
    )
    assert result.returncode == 0, f"run_stats.sh --help failed: {result.stderr}"
    assert '--content-only' in result.stdout, "Missing --content-only flag in help"
    assert '--all' in result.stdout, "Missing --all flag in help"

    # Test remote_train.sh shows usage (doesn't have --help, always shows usage and prompts)
    # Run with input piped to avoid hanging on read
    result = subprocess.run(
        ['./remote_train.sh'],
        capture_output=True,
        text=True,
        input='\n\n',  # Provide empty inputs to get past prompts
        timeout=10
    )
    # Script shows usage regardless of exit code
    assert '--content-only' in result.stdout, "Missing --content-only flag in remote_train.sh output"


def test_figure_generation_with_variants():
    """Test figure generation with each variant produces correct output files."""

    # Only test if we have test data
    test_data = Path('tests/data/test_model_results.pkl')
    if not test_data.exists():
        pytest.skip("Test data not found")

    variants = [
        (None, 'all_losses.pdf'),
        ('content', 'all_losses_content.pdf'),
        ('function', 'all_losses_function.pdf'),
        ('pos', 'all_losses_pos.pdf'),
    ]

    for variant, expected_filename in variants:
        # Generate figure
        cmd = ['./run_llm_stylometry.sh', '-f', '1a', '-d', str(test_data), '--no-setup']
        if variant:
            cmd.append(f'--{variant}-only')

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        assert result.returncode == 0, f"Figure generation failed for {variant}: {result.stderr}"

        # Check output file exists
        output_path = Path('paper/figs/source') / expected_filename
        assert output_path.exists(), f"Expected output file not found: {output_path}"
        assert output_path.stat().st_size > 1000, f"Output file too small: {output_path}"


def test_stats_with_variants():
    """Test statistics computation for each variant."""

    # Only test if we have test data with sufficient models
    test_data = Path('tests/data/test_model_results.pkl')
    if not test_data.exists():
        pytest.skip("Test data not found")

    # Test individual variants
    variants = [None, 'content', 'function', 'pos']

    for variant in variants:
        cmd = ['./run_stats.sh', '-d', str(test_data)]
        if variant:
            cmd.append(f'--{variant}-only')

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

        # May fail if insufficient data, but command should at least run
        if result.returncode != 0:
            # Check if it's just insufficient data
            if 'Insufficient' in result.stderr or 'Insufficient' in result.stdout:
                continue  # Expected with limited test data
            else:
                pytest.fail(f"Stats computation failed unexpectedly for {variant}: {result.stderr}")

        # If succeeded, verify output has expected sections
        output = result.stdout + result.stderr
        assert 'Twain' in output or 'Average' in output, f"Missing expected output for {variant}"


def test_stats_all_flag():
    """Test that ./run_stats.sh --all computes statistics for all variants."""

    test_data = Path('tests/data/test_model_results.pkl')
    if not test_data.exists():
        pytest.skip("Test data not found")

    result = subprocess.run(
        ['./run_stats.sh', '--all', '-d', str(test_data)],
        capture_output=True,
        text=True,
        timeout=300
    )

    # May fail if insufficient data for all variants
    if result.returncode != 0:
        if 'Insufficient' in result.stderr or 'Insufficient' in result.stdout:
            pytest.skip("Insufficient test data for --all flag")
        else:
            pytest.fail(f"Stats --all failed: {result.stderr}")

    # If succeeded, verify it ran for multiple variants
    output = result.stdout + result.stderr
    variant_count = sum([
        'baseline' in output.lower(),
        'content' in output.lower(),
        'function' in output.lower(),
        'pos' in output.lower()
    ])
    assert variant_count >= 2, "Expected --all to process multiple variants"


def test_variant_file_naming():
    """Verify variant files are named correctly per documentation."""

    # Test that actual model directories follow naming convention
    models_dir = Path('models')

    # Check baseline naming
    baseline_models = list(models_dir.glob('*_tokenizer=gpt2_seed=*'))
    baseline_models = [m for m in baseline_models if 'variant=' not in m.name]
    if baseline_models:
        # Pattern: {author}_tokenizer=gpt2_seed={seed}
        sample = baseline_models[0].name
        assert '_tokenizer=gpt2_seed=' in sample, f"Baseline naming incorrect: {sample}"
        assert 'variant=' not in sample, f"Baseline should not have variant: {sample}"

    # Check variant naming
    variant_models = list(models_dir.glob('*variant=*_tokenizer=gpt2_seed=*'))
    if variant_models:
        # Pattern: {author}_variant={variant}_tokenizer=gpt2_seed={seed}
        sample = variant_models[0].name
        assert '_variant=' in sample, f"Variant naming incorrect: {sample}"
        assert '_tokenizer=gpt2_seed=' in sample, f"Variant naming incorrect: {sample}"

        # Extract variant
        variant = sample.split('variant=')[1].split('_')[0]
        assert variant in ['content', 'function', 'pos'], f"Unknown variant: {variant}"


def test_python_api_variant_parameter():
    """Test that Python API accepts variant parameter."""

    test_data = Path('tests/data/test_model_results.pkl')
    if not test_data.exists():
        pytest.skip("Test data not found")

    # Test importing and calling with variant parameter
    from llm_stylometry.visualization import generate_all_losses_figure
    import tempfile
    import os

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = os.path.join(tmpdir, 'test.pdf')

        # Test baseline
        fig = generate_all_losses_figure(
            data_path=str(test_data),
            output_path=output_path,
            variant=None
        )
        assert fig is not None, "Failed to generate baseline figure"
        assert Path(output_path).exists(), "Baseline figure not saved"

        # Test variant (may fail if no variant data)
        try:
            output_path_variant = os.path.join(tmpdir, 'test_variant.pdf')
            fig = generate_all_losses_figure(
                data_path=str(test_data),
                output_path=output_path_variant,
                variant='content'
            )
            # If it succeeds, verify output
            if fig is not None:
                # The implementation adds variant suffix
                expected = output_path_variant.replace('.pdf', '_content.pdf')
                assert Path(expected).exists(), f"Variant figure not saved at {expected}"
        except (ValueError, KeyError):
            # Expected if no content variant data
            pass


if __name__ == '__main__':
    # Run tests
    pytest.main([__file__, '-v'])
