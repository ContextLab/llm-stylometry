#!/usr/bin/env python
"""
Comprehensive integration tests for analysis variant support.

Tests the full pipeline:
1. consolidate_model_results.py - variant parsing
2. compute_stats.py - variant filtering
3. All visualization functions - variant figures
4. Shell scripts - variant flags

Uses real test models (not mocks) from tests/test_models/
"""

import sys
import subprocess
import pickle
from pathlib import Path
import tempfile
import shutil

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from llm_stylometry.visualization import (
    generate_all_losses_figure,
    generate_stripplot_figure,
    generate_t_test_figure,
    generate_t_test_avg_figure,
    generate_loss_heatmap_figure,
    generate_3d_mds_figure,
    generate_oz_losses_figure
)


def test_consolidation():
    """Test that consolidated data has variant information."""
    print("\n" + "="*60)
    print("TEST 1: Model Results Consolidation (Verify)")
    print("="*60)

    data_path = 'tests/data/test_model_results_full.pkl'

    if not Path(data_path).exists():
        print(f"✗ Test data not found: {data_path}")
        print("  Run tests/create_test_models.py first to generate test data")
        return False

    # Load and verify the consolidated data
    import pandas as pd
    with open(data_path, 'rb') as f:
        df = pd.read_pickle(f)

    # Check variant column exists
    if 'variant' not in df.columns:
        print("✗ Missing 'variant' column in consolidated data")
        return False

    # Check we have baseline and variant models
    variants = df['variant'].unique()
    has_baseline = any(pd.isna(v) for v in variants)
    has_variants = any(v in ['content', 'function', 'pos'] for v in variants if pd.notna(v))

    if not has_baseline:
        print("✗ No baseline models found (variant=None)")
        return False

    if not has_variants:
        print("✗ No variant models found")
        return False

    print(f"✓ Consolidation data valid")
    print(f"  - Found {len(df['model_name'].unique())} unique models")
    print(f"  - Variants: {sorted([str(v) for v in variants])}")
    return True


def test_statistics():
    """Test compute_stats.py with variant filtering."""
    print("\n" + "="*60)
    print("TEST 2: Statistical Analysis with Variants")
    print("="*60)

    data_path = 'tests/data/test_model_results_full.pkl'

    # Test baseline
    print("\nTesting baseline statistics...")
    result = subprocess.run(
        [sys.executable, 'code/compute_stats.py', '--data', data_path],
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        print(f"✗ Baseline stats failed:")
        print(result.stderr)
        return False

    if "Baseline" not in result.stdout:
        print("✗ Baseline header not found in output")
        return False

    print("✓ Baseline statistics computed")

    # Test each variant
    for variant in ['content', 'function', 'pos']:
        print(f"\nTesting {variant} variant statistics...")
        result = subprocess.run(
            [sys.executable, 'code/compute_stats.py',
             '--data', data_path, '--variant', variant],
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            print(f"✗ {variant} stats failed:")
            print(result.stderr)
            return False

        if f"Variant: {variant}" not in result.stdout:
            print(f"✗ {variant} header not found in output")
            return False

        print(f"✓ {variant} statistics computed")

    return True


def test_visualizations():
    """Test all visualization functions with variant parameter."""
    print("\n" + "="*60)
    print("TEST 3: Visualization Functions with Variants")
    print("="*60)

    data_path = 'tests/data/test_model_results_full.pkl'

    # Create temporary output directory
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Test functions: (name, function, expected_file_prefix)
        test_cases = [
            ('all_losses', generate_all_losses_figure, 'all_losses'),
            ('stripplot', generate_stripplot_figure, 'stripplot'),
            ('t_test', generate_t_test_figure, 't_test'),
            ('t_test_avg', generate_t_test_avg_figure, 't_test_avg'),
            ('heatmap', generate_loss_heatmap_figure, 'heatmap'),
            ('mds', generate_3d_mds_figure, 'mds'),
            ('oz_losses', generate_oz_losses_figure, 'oz_losses'),
        ]

        for name, func, prefix in test_cases:
            print(f"\nTesting {name}...")

            # Test baseline (variant=None)
            baseline_path = tmpdir / f"{prefix}_baseline.pdf"
            try:
                kwargs = {'data_path': data_path, 'output_path': str(baseline_path)}
                if name in ['all_losses', 'stripplot', 't_test', 't_test_avg', 'oz_losses']:
                    kwargs['show_legend'] = False
                fig = func(**kwargs)
                import matplotlib.pyplot as plt
                plt.close(fig)

                if not baseline_path.exists():
                    print(f"  ✗ Baseline file not created")
                    return False
                print(f"  ✓ Baseline: {baseline_path.stat().st_size} bytes")
            except Exception as e:
                print(f"  ✗ Baseline failed: {e}")
                return False

            # Test content variant
            content_path = tmpdir / f"{prefix}_test.pdf"
            try:
                kwargs = {
                    'data_path': data_path,
                    'output_path': str(content_path),
                    'variant': 'content'
                }
                if name in ['all_losses', 'stripplot', 't_test', 't_test_avg', 'oz_losses']:
                    kwargs['show_legend'] = False
                fig = func(**kwargs)
                plt.close(fig)

                # Check that variant suffix was added
                expected_path = tmpdir / f"{prefix}_test_content.pdf"
                if not expected_path.exists():
                    print(f"  ✗ Content variant file not created with correct suffix")
                    print(f"     Expected: {expected_path}")
                    files = list(tmpdir.glob("*.pdf"))
                    print(f"     Found: {files}")
                    return False
                print(f"  ✓ Content variant: {expected_path.stat().st_size} bytes")
            except Exception as e:
                print(f"  ✗ Content variant failed: {e}")
                import traceback
                traceback.print_exc()
                return False

    print(f"\n✓ All {len(test_cases)} visualization functions tested successfully")
    return True


def test_shell_scripts():
    """Test shell script variant flag parsing."""
    print("\n" + "="*60)
    print("TEST 4: Shell Script Integration")
    print("="*60)

    # Test run_stats.sh help
    print("\nTesting run_stats.sh --help...")
    result = subprocess.run(['./run_stats.sh', '--help'], capture_output=True, text=True)
    if result.returncode != 0 or '--content-only' not in result.stdout:
        print("✗ run_stats.sh help text missing variant flags")
        return False
    print("✓ run_stats.sh help text includes variant flags")

    # Test run_llm_stylometry.sh help
    print("\nTesting run_llm_stylometry.sh --help...")
    result = subprocess.run(['./run_llm_stylometry.sh', '--help'], capture_output=True, text=True)
    if result.returncode != 0 or '--content-only' not in result.stdout:
        print("✗ run_llm_stylometry.sh help text missing variant flags")
        return False
    if 'function words masked' not in result.stdout:
        print("✗ run_llm_stylometry.sh help text missing variant descriptions")
        return False
    print("✓ run_llm_stylometry.sh help text includes variant flags and descriptions")

    return True


def main():
    """Run all integration tests."""
    print("╔══════════════════════════════════════════════════════════╗")
    print("║   Comprehensive Integration Tests for Variant Support   ║")
    print("╚══════════════════════════════════════════════════════════╝")

    # Ensure we're in the right directory
    if not Path('run_llm_stylometry.sh').exists():
        print("\n✗ Error: Must run from repository root")
        return 1

    # Run all tests
    tests = [
        ("Model Consolidation", test_consolidation),
        ("Statistical Analysis", test_statistics),
        ("Visualization Functions", test_visualizations),
        ("Shell Script Integration", test_shell_scripts),
    ]

    results = []
    for name, test_func in tests:
        try:
            success = test_func()
            results.append((name, success))
        except Exception as e:
            print(f"\n✗ Test '{name}' raised exception: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))

    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)

    for name, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{status}: {name}")

    passed = sum(1 for _, s in results if s)
    total = len(results)

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n✓ All integration tests passed!")
        return 0
    else:
        print(f"\n✗ {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    import pandas as pd  # Import here for testing
    sys.exit(main())
