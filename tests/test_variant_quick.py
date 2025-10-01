#!/usr/bin/env python
"""
Quick integration tests for analysis variant support.

Tests the key functionality without expensive computations:
1. Data loading with variant filtering
2. Visualization functions accept variant parameter
3. Shell scripts have correct flags

Uses real test data but skips expensive figure generation.
"""

import sys
import subprocess
import pickle
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_data_loading():
    """Test that test data has variant information."""
    print("\n" + "="*60)
    print("TEST 1: Data Loading and Variant Column")
    print("="*60)

    data_path = 'tests/data/test_model_results_full.pkl'

    if not Path(data_path).exists():
        print(f"✗ Test data not found: {data_path}")
        return False

    import pandas as pd
    df = pd.read_pickle(data_path)

    # Check variant column exists
    if 'variant' not in df.columns:
        print("✗ Missing 'variant' column")
        return False

    # Check we have all expected variants
    variants = set(str(v) for v in df['variant'].unique())
    # Should have 'None' (string representation of None) or 'nan', plus the 3 variants
    expected_variants = {'content', 'function', 'pos'}
    baseline_present = 'None' in variants or 'nan' in variants

    if not expected_variants.issubset(variants):
        print(f"✗ Missing expected variants. Expected {expected_variants}, got {variants}")
        return False

    if not baseline_present:
        print(f"✗ No baseline indicator found. Expected 'None' or 'nan', got {variants}")
        return False

    # Check baseline filtering
    df_baseline = df[df['variant'].isna()]
    if len(df_baseline) == 0:
        print("✗ No baseline models found")
        return False

    # Check variant filtering
    for variant in ['content', 'function', 'pos']:
        df_variant = df[df['variant'] == variant]
        if len(df_variant) == 0:
            print(f"✗ No {variant} models found")
            return False

    print(f"✓ Data loading successful")
    print(f"  - Total models: {len(df['model_name'].unique())}")
    print(f"  - Baseline models: {len(df_baseline['model_name'].unique())}")
    print(f"  - Content models: {len(df[df['variant']=='content']['model_name'].unique())}")
    print(f"  - Function models: {len(df[df['variant']=='function']['model_name'].unique())}")
    print(f"  - POS models: {len(df[df['variant']=='pos']['model_name'].unique())}")
    return True


def test_visualization_signatures():
    """Test that visualization functions accept variant parameter."""
    print("\n" + "="*60)
    print("TEST 2: Visualization Function Signatures")
    print("="*60)

    from llm_stylometry.visualization import (
        generate_all_losses_figure,
        generate_stripplot_figure,
        generate_t_test_figure,
        generate_t_test_avg_figure,
        generate_loss_heatmap_figure,
        generate_3d_mds_figure,
        generate_oz_losses_figure
    )
    import inspect

    functions = [
        ('all_losses', generate_all_losses_figure),
        ('stripplot', generate_stripplot_figure),
        ('t_test', generate_t_test_figure),
        ('t_test_avg', generate_t_test_avg_figure),
        ('heatmap', generate_loss_heatmap_figure),
        ('mds', generate_3d_mds_figure),
        ('oz_losses', generate_oz_losses_figure),
    ]

    for name, func in functions:
        sig = inspect.signature(func)
        if 'variant' not in sig.parameters:
            print(f"✗ {name}: missing 'variant' parameter")
            return False

        # Check default is None
        if sig.parameters['variant'].default is not None:
            print(f"✗ {name}: variant parameter should default to None")
            return False

    print(f"✓ All {len(functions)} visualization functions have variant parameter")
    return True


def test_variant_filtering():
    """Test that variant filtering works correctly."""
    print("\n" + "="*60)
    print("TEST 3: Variant Filtering Logic")
    print("="*60)

    data_path = 'tests/data/test_model_results_full.pkl'
    import pandas as pd

    # Load full data
    df_full = pd.read_pickle(data_path)

    # Test baseline filtering (variant=None)
    df_baseline = df_full[df_full['variant'].isna()].copy()

    # Test content filtering
    df_content = df_full[df_full['variant'] == 'content'].copy()

    # Verify no overlap
    baseline_models = set(df_baseline['model_name'].unique())
    content_models = set(df_content['model_name'].unique())

    if baseline_models & content_models:
        print("✗ Baseline and content models overlap!")
        return False

    # Verify all content models have variant='content'
    if not all(df_content['variant'] == 'content'):
        print("✗ Content filtered data contains non-content models")
        return False

    # Verify baseline models have variant=None
    if not all(df_baseline['variant'].isna()):
        print("✗ Baseline filtered data contains variant models")
        return False

    print("✓ Variant filtering works correctly")
    print(f"  - Baseline models: {len(baseline_models)}")
    print(f"  - Content models: {len(content_models)}")
    print(f"  - No overlap: True")
    return True


def test_compute_stats():
    """Test compute_stats.py accepts variant flag."""
    print("\n" + "="*60)
    print("TEST 4: compute_stats.py Variant Support")
    print("="*60)

    data_path = 'tests/data/test_model_results_full.pkl'

    # Test baseline
    result = subprocess.run(
        [sys.executable, 'code/compute_stats.py', '--data', data_path],
        capture_output=True,
        text=True,
        timeout=30
    )

    if result.returncode != 0:
        print(f"✗ Baseline stats failed: {result.stderr[:200]}")
        return False

    if "Baseline" not in result.stdout:
        print("✗ Baseline header not found")
        return False

    # Test content variant
    result = subprocess.run(
        [sys.executable, 'code/compute_stats.py', '--data', data_path, '--variant', 'content'],
        capture_output=True,
        text=True,
        timeout=30
    )

    if result.returncode != 0:
        print(f"✗ Content stats failed: {result.stderr[:200]}")
        return False

    if "Variant: content" not in result.stdout:
        print("✗ Content header not found")
        return False

    print("✓ compute_stats.py works with baseline and content variant")
    return True


def test_shell_scripts():
    """Test shell script variant flags."""
    print("\n" + "="*60)
    print("TEST 5: Shell Script Variant Flags")
    print("="*60)

    # Test run_stats.sh
    result = subprocess.run(['./run_stats.sh', '--help'], capture_output=True, text=True, timeout=10)
    if result.returncode != 0:
        print("✗ run_stats.sh --help failed")
        return False

    if '--content-only' not in result.stdout:
        print("✗ run_stats.sh missing --content-only flag")
        return False

    if '--all' not in result.stdout:
        print("✗ run_stats.sh missing --all flag")
        return False

    # Test run_llm_stylometry.sh
    result = subprocess.run(['./run_llm_stylometry.sh', '--help'], capture_output=True, text=True, timeout=10)
    if result.returncode != 0:
        print("✗ run_llm_stylometry.sh --help failed")
        return False

    if '--content-only' not in result.stdout:
        print("✗ run_llm_stylometry.sh missing --content-only flag")
        return False

    if 'function words masked' not in result.stdout:
        print("✗ run_llm_stylometry.sh missing variant descriptions")
        return False

    print("✓ Shell scripts have correct variant flags and help text")
    return True


def main():
    """Run all quick tests."""
    print("╔══════════════════════════════════════════════════════════╗")
    print("║         Quick Integration Tests for Variants            ║")
    print("╚══════════════════════════════════════════════════════════╝")

    if not Path('run_llm_stylometry.sh').exists():
        print("\n✗ Error: Must run from repository root")
        return 1

    tests = [
        ("Data Loading", test_data_loading),
        ("Function Signatures", test_visualization_signatures),
        ("Variant Filtering", test_variant_filtering),
        ("compute_stats.py", test_compute_stats),
        ("Shell Scripts", test_shell_scripts),
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
    print("SUMMARY")
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
    sys.exit(main())
