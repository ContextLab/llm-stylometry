#!/usr/bin/env python
"""
Test Experiment class with variants.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'code'))

from experiment import Experiment
from constants import AUTHORS, ANALYSIS_VARIANTS

def test_experiment_variants():
    """Test Experiment class with all variants."""
    print("\n" + "="*60)
    print("Test: Experiment Class with Variants")
    print("="*60)

    test_author = "fitzgerald"
    test_seed = 0

    # Test baseline experiment
    print("\n--- Testing baseline experiment ---")
    exp_baseline = Experiment(
        train_author=test_author,
        seed=test_seed,
        tokenizer_name="gpt2",
        analysis_variant=None
    )

    expected_name = f"{test_author}_tokenizer=gpt2_seed={test_seed}"
    assert exp_baseline.name == expected_name, f"Name mismatch: {exp_baseline.name} != {expected_name}"
    assert exp_baseline.analysis_variant is None
    print(f"✓ Baseline name: {exp_baseline.name}")
    print(f"✓ Data dir: {exp_baseline.data_dir}")

    # Test each variant experiment
    for variant in ANALYSIS_VARIANTS:
        print(f"\n--- Testing {variant} variant ---")

        exp_variant = Experiment(
            train_author=test_author,
            seed=test_seed,
            tokenizer_name="gpt2",
            analysis_variant=variant
        )

        expected_name = f"{test_author}_variant={variant}_tokenizer=gpt2_seed={test_seed}"
        assert exp_variant.name == expected_name, f"Name mismatch: {exp_variant.name} != {expected_name}"
        assert exp_variant.analysis_variant == variant
        assert f"{variant}_only" in str(exp_variant.data_dir)

        print(f"✓ Variant name: {exp_variant.name}")
        print(f"✓ Data dir: {exp_variant.data_dir}")

        # Verify eval paths exist
        for author in AUTHORS:
            eval_path = exp_variant.eval_paths[author]
            assert eval_path.exists(), f"Eval path doesn't exist: {eval_path}"
        print(f"✓ All {len(AUTHORS)} eval paths exist")

    # Test invalid variant
    print("\n--- Testing invalid variant ---")
    try:
        exp_invalid = Experiment(
            train_author=test_author,
            seed=test_seed,
            tokenizer_name="gpt2",
            analysis_variant="invalid"
        )
        assert False, "Should have raised ValueError for invalid variant"
    except ValueError as e:
        print(f"✓ Correctly raised ValueError: {e}")

    # Test special eval paths for Baum (baseline only)
    print("\n--- Testing special eval paths for Baum ---")
    exp_baum = Experiment(
        train_author="baum",
        seed=0,
        tokenizer_name="gpt2",
        analysis_variant=None
    )
    assert "non_oz_baum" in exp_baum.eval_paths
    assert "non_oz_thompson" in exp_baum.eval_paths
    assert "contested" in exp_baum.eval_paths
    print("✓ Special eval paths present for baseline Baum")

    exp_baum_variant = Experiment(
        train_author="baum",
        seed=0,
        tokenizer_name="gpt2",
        analysis_variant="content"
    )
    assert "non_oz_baum" not in exp_baum_variant.eval_paths
    print("✓ Special eval paths absent for variant Baum")

    print("\n" + "="*60)
    print("✓ ALL EXPERIMENT VARIANT TESTS PASSED")
    print("="*60)

if __name__ == "__main__":
    test_experiment_variants()