#!/usr/bin/env python
"""
Train comprehensive test models for full statistical testing.

Creates 320 models total:
- 8 authors × 10 seeds × 4 variants = 320 models
- Small models (2 layers, 64 embd) for speed
- 50 epochs each for threshold testing
- ~640 minutes total (~2 min/model)

This provides enough data for all statistical tests in compute_stats.py to work.
"""

import sys
import subprocess
from pathlib import Path

# Add code to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'code'))

from constants import AUTHORS

def main():
    """Train all test models."""
    print("="*60)
    print("Training Comprehensive Test Suite")
    print("="*60)
    print(f"Authors: {len(AUTHORS)} - {AUTHORS}")
    print(f"Seeds: 10 (0-9)")
    print(f"Variants: 4 (baseline, content, function, pos)")
    print(f"Total models: {len(AUTHORS) * 10 * 4} = 320")
    print(f"Estimated time: ~640 minutes (~10.7 hours)")
    print("="*60)

    response = input("\nThis will take ~11 hours. Proceed? [y/N]: ")
    if response.lower() != 'y':
        print("Cancelled.")
        return 1

    # Use create_test_models.py with all combinations
    variants = ['baseline', 'content', 'function', 'pos']
    seeds = list(range(10))

    models_trained = 0
    models_failed = 0

    for variant in variants:
        print(f"\n{'='*60}")
        print(f"Training {variant.upper()} Variant")
        print(f"{'='*60}")

        for author in AUTHORS:
            for seed in seeds:
                print(f"\nTraining: {author}, seed={seed}, variant={variant}")

                # Build command to train this model
                variant_flag = f"--variant={variant}" if variant != 'baseline' else ""
                cmd = [
                    sys.executable,
                    'tests/create_test_models.py',
                    '--author', author,
                    '--seed', str(seed),
                    '--epochs', '50',
                    '--tokens', '5000'
                ]
                if variant_flag:
                    cmd.append(variant_flag)

                try:
                    result = subprocess.run(
                        cmd,
                        capture_output=True,
                        text=True,
                        timeout=300  # 5 min timeout per model
                    )

                    if result.returncode == 0:
                        models_trained += 1
                        print(f"  ✓ Success ({models_trained}/{len(AUTHORS)*10*4})")
                    else:
                        models_failed += 1
                        print(f"  ✗ Failed: {result.stderr[:200]}")
                except subprocess.TimeoutExpired:
                    models_failed += 1
                    print(f"  ✗ Timeout")
                except Exception as e:
                    models_failed += 1
                    print(f"  ✗ Error: {e}")

    print(f"\n{'='*60}")
    print(f"Training Complete")
    print(f"{'='*60}")
    print(f"Models trained: {models_trained}")
    print(f"Models failed: {models_failed}")

    # Consolidate
    print(f"\nConsolidating results...")
    result = subprocess.run(
        [sys.executable, 'code/consolidate_model_results.py'],
        capture_output=True,
        text=True
    )

    if result.returncode == 0:
        print("✓ Consolidation successful")
        print(f"  Output: data/model_results.pkl")
    else:
        print(f"✗ Consolidation failed: {result.stderr}")
        return 1

    # Test stats
    print(f"\nTesting statistics...")
    result = subprocess.run(
        [sys.executable, 'code/compute_stats.py'],
        capture_output=True,
        text=True,
        timeout=60
    )

    if result.returncode == 0:
        print("✓ Statistics computed successfully")
        print("\nSample output:")
        print(result.stdout[:500])
    else:
        print(f"✗ Statistics failed: {result.stderr[:200]}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
