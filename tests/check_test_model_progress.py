#!/usr/bin/env python
"""Check progress of test model training."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'code'))

from constants import AUTHORS, MODELS_DIR

def main():
    """Check how many test models exist."""

    variants = ['baseline', 'content', 'function', 'pos']
    seeds = list(range(10))

    total_expected = len(AUTHORS) * len(seeds) * len(variants)

    # Count existing models
    models_by_variant = {v: 0 for v in variants}
    models_by_author = {a: 0 for a in AUTHORS}

    for variant in variants:
        for author in AUTHORS:
            for seed in seeds:
                # Check model directory
                if variant == 'baseline':
                    model_name = f"{author}_tokenizer=gpt2_seed={seed}"
                else:
                    model_name = f"{author}_variant={variant}_tokenizer=gpt2_seed={seed}"

                model_dir = MODELS_DIR / model_name
                if model_dir.exists():
                    # Check if training completed (has loss_logs.csv)
                    if (model_dir / 'loss_logs.csv').exists():
                        models_by_variant[variant] += 1
                        models_by_author[author] += 1

    total_complete = sum(models_by_variant.values())

    print("="*60)
    print("Test Model Training Progress")
    print("="*60)
    print(f"Total models: {total_complete} / {total_expected} ({100*total_complete/total_expected:.1f}%)")
    print()
    print("By variant:")
    for variant in variants:
        expected = len(AUTHORS) * len(seeds)
        count = models_by_variant[variant]
        pct = 100 * count / expected if expected > 0 else 0
        print(f"  {variant:10s}: {count:3d} / {expected:3d} ({pct:.1f}%)")
    print()
    print("By author:")
    for author in AUTHORS:
        expected = len(seeds) * len(variants)
        count = models_by_author[author]
        pct = 100 * count / expected if expected > 0 else 0
        print(f"  {author:10s}: {count:3d} / {expected:3d} ({pct:.1f}%)")

    print("="*60)

    if total_complete == total_expected:
        print("✓ All models complete!")
        return 0
    else:
        print(f"⏳ Training in progress... ({total_expected - total_complete} remaining)")
        return 1


if __name__ == '__main__':
    sys.exit(main())
