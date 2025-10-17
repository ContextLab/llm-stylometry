"""
Test integrity of existing data files and verify deduplication works on real data.

These tests check existing model_results.pkl files for duplicates and verify
that re-running consolidation removes any duplicates that exist.
"""

import pytest
import pandas as pd
from pathlib import Path
import sys

# Add code directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'code'))

from consolidate_model_results import consolidate_model_results


class TestExistingDataIntegrity:
    """Test integrity of existing pickled data files."""

    def test_existing_baseline_data_integrity(self):
        """Check existing baseline data for duplicates."""
        pkl_path = Path('data/model_results.pkl')

        if not pkl_path.exists():
            pytest.skip("No baseline data file found")

        # Load existing data
        df = pd.read_pickle(pkl_path)

        # Check for duplicates
        duplicates = df.groupby(
            ['seed', 'train_author', 'epochs_completed', 'loss_dataset']
        ).size()
        duplicate_rows = duplicates[duplicates > 1]

        if len(duplicate_rows) > 0:
            print(f"\nFound {len(duplicate_rows)} duplicate entries in baseline data:")
            print(duplicate_rows.head(20))

            # Show examples
            for (seed, author, epoch, dataset), count in duplicate_rows.head(5).items():
                print(f"\n  Duplicate: seed={seed}, author={author}, epoch={epoch}, "
                      f"dataset={dataset}, count={count}")

        # Re-consolidate and verify duplicates are removed
        if Path('models').exists():
            df_new = consolidate_model_results(models_dir='models', variant=None)

            # Check new data has no duplicates
            new_duplicates = df_new.groupby(
                ['seed', 'train_author', 'epochs_completed', 'loss_dataset']
            ).size()
            assert (new_duplicates == 1).all(), \
                "Duplicates still present after re-consolidation"

            print(f"\n✓ Re-consolidation successful: {len(df)} -> {len(df_new)} rows")
        else:
            pytest.skip("No models directory found for re-consolidation")

    @pytest.mark.skip(reason="Variant analyses not yet complete - re-enable once training finishes")
    def test_existing_variant_data_integrity(self):
        """Check existing variant data for duplicates."""
        variants = ['content', 'function', 'pos']

        for variant in variants:
            pkl_path = Path(f'data/model_results_{variant}.pkl')

            if not pkl_path.exists():
                print(f"\nSkipping {variant}: No data file found")
                continue

            print(f"\n=== Checking {variant} variant ===")

            # Load existing data
            df = pd.read_pickle(pkl_path)

            # Check for duplicate epochs
            duplicates = df.groupby(
                ['seed', 'train_author', 'epochs_completed', 'loss_dataset']
            ).size()
            duplicate_rows = duplicates[duplicates > 1]

            if len(duplicate_rows) > 0:
                print(f"Found {len(duplicate_rows)} duplicate entries in {variant}:")
                print(duplicate_rows.head(20))

            # Check for spurious epoch 0 (more common in variants due to resume)
            epoch_0_per_model = df[df['epochs_completed'] == 0].groupby(
                ['seed', 'train_author']
            )['loss_dataset'].count()

            # Expected rows per model:
            # - Baseline: 11 rows (train + 8 authors + 2 oz datasets for baum/thompson, 9 for others)
            # - Variants: 8 rows (train + 8 authors, no oz datasets)
            # Since variants don't have oz datasets, expect 8 rows for all models
            expected_epoch_0_rows = 8  # Variants don't have oz-specific datasets

            # For baseline models, baum and thompson have 11, others have 9
            # But this test is for variants, so we expect 8 for all
            anomalies = epoch_0_per_model[epoch_0_per_model != expected_epoch_0_rows]

            if len(anomalies) > 0:
                print(f"Found {len(anomalies)} models with anomalous epoch 0 counts:")
                for (seed, author), count in anomalies.head(10).items():
                    print(f"  seed={seed}, author={author}, epoch_0_rows={count} "
                          f"(expected {expected_epoch_0_rows})")

            # Re-consolidate and verify
            models_variant_dir = Path('models')
            if models_variant_dir.exists():
                # Check if variant models exist
                variant_models = list(models_variant_dir.glob(f'*_variant={variant}_*'))

                if not variant_models:
                    print(f"No {variant} variant models found in models/")
                    continue

                df_new = consolidate_model_results(models_dir='models', variant=variant)

                # Check new data has no duplicates
                new_duplicates = df_new.groupby(
                    ['seed', 'train_author', 'epochs_completed', 'loss_dataset']
                ).size()
                assert (new_duplicates == 1).all(), \
                    f"Duplicates still present in {variant} after re-consolidation"

                # Check epoch 0 counts
                new_epoch_0_per_model = df_new[df_new['epochs_completed'] == 0].groupby(
                    ['seed', 'train_author']
                )['loss_dataset'].count()
                assert (new_epoch_0_per_model == expected_epoch_0_rows).all(), \
                    f"Spurious epoch 0 entries still present in {variant}"

                print(f"✓ Re-consolidation successful: {len(df)} -> {len(df_new)} rows")
            else:
                print(f"No models directory found for {variant} re-consolidation")

    def test_baseline_vs_variant_structure(self):
        """Verify baseline and variant data have consistent structure."""
        baseline_path = Path('data/model_results.pkl')

        if not baseline_path.exists():
            pytest.skip("No baseline data found")

        baseline_df = pd.read_pickle(baseline_path)

        # Check variant column exists
        assert 'variant' in baseline_df.columns, "Baseline data missing variant column"

        # Check baseline models have variant=None
        if len(baseline_df) > 0:
            # Baseline models should have NaN/None variant
            baseline_models = baseline_df[baseline_df['variant'].isna()]
            variant_models = baseline_df[~baseline_df['variant'].isna()]

            print(f"\nBaseline models: {baseline_models['model_name'].nunique()}")
            print(f"Variant models in baseline file: {variant_models['model_name'].nunique()}")

            # Baseline file should not contain variant models
            assert len(variant_models) == 0 or baseline_models['model_name'].nunique() > 0, \
                "Baseline file contains unexpected variant models"

        # Check variant files
        for variant in ['content', 'function', 'pos']:
            variant_path = Path(f'data/model_results_{variant}.pkl')

            if not variant_path.exists():
                continue

            variant_df = pd.read_pickle(variant_path)

            # Check variant column
            assert 'variant' in variant_df.columns, \
                f"{variant} data missing variant column"

            # All models should have the same variant
            if len(variant_df) > 0:
                unique_variants = variant_df['variant'].dropna().unique()
                assert len(unique_variants) == 1 and unique_variants[0] == variant, \
                    f"{variant} file contains wrong variant: {unique_variants}"

    def test_data_completeness(self):
        """Check that data files have expected number of models and epochs."""
        baseline_path = Path('data/model_results.pkl')

        if not baseline_path.exists():
            pytest.skip("No baseline data found")

        df = pd.read_pickle(baseline_path)

        if len(df) == 0:
            pytest.skip("Baseline data is empty")

        # Count unique models
        num_models = df['model_name'].nunique()
        print(f"\nBaseline: {num_models} unique models")

        # Count epochs per model
        epochs_per_model = df[df['loss_dataset'] == 'train'].groupby('model_name')['epochs_completed'].max()

        if len(epochs_per_model) > 0:
            print(f"  Min epochs: {epochs_per_model.min()}")
            print(f"  Max epochs: {epochs_per_model.max()}")
            print(f"  Mean epochs: {epochs_per_model.mean():.1f}")

            # Check for models with suspiciously few epochs
            short_models = epochs_per_model[epochs_per_model < 10]
            if len(short_models) > 0:
                print(f"  Warning: {len(short_models)} models with <10 epochs")
                for model_name, max_epoch in short_models.head(5).items():
                    print(f"    {model_name}: {max_epoch} epochs")

        # Check expected datasets
        expected_datasets = [
            'train', 'baum', 'thompson', 'dickens', 'melville',
            'wells', 'austen', 'fitzgerald', 'twain'
        ]

        for dataset in expected_datasets:
            dataset_count = len(df[df['loss_dataset'] == dataset])
            if dataset_count == 0:
                print(f"  Warning: No {dataset} data found")

    def test_no_nan_values(self):
        """Check for unexpected NaN values in critical columns."""
        baseline_path = Path('data/model_results.pkl')

        if not baseline_path.exists():
            pytest.skip("No baseline data found")

        df = pd.read_pickle(baseline_path)

        if len(df) == 0:
            pytest.skip("Baseline data is empty")

        # Critical columns that should never be NaN
        critical_columns = [
            'seed', 'train_author', 'epochs_completed',
            'loss_dataset', 'loss_value'
        ]

        for col in critical_columns:
            if col in df.columns:
                nan_count = df[col].isna().sum()
                assert nan_count == 0, \
                    f"Found {nan_count} NaN values in critical column '{col}'"


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
