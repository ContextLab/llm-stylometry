"""
Test duplicate epoch removal in consolidate_model_results.py

These tests use synthetic CSV files with known duplicate patterns to verify
that the deduplication logic correctly handles:
1. Duplicate epochs (same epoch logged multiple times)
2. Spurious epoch 0 entries (re-evaluations on resume)
"""

import pytest
import pandas as pd
from pathlib import Path
import sys
import tempfile

# Add code directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'code'))

from consolidate_model_results import consolidate_model_results
from fixtures.duplicate_data_generator import (
    create_test_model_directory,
    cleanup_test_directory
)


class TestDuplicateEpochRemoval:
    """Test duplicate epoch removal logic."""

    @pytest.fixture
    def temp_test_dir(self):
        """Create temporary directory for test models."""
        test_dir = Path(tempfile.mkdtemp(prefix='test_models_'))
        yield test_dir
        cleanup_test_directory(test_dir)

    def test_duplicate_epoch_removal_basic(self, temp_test_dir):
        """Test that duplicate epochs are removed, keeping last occurrence."""
        # Create test model with duplicate epoch 27
        model_dir = create_test_model_directory(
            base_dir=temp_test_dir,
            author='baum',
            seed=0,
            max_epochs=30,
            duplicate_epochs=[27]
        )

        # Read raw CSV to verify duplicate exists
        raw_df = pd.read_csv(model_dir / 'loss_logs.csv')
        epoch_27_train = raw_df[
            (raw_df['epochs_completed'] == 27) &
            (raw_df['loss_dataset'] == 'train')
        ]
        assert len(epoch_27_train) == 2, "Test data should have duplicate epoch 27"

        # Run consolidation
        df = consolidate_model_results(models_dir=str(temp_test_dir))

        # Check: epoch 27 appears exactly once per loss_dataset
        epoch_27_counts = df[df['epochs_completed'] == 27].groupby('loss_dataset').size()
        assert (epoch_27_counts == 1).all(), "Duplicate epochs not removed properly"

        # Check: we kept the LAST occurrence (should have slightly different loss values)
        consolidated_epoch_27_train = df[
            (df['epochs_completed'] == 27) &
            (df['loss_dataset'] == 'train')
        ]['loss_value'].iloc[0]

        # The last occurrence should be the second entry in raw data
        expected_loss = epoch_27_train.iloc[1]['loss_value']
        assert abs(consolidated_epoch_27_train - expected_loss) < 1e-6, \
            "Should keep last occurrence of duplicate epoch"

    def test_duplicate_multiple_epochs(self, temp_test_dir):
        """Test removal of multiple duplicate epochs."""
        # Create model with duplicates at epochs 15, 25, and 35
        model_dir = create_test_model_directory(
            base_dir=temp_test_dir,
            author='dickens',
            seed=1,
            max_epochs=40,
            duplicate_epochs=[15, 25, 35]
        )

        # Run consolidation
        df = consolidate_model_results(models_dir=str(temp_test_dir))

        # Check: each epoch appears exactly once per loss_dataset
        for epoch in [15, 25, 35]:
            epoch_counts = df[df['epochs_completed'] == epoch].groupby('loss_dataset').size()
            assert (epoch_counts == 1).all(), f"Duplicate epoch {epoch} not removed properly"

    def test_spurious_epoch_0_removal(self, temp_test_dir):
        """Test that spurious epoch 0 entries after resume are removed."""
        # Create model with spurious epoch 0 after epochs 15 and 25
        model_dir = create_test_model_directory(
            base_dir=temp_test_dir,
            author='thompson',
            seed=2,
            max_epochs=30,
            spurious_epoch_0_at=[15, 25]
        )

        # Read raw CSV to verify spurious epoch 0 exists
        raw_df = pd.read_csv(model_dir / 'loss_logs.csv')
        epoch_0_rows = raw_df[raw_df['epochs_completed'] == 0]
        # Should have: 11 legitimate + 11 after epoch 15 + 11 after epoch 25 = 33 rows
        assert len(epoch_0_rows) > 11, "Test data should have spurious epoch 0 entries"

        # Run consolidation
        df = consolidate_model_results(models_dir=str(temp_test_dir))

        # Check: only first set of epoch 0 entries remain (11 datasets)
        epoch_0_rows_clean = df[df['epochs_completed'] == 0]
        expected_rows = 11  # train + 8 authors + 2 oz datasets
        assert len(epoch_0_rows_clean) == expected_rows, \
            f"Expected {expected_rows} epoch 0 rows, got {len(epoch_0_rows_clean)}"

        # Check: epoch 0 rows appear early in the data (low indices)
        # After consolidation, indices are reset, but epoch 0 should still be at start
        first_epochs = df.head(20)['epochs_completed'].unique()
        assert 0 in first_epochs, "Epoch 0 should appear at start of data"

    def test_combined_issues(self, temp_test_dir):
        """Test handling of both duplicate epochs and spurious epoch 0."""
        # Create model with both issues
        model_dir = create_test_model_directory(
            base_dir=temp_test_dir,
            author='melville',
            seed=3,
            max_epochs=35,
            duplicate_epochs=[27, 30],
            spurious_epoch_0_at=[15, 25]
        )

        # Read raw CSV to count issues
        raw_df = pd.read_csv(model_dir / 'loss_logs.csv')
        raw_rows = len(raw_df)

        # Run consolidation
        df = consolidate_model_results(models_dir=str(temp_test_dir))

        # Check: spurious epoch 0 removed
        assert len(df[df['epochs_completed'] == 0]) == 11, \
            "Spurious epoch 0 not removed"

        # Check: duplicate epochs removed
        for epoch in [27, 30]:
            epoch_counts = df[df['epochs_completed'] == epoch].groupby('loss_dataset').size()
            assert (epoch_counts == 1).all(), f"Duplicate epoch {epoch} not removed"

        # Check: we removed the expected number of rows
        # Spurious epoch 0: 2 sets × 11 = 22 rows
        # Duplicate epochs: 2 epochs × 12 rows (train + 11 eval) = 24 rows
        # Total expected removals: 22 + 24 = 46 rows
        expected_clean_rows = raw_rows - 46
        assert len(df) == expected_clean_rows, \
            f"Expected {expected_clean_rows} rows after cleanup, got {len(df)}"

    def test_multiple_models(self, temp_test_dir):
        """Test consolidation of multiple models with different duplicate patterns."""
        # Create 3 models with different issues
        create_test_model_directory(
            base_dir=temp_test_dir,
            author='baum',
            seed=0,
            max_epochs=30,
            duplicate_epochs=[27]
        )
        create_test_model_directory(
            base_dir=temp_test_dir,
            author='thompson',
            seed=1,
            max_epochs=30,
            spurious_epoch_0_at=[20]
        )
        create_test_model_directory(
            base_dir=temp_test_dir,
            author='dickens',
            seed=2,
            max_epochs=30,
            duplicate_epochs=[25],
            spurious_epoch_0_at=[15]
        )

        # Run consolidation
        df = consolidate_model_results(models_dir=str(temp_test_dir))

        # Check: 3 unique models
        assert df['model_name'].nunique() == 3, "Should have 3 models"

        # Check: each model has clean data (no duplicates)
        for (seed, author), group in df.groupby(['seed', 'train_author']):
            # Check no duplicate (epoch, loss_dataset) combinations
            duplicate_check = group.groupby(['epochs_completed', 'loss_dataset']).size()
            assert (duplicate_check == 1).all(), \
                f"Model {author} seed {seed} has duplicates"

            # Check epoch 0 appears exactly 11 times
            epoch_0_count = len(group[group['epochs_completed'] == 0])
            assert epoch_0_count == 11, \
                f"Model {author} seed {seed} has {epoch_0_count} epoch 0 entries (expected 11)"

    def test_no_false_positives(self, temp_test_dir):
        """Test that clean data is not modified."""
        # Create model without any duplicates
        model_dir = create_test_model_directory(
            base_dir=temp_test_dir,
            author='austen',
            seed=4,
            max_epochs=30,
            duplicate_epochs=None,
            spurious_epoch_0_at=None
        )

        # Read original data
        original_df = pd.read_csv(model_dir / 'loss_logs.csv')
        original_rows = len(original_df)

        # Run consolidation
        df = consolidate_model_results(models_dir=str(temp_test_dir))

        # Check: same number of rows (no removals)
        assert len(df) == original_rows, \
            f"Clean data was modified: {original_rows} -> {len(df)} rows"

        # Check: all original data present
        for _, row in original_df.iterrows():
            match = df[
                (df['seed'] == row['seed']) &
                (df['train_author'] == row['train_author']) &
                (df['epochs_completed'] == row['epochs_completed']) &
                (df['loss_dataset'] == row['loss_dataset'])
            ]
            assert len(match) == 1, \
                f"Row missing or duplicated: epoch {row['epochs_completed']}, dataset {row['loss_dataset']}"
            assert abs(match['loss_value'].iloc[0] - row['loss_value']) < 1e-6, \
                "Loss value changed"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
