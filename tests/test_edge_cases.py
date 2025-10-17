"""
Test edge cases for duplicate removal in consolidate_model_results.py

These tests verify correct behavior in edge cases:
- Empty loss_logs.csv files
- Only epoch 0 entries (no training)
- No duplicates (clean data)
- Multiple resume cycles with complex patterns
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


class TestEdgeCases:
    """Test edge cases for deduplication logic."""

    @pytest.fixture
    def temp_test_dir(self):
        """Create temporary directory for test models."""
        test_dir = Path(tempfile.mkdtemp(prefix='test_edge_cases_'))
        yield test_dir
        cleanup_test_directory(test_dir)

    def test_empty_loss_logs(self, temp_test_dir):
        """Test handling of empty loss_logs.csv files."""
        # Create model directory with empty CSV
        model_name = 'baum_tokenizer=gpt2_seed=0'
        model_dir = temp_test_dir / model_name
        model_dir.mkdir(parents=True)

        # Create empty CSV with just headers
        empty_csv = model_dir / 'loss_logs.csv'
        pd.DataFrame(columns=[
            'seed', 'train_author', 'epochs_completed', 'loss_dataset', 'loss_value'
        ]).to_csv(empty_csv, index=False)

        # Should not crash, just produce empty result
        df = consolidate_model_results(models_dir=str(temp_test_dir))
        assert len(df) == 0, "Expected empty DataFrame for empty CSV"

    def test_only_epoch_0(self, temp_test_dir):
        """Test model that only has initial evaluation (epoch 0)."""
        # Create model with only epoch 0
        model_dir = create_test_model_directory(
            base_dir=temp_test_dir,
            author='thompson',
            seed=5,
            max_epochs=0  # No training epochs
        )

        # Verify raw data has only epoch 0
        raw_df = pd.read_csv(model_dir / 'loss_logs.csv')
        assert (raw_df['epochs_completed'] == 0).all(), "Test data should only have epoch 0"

        # Run consolidation
        df = consolidate_model_results(models_dir=str(temp_test_dir))

        # All epoch 0 rows should be kept
        assert len(df[df['epochs_completed'] == 0]) == 11, \
            "Should keep all 11 epoch 0 entries"
        assert len(df[df['epochs_completed'] > 0]) == 0, \
            "Should have no training epochs"

    def test_no_duplicates_unchanged(self, temp_test_dir):
        """Test that clean data is not modified."""
        # Create model without any duplicates
        model_dir = create_test_model_directory(
            base_dir=temp_test_dir,
            author='austen',
            seed=6,
            max_epochs=40,
            duplicate_epochs=None,
            spurious_epoch_0_at=None
        )

        # Read original data
        original_df = pd.read_csv(model_dir / 'loss_logs.csv')
        original_rows = len(original_df)

        # Run consolidation
        df = consolidate_model_results(models_dir=str(temp_test_dir))

        # Should have same number of rows (no removals)
        assert len(df) == original_rows, \
            f"Clean data was modified: {original_rows} -> {len(df)} rows"

        # Verify data integrity: each (epoch, loss_dataset) appears exactly once
        duplicates = df.groupby(['epochs_completed', 'loss_dataset']).size()
        assert (duplicates == 1).all(), "Clean data should have no duplicates"

    def test_multiple_resume_cycles(self, temp_test_dir):
        """Test handling of multiple training interruptions."""
        # Create CSV with multiple resume patterns:
        # - Duplicate epoch 10 (first resume)
        # - Spurious epoch 0 after epoch 10
        # - Duplicate epoch 20 (second resume)
        # - Spurious epoch 0 after epoch 20
        # - Duplicate epoch 30 (third resume)
        # - Spurious epoch 0 after epoch 30
        model_dir = create_test_model_directory(
            base_dir=temp_test_dir,
            author='melville',
            seed=7,
            max_epochs=35,
            duplicate_epochs=[10, 20, 30],
            spurious_epoch_0_at=[10, 20, 30]
        )

        # Read raw data to verify test setup
        raw_df = pd.read_csv(model_dir / 'loss_logs.csv')
        raw_epoch_0_count = len(raw_df[raw_df['epochs_completed'] == 0])
        assert raw_epoch_0_count > 11, "Should have multiple epoch 0 sets"

        # Run consolidation
        df = consolidate_model_results(models_dir=str(temp_test_dir))

        # Check: only one set of epoch 0 entries (11 rows)
        epoch_0_count = len(df[df['epochs_completed'] == 0])
        assert epoch_0_count == 11, \
            f"Expected 11 epoch 0 rows, got {epoch_0_count}"

        # Check: epochs 10, 20, 30 appear exactly once per dataset
        for epoch in [10, 20, 30]:
            epoch_counts = df[df['epochs_completed'] == epoch].groupby('loss_dataset').size()
            assert (epoch_counts == 1).all(), \
                f"Duplicate epoch {epoch} not removed properly"
            assert len(epoch_counts) == 12, \
                f"Epoch {epoch} should have 12 entries (train + 11 eval)"

        # Check: epoch sequence is complete (no gaps)
        train_epochs = sorted(df[df['loss_dataset'] == 'train']['epochs_completed'].unique())
        expected_epochs = list(range(1, 36))  # 1 to 35
        assert train_epochs == expected_epochs, \
            f"Epoch sequence has gaps: {train_epochs}"

    def test_single_epoch_training(self, temp_test_dir):
        """Test model that trained for only one epoch."""
        # Create model with just epoch 0 and epoch 1
        model_dir = create_test_model_directory(
            base_dir=temp_test_dir,
            author='fitzgerald',
            seed=8,
            max_epochs=1
        )

        # Run consolidation
        df = consolidate_model_results(models_dir=str(temp_test_dir))

        # Should have epoch 0 (11 rows) + epoch 1 (12 rows) = 23 rows
        assert len(df) == 23, f"Expected 23 rows, got {len(df)}"

        # Check epoch distribution
        epoch_0_count = len(df[df['epochs_completed'] == 0])
        epoch_1_count = len(df[df['epochs_completed'] == 1])
        assert epoch_0_count == 11, f"Expected 11 epoch 0 rows, got {epoch_0_count}"
        assert epoch_1_count == 12, f"Expected 12 epoch 1 rows, got {epoch_1_count}"

    def test_very_long_training(self, temp_test_dir):
        """Test model with many epochs to ensure scalability."""
        # Create model with 500 epochs (realistic training length)
        model_dir = create_test_model_directory(
            base_dir=temp_test_dir,
            author='wells',
            seed=9,
            max_epochs=500,
            duplicate_epochs=[250],  # One duplicate in the middle
            spurious_epoch_0_at=[250]  # One spurious epoch 0
        )

        # Run consolidation
        df = consolidate_model_results(models_dir=str(temp_test_dir))

        # Check: epoch sequence is complete
        train_epochs = sorted(df[df['loss_dataset'] == 'train']['epochs_completed'].unique())
        expected_epochs = list(range(1, 501))
        assert train_epochs == expected_epochs, "Epoch sequence incomplete for long training"

        # Check: only one epoch 0 set
        assert len(df[df['epochs_completed'] == 0]) == 11, \
            "Should have exactly 11 epoch 0 entries"

        # Check: no duplicates at epoch 250
        epoch_250_counts = df[df['epochs_completed'] == 250].groupby('loss_dataset').size()
        assert (epoch_250_counts == 1).all(), "Duplicate at epoch 250 not removed"

    def test_missing_loss_logs_file(self, temp_test_dir):
        """Test handling when loss_logs.csv is missing."""
        # Create model directory without loss_logs.csv
        model_name = 'twain_tokenizer=gpt2_seed=0'
        model_dir = temp_test_dir / model_name
        model_dir.mkdir(parents=True)
        # Don't create loss_logs.csv

        # Should handle gracefully with warning (captured in output)
        df = consolidate_model_results(models_dir=str(temp_test_dir))

        # Should return empty DataFrame (no data to consolidate)
        assert len(df) == 0, "Should return empty DataFrame when no valid models"

    def test_variant_models(self, temp_test_dir):
        """Test deduplication works for variant models."""
        # Create variant models with duplicates
        create_test_model_directory(
            base_dir=temp_test_dir,
            author='baum',
            seed=0,
            variant='content',
            max_epochs=30,
            duplicate_epochs=[27],
            spurious_epoch_0_at=[15]
        )
        create_test_model_directory(
            base_dir=temp_test_dir,
            author='thompson',
            seed=0,
            variant='function',
            max_epochs=30,
            duplicate_epochs=[25],
            spurious_epoch_0_at=[20]
        )

        # Run consolidation for content variant
        df_content = consolidate_model_results(
            models_dir=str(temp_test_dir),
            variant='content'
        )

        # Check: only content models included
        assert df_content['variant'].nunique() == 1, "Should have only one variant"
        assert (df_content['variant'] == 'content').all(), "Should only have content variant"

        # Check: deduplication worked
        assert len(df_content[df_content['epochs_completed'] == 0]) == 11, \
            "Content variant should have 11 epoch 0 entries"
        epoch_27_counts = df_content[df_content['epochs_completed'] == 27].groupby('loss_dataset').size()
        assert (epoch_27_counts == 1).all(), "Duplicate epoch 27 not removed in content variant"

        # Run consolidation for function variant
        df_function = consolidate_model_results(
            models_dir=str(temp_test_dir),
            variant='function'
        )

        # Check: only function models included
        assert (df_function['variant'] == 'function').all(), "Should only have function variant"

        # Check: deduplication worked
        assert len(df_function[df_function['epochs_completed'] == 0]) == 11, \
            "Function variant should have 11 epoch 0 entries"
        epoch_25_counts = df_function[df_function['epochs_completed'] == 25].groupby('loss_dataset').size()
        assert (epoch_25_counts == 1).all(), "Duplicate epoch 25 not removed in function variant"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
