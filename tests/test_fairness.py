#!/usr/bin/env python
"""
Test fairness-based loss thresholding with REAL DATA.

These tests use actual model results from the function-only variant to
verify correctness. NO MOCKS OR SIMULATIONS are used - all tests run
against real trained model data.
"""

import pytest
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import tempfile
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from llm_stylometry.analysis.fairness import (
    compute_fairness_threshold,
    apply_fairness_threshold
)
from llm_stylometry.visualization import (
    generate_all_losses_figure,
    generate_stripplot_figure,
    generate_loss_heatmap_figure,
    generate_3d_mds_figure,
    generate_oz_losses_figure
)


class TestFairnessThreshold:
    """Test fairness threshold computation with real data."""

    @classmethod
    def setup_class(cls):
        """Load real function variant data."""
        cls.data_path = Path(__file__).parent.parent / "data" / "model_results_function.pkl"

        if not cls.data_path.exists():
            pytest.skip(f"Function variant data not found at {cls.data_path}")

        cls.df = pd.read_pickle(cls.data_path)

        # Verify required columns exist
        required_cols = ['loss_dataset', 'epochs_completed', 'loss_value', 'train_author', 'seed']
        missing = [c for c in required_cols if c not in cls.df.columns]
        if missing:
            pytest.fail(f"Missing required columns in data: {missing}")

    def test_data_loaded_correctly(self):
        """Verify real data is loaded and has expected structure."""
        assert len(self.df) > 0, "No data loaded"
        assert 'variant' in self.df.columns, "Variant column missing"
        assert (self.df['variant'] == 'function').all(), "Not all rows are function variant"

        # Verify we have 8 authors × 10 seeds = 80 models
        unique_models = self.df.groupby(['train_author', 'seed']).ngroups
        assert unique_models == 80, f"Expected 80 models, found {unique_models}"

    def test_compute_threshold_real_data(self):
        """Test threshold computation with real function variant data."""
        threshold = compute_fairness_threshold(self.df, min_epochs=500)

        # Verify threshold is a valid number
        assert isinstance(threshold, float), f"Threshold should be float, got {type(threshold)}"
        assert not np.isnan(threshold), "Threshold is NaN"
        assert threshold > 0, f"Threshold should be positive, got {threshold}"

        # Based on our analysis, threshold should be around 1.27 (Austen's minimum)
        assert 1.20 < threshold < 1.35, f"Threshold {threshold:.4f} outside expected range [1.20, 1.35]"

        print(f"\n✓ Computed fairness threshold: {threshold:.4f}")

    def test_threshold_is_maximum_of_minimums(self):
        """Verify threshold is the maximum of all models' minimum losses."""
        threshold = compute_fairness_threshold(self.df, min_epochs=500)

        # Manually compute what the threshold should be
        train_df = self.df[self.df['loss_dataset'] == 'train']
        train_df = train_df[train_df['epochs_completed'] <= 500]

        min_losses = train_df.groupby(['train_author', 'seed'])['loss_value'].min()
        expected_threshold = min_losses.max()

        assert abs(threshold - expected_threshold) < 0.0001, \
            f"Threshold {threshold:.4f} != expected {expected_threshold:.4f}"

        print(f"✓ Threshold correctly computed as max of minimums")
        print(f"  Min across models: {min_losses.min():.4f}")
        print(f"  Max across models: {min_losses.max():.4f}")

    def test_apply_threshold_truncates_correctly(self):
        """Test that apply_fairness_threshold truncates data correctly."""
        threshold = compute_fairness_threshold(self.df, min_epochs=500)
        df_fair = apply_fairness_threshold(self.df, threshold, use_first_crossing=True)

        # Verify truncated data is smaller or equal (some models might use all 500 epochs)
        assert len(df_fair) <= len(self.df), "Truncated data should have fewer or equal rows"

        # Verify each model is truncated at first epoch where loss <= threshold
        for (author, seed), group in df_fair.groupby(['train_author', 'seed']):
            train_data = group[group['loss_dataset'] == 'train'].sort_values('epochs_completed')

            if len(train_data) == 0:
                continue

            # Get final epoch for this model
            final_epoch = train_data['epochs_completed'].max()
            final_loss = train_data[train_data['epochs_completed'] == final_epoch]['loss_value'].values[0]

            # Final loss should be <= threshold (this is where we truncated)
            assert final_loss <= threshold + 0.001, \
                f"Model {author} seed {seed}: final loss {final_loss:.4f} > threshold {threshold:.4f}"

        print(f"✓ Truncated {len(self.df)} -> {len(df_fair)} rows ({100*len(df_fair)/len(self.df):.1f}%)")
        print(f"✓ All models truncated at first epoch where loss ≤ {threshold:.4f}")

    def test_evaluation_datasets_truncated_with_training(self):
        """Verify evaluation datasets are truncated at same epoch as training."""
        threshold = compute_fairness_threshold(self.df, min_epochs=500)
        df_fair = apply_fairness_threshold(self.df, threshold, use_first_crossing=True)

        # For each model, verify training and evaluation datasets end at same epoch
        for (author, seed), group in df_fair.groupby(['train_author', 'seed']):
            train_max_epoch = group[group['loss_dataset'] == 'train']['epochs_completed'].max()

            for dataset in group['loss_dataset'].unique():
                if dataset != 'train':
                    eval_max_epoch = group[group['loss_dataset'] == dataset]['epochs_completed'].max()
                    assert eval_max_epoch == train_max_epoch, \
                        f"Model {author} seed {seed}: eval dataset {dataset} ends at epoch {eval_max_epoch}, train ends at {train_max_epoch}"

        print(f"✓ All evaluation datasets truncated at same epoch as training")

    def test_data_integrity_preserved(self):
        """Verify that truncation preserves data integrity."""
        threshold = compute_fairness_threshold(self.df, min_epochs=500)
        df_fair = apply_fairness_threshold(self.df, threshold, use_first_crossing=True)

        # Verify all columns preserved
        assert set(df_fair.columns) == set(self.df.columns), "Columns changed"

        # Verify all models still present
        original_models = set(self.df.groupby(['train_author', 'seed']).groups.keys())
        truncated_models = set(df_fair.groupby(['train_author', 'seed']).groups.keys())
        assert truncated_models == original_models, "Some models missing after truncation"

        # Verify no new data created
        assert df_fair['loss_value'].min() >= self.df['loss_value'].min(), "Minimum loss decreased"

        print(f"✓ Data integrity preserved (80 models present, all columns intact)")

    def test_models_converge_at_different_epochs(self):
        """Verify different models are truncated at different epochs (fairness in action)."""
        threshold = compute_fairness_threshold(self.df, min_epochs=500)
        df_fair = apply_fairness_threshold(self.df, threshold, use_first_crossing=True)

        # Get max epoch for each model after truncation
        max_epochs = df_fair.groupby(['train_author', 'seed'])['epochs_completed'].max()

        # Should have variety of max epochs
        unique_max_epochs = max_epochs.unique()
        assert len(unique_max_epochs) > 1, "All models truncated at same epoch (unexpected)"

        # Some models should reach threshold early, others late
        earliest = max_epochs.min()
        latest = max_epochs.max()
        assert latest - earliest > 50, \
            f"Expected >50 epoch spread, got {latest - earliest}"

        print(f"✓ Models truncated at different epochs:")
        print(f"  Earliest: epoch {earliest}")
        print(f"  Latest: epoch {latest}")
        print(f"  Spread: {latest - earliest} epochs")


class TestFairnessEdgeCases:
    """Test edge cases with constructed or filtered data."""

    @classmethod
    def setup_class(cls):
        """Load real data for edge case testing."""
        cls.data_path = Path(__file__).parent.parent / "data" / "model_results_function.pkl"

        if not cls.data_path.exists():
            pytest.skip(f"Function variant data not found at {cls.data_path}")

        cls.df_full = pd.read_pickle(cls.data_path)

    def test_single_model_subset(self):
        """Test with only one model's data."""
        # Take just one model (austen, seed 0)
        df_single = self.df_full[
            (self.df_full['train_author'] == 'austen') &
            (self.df_full['seed'] == 0)
        ].copy()

        threshold = compute_fairness_threshold(df_single, min_epochs=500)
        assert not np.isnan(threshold), "Threshold should work with single model"
        assert threshold > 0, "Threshold should be positive"

        df_fair = apply_fairness_threshold(df_single, threshold)
        assert len(df_fair) <= len(df_single), "Should truncate or keep same"

        print(f"✓ Single model test passed (threshold={threshold:.4f})")

    def test_insufficient_epochs(self):
        """Test behavior when requesting more epochs than available."""
        # Request 1000 epochs when max is 500
        threshold = compute_fairness_threshold(self.df_full, min_epochs=1000)
        assert not np.isnan(threshold), "Should still compute threshold"

        print(f"✓ Handled insufficient epochs gracefully")

    def test_missing_column_raises_error(self):
        """Test that missing columns raise appropriate errors."""
        df_bad = self.df_full.drop(columns=['loss_value'])

        with pytest.raises(ValueError, match="Missing required columns"):
            compute_fairness_threshold(df_bad)

        print(f"✓ Missing column error raised correctly")

    def test_no_training_data_raises_error(self):
        """Test that missing training data raises error."""
        df_no_train = self.df_full[self.df_full['loss_dataset'] != 'train'].copy()

        with pytest.raises(ValueError, match="No training loss data found"):
            compute_fairness_threshold(df_no_train)

        print(f"✓ No training data error raised correctly")


class TestFairnessWithVisualization:
    """Test that fairness works correctly with actual figure generation."""

    @classmethod
    def setup_class(cls):
        """Setup for visualization tests."""
        cls.data_path = Path(__file__).parent.parent / "data" / "model_results_function.pkl"

        if not cls.data_path.exists():
            pytest.skip(f"Function variant data not found at {cls.data_path}")

        cls.temp_dir = tempfile.mkdtemp()

    def test_all_losses_figure_with_fairness(self):
        """Test Figure 1A generation with fairness enabled."""
        output_path = Path(self.temp_dir) / "test_all_losses_fair.pdf"
        # Note: variant suffix will be added automatically
        expected_path = Path(self.temp_dir) / "test_all_losses_fair_function.pdf"

        # Generate with fairness
        fig = generate_all_losses_figure(
            data_path=str(self.data_path),
            output_path=str(output_path),
            variant='function',
            apply_fairness=True,
            show_legend=False
        )

        assert fig is not None, "Figure generation failed"
        assert expected_path.exists(), f"PDF not created at {expected_path}"
        assert expected_path.stat().st_size > 1000, "PDF too small"

        plt.close(fig)
        print(f"✓ All losses figure with fairness generated successfully")

    def test_stripplot_figure_with_fairness(self):
        """Test Figure 1B generation with fairness enabled."""
        output_path = Path(self.temp_dir) / "test_stripplot_fair.pdf"
        expected_path = Path(self.temp_dir) / "test_stripplot_fair_function.pdf"

        fig = generate_stripplot_figure(
            data_path=str(self.data_path),
            output_path=str(output_path),
            variant='function',
            apply_fairness=True
        )

        assert fig is not None, "Figure generation failed"
        assert expected_path.exists(), f"PDF not created at {expected_path}"

        plt.close(fig)
        print(f"✓ Stripplot figure with fairness generated successfully")

    def test_heatmap_figure_with_fairness(self):
        """Test Figure 3 generation with fairness enabled."""
        output_path = Path(self.temp_dir) / "test_heatmap_fair.pdf"
        expected_path = Path(self.temp_dir) / "test_heatmap_fair_function.pdf"

        fig = generate_loss_heatmap_figure(
            data_path=str(self.data_path),
            output_path=str(output_path),
            variant='function',
            apply_fairness=True
        )

        assert fig is not None, "Figure generation failed"
        assert expected_path.exists(), f"PDF not created at {expected_path}"

        plt.close(fig)
        print(f"✓ Heatmap figure with fairness generated successfully")

    def test_mds_figure_with_fairness(self):
        """Test Figure 4 generation with fairness enabled."""
        output_path = Path(self.temp_dir) / "test_mds_fair.pdf"
        expected_path = Path(self.temp_dir) / "test_mds_fair_function.pdf"

        fig = generate_3d_mds_figure(
            data_path=str(self.data_path),
            output_path=str(output_path),
            variant='function',
            apply_fairness=True
        )

        assert fig is not None, "Figure generation failed"
        assert expected_path.exists(), f"PDF not created at {expected_path}"

        plt.close(fig)
        print(f"✓ MDS figure with fairness generated successfully")

    def test_oz_losses_figure_with_fairness(self):
        """Test Figure 5 skipped for variants (Oz analysis is baseline-only)."""
        output_path = Path(self.temp_dir) / "test_oz_fair.pdf"

        # Figure 5 should return None for variants (Oz analysis is baseline-only)
        fig = generate_oz_losses_figure(
            data_path=str(self.data_path),
            output_path=str(output_path),
            variant='function',
            apply_fairness=True
        )

        assert fig is None, "Figure 5 should return None for variants (Oz analysis is baseline-only)"
        assert not output_path.exists(), "PDF should not be created for variant"

        print(f"✓ Oz losses figure correctly skipped for variant (baseline-only analysis)")

    def test_fairness_disabled_flag(self):
        """Test that apply_fairness=False bypasses fairness thresholding."""
        output_fair = Path(self.temp_dir) / "test_fair_enabled.pdf"
        output_nofair = Path(self.temp_dir) / "test_fair_disabled.pdf"
        # Variant suffix will be added
        expected_fair = Path(self.temp_dir) / "test_fair_enabled_function.pdf"
        expected_nofair = Path(self.temp_dir) / "test_fair_disabled_function.pdf"

        # Generate with fairness
        fig_fair = generate_all_losses_figure(
            data_path=str(self.data_path),
            output_path=str(output_fair),
            variant='function',
            apply_fairness=True
        )

        # Generate without fairness
        fig_nofair = generate_all_losses_figure(
            data_path=str(self.data_path),
            output_path=str(output_nofair),
            variant='function',
            apply_fairness=False
        )

        # Both should succeed but produce different files
        assert expected_fair.exists() and expected_nofair.exists(), "PDFs not created"

        # Files should be different sizes (different data)
        size_fair = expected_fair.stat().st_size
        size_nofair = expected_nofair.stat().st_size
        assert size_fair != size_nofair, "Fairness flag had no effect on output"

        plt.close(fig_fair)
        plt.close(fig_nofair)
        print(f"✓ apply_fairness flag works correctly")
        print(f"  With fairness: {size_fair} bytes")
        print(f"  Without fairness: {size_nofair} bytes")

    @classmethod
    def teardown_class(cls):
        """Clean up temporary files."""
        import shutil
        if hasattr(cls, 'temp_dir') and Path(cls.temp_dir).exists():
            shutil.rmtree(cls.temp_dir)


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "--tb=short", "-s"])
