#!/usr/bin/env python
"""Test t-test visualization edge cases with real data scenarios."""

import pytest
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tempfile
import logging

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from llm_stylometry.visualization.t_tests import (
    generate_t_test_figure,
    generate_t_test_avg_figure,
    calculate_t_statistics
)


class TestTTestEdgeCases:
    """Test t-test visualization with edge cases that previously caused NaN/Inf issues."""

    @classmethod
    def setup_class(cls):
        """Set up test fixtures."""
        cls.temp_dir = tempfile.mkdtemp()

    def create_test_dataframe(self, scenario):
        """
        Create test dataframe for specific edge case scenario.

        Args:
            scenario: String identifying the test scenario

        Returns:
            pandas DataFrame with structure matching model_results.pkl
        """
        authors = ["Baum", "Thompson"]

        if scenario == "all_nan":
            # Scenario: Single sample per group (n=1) - produces NaN t-statistics
            data = []
            for author in authors:
                for epoch in range(1, 6):
                    # Only 1 sample per group - insufficient for t-test
                    data.append({
                        "train_author": author.lower(),
                        "loss_dataset": author.lower(),
                        "epochs_completed": epoch,
                        "loss_value": 2.5,
                        "model_name": f"{author.lower()}_seed=1",
                        "seed": 1
                    })
                    # Other author - also only 1 sample
                    other = "thompson" if author == "Baum" else "baum"
                    data.append({
                        "train_author": author.lower(),
                        "loss_dataset": other,
                        "epochs_completed": epoch,
                        "loss_value": 3.0,
                        "model_name": f"{author.lower()}_seed=1",
                        "seed": 1
                    })
            return pd.DataFrame(data)

        elif scenario == "mixed_nan":
            # Scenario: Mix of valid (n>=2) and invalid (n=1) data
            data = []
            for author in authors:
                for epoch in range(1, 6):
                    # Early epochs: Only 1 sample (will produce NaN)
                    n_samples = 1 if epoch <= 2 else 3
                    for seed in range(1, n_samples + 1):
                        data.append({
                            "train_author": author.lower(),
                            "loss_dataset": author.lower(),
                            "epochs_completed": epoch,
                            "loss_value": 2.5 + np.random.normal(0, 0.1),
                            "model_name": f"{author.lower()}_seed={seed}",
                            "seed": seed
                        })
                        other = "thompson" if author == "Baum" else "baum"
                        data.append({
                            "train_author": author.lower(),
                            "loss_dataset": other,
                            "epochs_completed": epoch,
                            "loss_value": 3.0 + np.random.normal(0, 0.1),
                            "model_name": f"{author.lower()}_seed={seed}",
                            "seed": seed
                        })
            return pd.DataFrame(data)

        elif scenario == "all_infinite":
            # Scenario: Zero variance in both groups - produces Inf t-statistics
            data = []
            for author in authors:
                for epoch in range(1, 6):
                    for seed in range(1, 4):
                        # Identical values - zero variance
                        data.append({
                            "train_author": author.lower(),
                            "loss_dataset": author.lower(),
                            "epochs_completed": epoch,
                            "loss_value": 2.5,  # Exactly identical
                            "model_name": f"{author.lower()}_seed={seed}",
                            "seed": seed
                        })
                        other = "thompson" if author == "Baum" else "baum"
                        data.append({
                            "train_author": author.lower(),
                            "loss_dataset": other,
                            "epochs_completed": epoch,
                            "loss_value": 3.0,  # Exactly identical
                            "model_name": f"{author.lower()}_seed={seed}",
                            "seed": seed
                        })
            return pd.DataFrame(data)

        elif scenario == "mixed_infinite":
            # Scenario: Mix of zero-variance and normal-variance groups
            data = []
            for author in authors:
                for epoch in range(1, 6):
                    for seed in range(1, 4):
                        # Early epochs: zero variance, later epochs: normal variance
                        if epoch <= 2:
                            loss_true = 2.5
                            loss_other = 3.0
                        else:
                            loss_true = 2.5 + np.random.normal(0, 0.1)
                            loss_other = 3.0 + np.random.normal(0, 0.1)

                        data.append({
                            "train_author": author.lower(),
                            "loss_dataset": author.lower(),
                            "epochs_completed": epoch,
                            "loss_value": loss_true,
                            "model_name": f"{author.lower()}_seed={seed}",
                            "seed": seed
                        })
                        other = "thompson" if author == "Baum" else "baum"
                        data.append({
                            "train_author": author.lower(),
                            "loss_dataset": other,
                            "epochs_completed": epoch,
                            "loss_value": loss_other,
                            "model_name": f"{author.lower()}_seed={seed}",
                            "seed": seed
                        })
            return pd.DataFrame(data)

        elif scenario == "empty_data":
            # Scenario: No data at all for certain epochs
            data = []
            for author in authors:
                # Only epochs 1, 3, 5 have data; epochs 2, 4 are missing
                for epoch in [1, 3, 5]:
                    for seed in range(1, 4):
                        data.append({
                            "train_author": author.lower(),
                            "loss_dataset": author.lower(),
                            "epochs_completed": epoch,
                            "loss_value": 2.5 + np.random.normal(0, 0.1),
                            "model_name": f"{author.lower()}_seed={seed}",
                            "seed": seed
                        })
                        other = "thompson" if author == "Baum" else "baum"
                        data.append({
                            "train_author": author.lower(),
                            "loss_dataset": other,
                            "epochs_completed": epoch,
                            "loss_value": 3.0 + np.random.normal(0, 0.1),
                            "model_name": f"{author.lower()}_seed={seed}",
                            "seed": seed
                        })
            return pd.DataFrame(data)

        elif scenario == "extreme_outliers":
            # Scenario: Data with extreme outliers
            data = []
            for author in authors:
                for epoch in range(1, 6):
                    for seed in range(1, 4):
                        # Add extreme outlier at epoch 3
                        if epoch == 3 and seed == 1:
                            loss_true = 100.0  # Extreme outlier
                            loss_other = 150.0
                        else:
                            loss_true = 2.5 + np.random.normal(0, 0.1)
                            loss_other = 3.0 + np.random.normal(0, 0.1)

                        data.append({
                            "train_author": author.lower(),
                            "loss_dataset": author.lower(),
                            "epochs_completed": epoch,
                            "loss_value": loss_true,
                            "model_name": f"{author.lower()}_seed={seed}",
                            "seed": seed
                        })
                        other = "thompson" if author == "Baum" else "baum"
                        data.append({
                            "train_author": author.lower(),
                            "loss_dataset": other,
                            "epochs_completed": epoch,
                            "loss_value": loss_other,
                            "model_name": f"{author.lower()}_seed={seed}",
                            "seed": seed
                        })
            return pd.DataFrame(data)

        elif scenario == "small_samples":
            # Scenario: Very small sample sizes (n=2, minimum for t-test)
            data = []
            for author in authors:
                for epoch in range(1, 6):
                    # Exactly 2 samples - minimum required
                    for seed in range(1, 3):
                        data.append({
                            "train_author": author.lower(),
                            "loss_dataset": author.lower(),
                            "epochs_completed": epoch,
                            "loss_value": 2.5 + np.random.normal(0, 0.1),
                            "model_name": f"{author.lower()}_seed={seed}",
                            "seed": seed
                        })
                        other = "thompson" if author == "Baum" else "baum"
                        data.append({
                            "train_author": author.lower(),
                            "loss_dataset": other,
                            "epochs_completed": epoch,
                            "loss_value": 3.0 + np.random.normal(0, 0.1),
                            "model_name": f"{author.lower()}_seed={seed}",
                            "seed": seed
                        })
            return pd.DataFrame(data)

        elif scenario == "normal":
            # Scenario: Normal valid data for baseline comparison
            data = []
            for author in authors:
                for epoch in range(1, 101):  # More epochs
                    for seed in range(1, 11):  # 10 seeds
                        data.append({
                            "train_author": author.lower(),
                            "loss_dataset": author.lower(),
                            "epochs_completed": epoch,
                            "loss_value": 2.5 + np.random.normal(0, 0.1),
                            "model_name": f"{author.lower()}_seed={seed}",
                            "seed": seed
                        })
                        other = "thompson" if author == "Baum" else "baum"
                        data.append({
                            "train_author": author.lower(),
                            "loss_dataset": other,
                            "epochs_completed": epoch,
                            "loss_value": 3.0 + np.random.normal(0, 0.1),
                            "model_name": f"{author.lower()}_seed={seed}",
                            "seed": seed
                        })
            return pd.DataFrame(data)

        else:
            raise ValueError(f"Unknown scenario: {scenario}")

    def test_all_nan_t_statistics(self):
        """Test 1: All t-statistics are NaN (single sample per group)."""
        df = self.create_test_dataframe("all_nan")
        data_path = Path(self.temp_dir) / "test_all_nan.pkl"
        df.to_pickle(data_path)

        output_path = Path(self.temp_dir) / "test_all_nan.pdf"

        # Should not raise ValueError
        fig = generate_t_test_figure(
            data_path=str(data_path),
            output_path=str(output_path),
            show_legend=False
        )

        assert fig is not None, "Figure should be created even with all NaN values"
        assert output_path.exists(), "Output file should be created"

        # Verify axis limits are valid
        ax = fig.get_axes()[0]
        y_min, y_max = ax.get_ylim()
        assert np.isfinite(y_min) and np.isfinite(y_max), "Axis limits should be finite"
        assert y_min < y_max, "Y-min should be less than y-max"

        plt.close(fig)

    def test_mixed_nan_and_valid(self):
        """Test 2: Mix of NaN and valid t-statistics."""
        df = self.create_test_dataframe("mixed_nan")
        data_path = Path(self.temp_dir) / "test_mixed_nan.pkl"
        df.to_pickle(data_path)

        output_path = Path(self.temp_dir) / "test_mixed_nan.pdf"

        fig = generate_t_test_figure(
            data_path=str(data_path),
            output_path=str(output_path),
            show_legend=False
        )

        assert fig is not None, "Figure should be created with mixed NaN/valid"
        assert output_path.exists()

        # Verify axis limits based on valid data only
        ax = fig.get_axes()[0]
        y_min, y_max = ax.get_ylim()
        assert np.isfinite(y_min) and np.isfinite(y_max)
        assert y_min < y_max

        plt.close(fig)

    def test_all_infinite_t_statistics(self):
        """Test 3: All t-statistics are Inf (zero variance)."""
        df = self.create_test_dataframe("all_infinite")
        data_path = Path(self.temp_dir) / "test_all_inf.pkl"
        df.to_pickle(data_path)

        output_path = Path(self.temp_dir) / "test_all_inf.pdf"

        # Should not raise ValueError
        fig = generate_t_test_figure(
            data_path=str(data_path),
            output_path=str(output_path),
            show_legend=False
        )

        assert fig is not None, "Figure should be created even with all Inf values"
        assert output_path.exists()

        # With all Inf values, should use default limits
        ax = fig.get_axes()[0]
        y_min, y_max = ax.get_ylim()
        assert np.isfinite(y_min) and np.isfinite(y_max)
        assert y_min < y_max

        plt.close(fig)

    def test_mixed_infinite_and_valid(self):
        """Test 4: Mix of Inf and valid t-statistics."""
        df = self.create_test_dataframe("mixed_infinite")
        data_path = Path(self.temp_dir) / "test_mixed_inf.pkl"
        df.to_pickle(data_path)

        output_path = Path(self.temp_dir) / "test_mixed_inf.pdf"

        fig = generate_t_test_figure(
            data_path=str(data_path),
            output_path=str(output_path),
            show_legend=False
        )

        assert fig is not None, "Figure should handle mixed Inf/valid"
        assert output_path.exists()

        ax = fig.get_axes()[0]
        y_min, y_max = ax.get_ylim()
        assert np.isfinite(y_min) and np.isfinite(y_max)
        assert y_min < y_max

        plt.close(fig)

    def test_empty_data_groups(self):
        """Test 5: Some epochs have no data at all."""
        df = self.create_test_dataframe("empty_data")
        data_path = Path(self.temp_dir) / "test_empty.pkl"
        df.to_pickle(data_path)

        output_path = Path(self.temp_dir) / "test_empty.pdf"

        fig = generate_t_test_figure(
            data_path=str(data_path),
            output_path=str(output_path),
            show_legend=False
        )

        assert fig is not None, "Figure should handle missing epochs"
        assert output_path.exists()

        ax = fig.get_axes()[0]
        y_min, y_max = ax.get_ylim()
        assert np.isfinite(y_min) and np.isfinite(y_max)
        assert y_min < y_max

        plt.close(fig)

    def test_extreme_outliers(self):
        """Test 6: Data with extreme outliers."""
        df = self.create_test_dataframe("extreme_outliers")
        data_path = Path(self.temp_dir) / "test_outliers.pkl"
        df.to_pickle(data_path)

        output_path = Path(self.temp_dir) / "test_outliers.pdf"

        fig = generate_t_test_figure(
            data_path=str(data_path),
            output_path=str(output_path),
            show_legend=False
        )

        assert fig is not None, "Figure should handle extreme outliers"
        assert output_path.exists()

        # Axis limits should still be reasonable despite outliers
        ax = fig.get_axes()[0]
        y_min, y_max = ax.get_ylim()
        assert np.isfinite(y_min) and np.isfinite(y_max)
        assert y_min < y_max

        plt.close(fig)

    def test_small_sample_sizes(self):
        """Test 7: Very small sample sizes (n=2, minimum required)."""
        df = self.create_test_dataframe("small_samples")
        data_path = Path(self.temp_dir) / "test_small_n.pkl"
        df.to_pickle(data_path)

        output_path = Path(self.temp_dir) / "test_small_n.pdf"

        fig = generate_t_test_figure(
            data_path=str(data_path),
            output_path=str(output_path),
            show_legend=False
        )

        assert fig is not None, "Figure should handle minimum sample size"
        assert output_path.exists()

        ax = fig.get_axes()[0]
        y_min, y_max = ax.get_ylim()
        assert np.isfinite(y_min) and np.isfinite(y_max)
        assert y_min < y_max

        plt.close(fig)

    def test_average_figure_all_nan(self):
        """Test 8: Average t-test figure with all NaN."""
        df = self.create_test_dataframe("all_nan")
        data_path = Path(self.temp_dir) / "test_avg_nan.pkl"
        df.to_pickle(data_path)

        output_path = Path(self.temp_dir) / "test_avg_nan.pdf"

        # Should not raise ValueError
        fig = generate_t_test_avg_figure(
            data_path=str(data_path),
            output_path=str(output_path),
            show_legend=False
        )

        assert fig is not None, "Average figure should handle all NaN"
        assert output_path.exists()

        ax = fig.get_axes()[0]
        y_min, y_max = ax.get_ylim()
        assert np.isfinite(y_min) and np.isfinite(y_max)
        assert y_min < y_max

        plt.close(fig)

    def test_average_figure_mixed(self):
        """Test 9: Average t-test figure with mixed valid/invalid data."""
        df = self.create_test_dataframe("mixed_nan")
        data_path = Path(self.temp_dir) / "test_avg_mixed.pkl"
        df.to_pickle(data_path)

        output_path = Path(self.temp_dir) / "test_avg_mixed.pdf"

        fig = generate_t_test_avg_figure(
            data_path=str(data_path),
            output_path=str(output_path),
            show_legend=False
        )

        assert fig is not None, "Average figure should handle mixed data"
        assert output_path.exists()

        ax = fig.get_axes()[0]
        y_min, y_max = ax.get_ylim()
        assert np.isfinite(y_min) and np.isfinite(y_max)
        assert y_min < y_max

        plt.close(fig)

    def test_logging_verification(self, caplog):
        """Test 10: Verify logging for data quality issues."""
        with caplog.at_level(logging.DEBUG):
            df = self.create_test_dataframe("all_nan")
            data_path = Path(self.temp_dir) / "test_logging.pkl"
            df.to_pickle(data_path)

            # Calculate t-statistics to trigger logging
            t_raws_df, _ = calculate_t_statistics(df, max_epochs=5)

            # Check for expected log messages about insufficient data
            log_messages = [record.message for record in caplog.records]

            # Should have debug messages about insufficient data
            insufficient_data_logs = [
                msg for msg in log_messages
                if "Insufficient data for t-test" in msg or "n_true=1" in msg
            ]

            assert len(insufficient_data_logs) > 0, \
                "Should log warnings about insufficient sample sizes"

            # Verify log message format
            for msg in insufficient_data_logs:
                assert "need at least 2 samples per group" in msg or "n_true=1" in msg, \
                    f"Log message should explain minimum sample requirement: {msg}"

    def test_normal_data_baseline(self):
        """Test 11: Verify normal data still works correctly."""
        df = self.create_test_dataframe("normal")
        data_path = Path(self.temp_dir) / "test_normal.pkl"
        df.to_pickle(data_path)

        output_path = Path(self.temp_dir) / "test_normal.pdf"

        # Should work without issues
        fig = generate_t_test_figure(
            data_path=str(data_path),
            output_path=str(output_path),
            show_legend=False
        )

        assert fig is not None
        assert output_path.exists()
        assert output_path.stat().st_size > 1000, "Normal data should produce reasonable plot"

        # Should have many valid t-statistics
        t_raws_df, _ = calculate_t_statistics(df, max_epochs=100)
        valid_t_values = t_raws_df['t_raw'].replace([np.inf, -np.inf], np.nan).dropna()
        assert len(valid_t_values) > 100, "Should have many valid t-statistics with normal data"

        plt.close(fig)

    @classmethod
    def teardown_class(cls):
        """Clean up temporary directory."""
        import shutil
        if hasattr(cls, 'temp_dir') and Path(cls.temp_dir).exists():
            shutil.rmtree(cls.temp_dir)


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])
