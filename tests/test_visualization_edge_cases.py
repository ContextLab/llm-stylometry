"""
Comprehensive tests for visualization edge cases.

Tests the handling of negative t-statistics and dynamic y-axis limits
in t-test figures. All tests use REAL data and REAL figure generation
(no mocks or simulations).

Related to issue #25: Austen models not appearing in function-only variant figures.
"""

import pytest
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import subprocess
import tempfile

from llm_stylometry.visualization import (
    generate_t_test_figure,
    generate_t_test_avg_figure,
)
from llm_stylometry.visualization.t_tests import calculate_t_statistics


class TestNegativeTStatistics:
    """Test that negative t-statistics are properly displayed."""

    def test_negative_t_statistics_visible(self):
        """
        Test that Austen's negative t-statistics are visible in function variant.

        Uses REAL data from data/model_results_function.pkl.
        Verifies that:
        1. Austen data is present in the figure
        2. Austen has negative t-statistic values
        3. Y-axis limits encompass all Austen values
        """
        data_path = 'data/model_results_function.pkl'

        # Check if data file exists
        if not Path(data_path).exists():
            pytest.skip(f"Data file {data_path} not found")

        # Generate figure with function variant
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_fig.pdf"
            fig = generate_t_test_figure(
                data_path=data_path,
                output_path=str(output_path),
                variant='function'
            )

            # Extract axes from figure
            ax = fig.axes[0]

            # Get y-axis limits
            y_min, y_max = ax.get_ylim()

            # Load data and calculate t-statistics to verify
            df = pd.read_pickle(data_path)
            df = df[df['variant'] == 'function'].copy()
            t_raws_df, _ = calculate_t_statistics(df)

            # Check Austen data exists
            austen_data = t_raws_df[t_raws_df['Author'] == 'Austen']
            assert len(austen_data) > 0, "Austen data should be present"

            # Check that Austen has some negative t-statistics
            austen_min = austen_data['t_raw'].min()
            assert austen_min < 0, f"Austen should have negative t-stats, got min={austen_min}"

            # Check that y-axis limits encompass Austen data
            austen_max = austen_data['t_raw'].max()
            assert y_min <= austen_min, f"Y-min ({y_min}) should be <= Austen min ({austen_min})"
            assert y_max >= austen_max, f"Y-max ({y_max}) should be >= Austen max ({austen_max})"

            # Verify lines were actually plotted
            lines = ax.get_lines()
            # Should have 8 authors + 1 threshold line
            assert len(lines) >= 8, f"Should have at least 8 author lines, got {len(lines)}"

            # Check that at least one line has negative y-values
            has_negative = False
            for line in lines:
                y_data = line.get_ydata()
                if np.any(y_data < 0):
                    has_negative = True
                    break
            assert has_negative, "At least one line should have negative y-values"

            plt.close(fig)


class TestYLimContainsAllData:
    """Test that y-axis limits contain all data for all variants."""

    @pytest.mark.parametrize("variant,data_file", [
        (None, 'data/model_results.pkl'),
        ('function', 'data/model_results_function.pkl'),
    ])
    def test_ylim_contains_all_data(self, variant, data_file):
        """
        Test that y-axis limits contain all t-statistic values.

        Tests both baseline and function variant with REAL data.
        """
        if not Path(data_file).exists():
            pytest.skip(f"Data file {data_file} not found")

        # Generate figure
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_fig.pdf"
            fig = generate_t_test_figure(
                data_path=data_file,
                output_path=str(output_path),
                variant=variant
            )

            # Extract axes and limits
            ax = fig.axes[0]
            y_min, y_max = ax.get_ylim()

            # Load data and calculate t-statistics
            df = pd.read_pickle(data_file)
            if variant is None:
                if 'variant' in df.columns:
                    df = df[df['variant'].isna()].copy()
            else:
                df = df[df['variant'] == variant].copy()

            t_raws_df, _ = calculate_t_statistics(df)

            # Check all t-statistics are within limits
            data_min = t_raws_df['t_raw'].min()
            data_max = t_raws_df['t_raw'].max()

            assert y_min <= data_min, f"Y-min ({y_min}) should be <= data min ({data_min})"
            assert y_max >= data_max, f"Y-max ({y_max}) should be >= data max ({data_max})"

            plt.close(fig)


class TestBaselineRegression:
    """Test that baseline figures remain unchanged (regression test)."""

    def test_baseline_remains_positive(self):
        """
        Test that baseline figures still show all positive t-statistics.

        This is a regression test to ensure our fix doesn't break
        the baseline case where all authors should have positive t-stats.
        """
        data_path = 'data/model_results.pkl'

        if not Path(data_path).exists():
            pytest.skip(f"Data file {data_path} not found")

        # Generate baseline figure
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_baseline.pdf"
            fig = generate_t_test_figure(
                data_path=data_path,
                output_path=str(output_path),
                variant=None  # Baseline
            )

            # Extract axes
            ax = fig.axes[0]
            y_min, y_max = ax.get_ylim()

            # Load data and calculate t-statistics
            df = pd.read_pickle(data_path)
            if 'variant' in df.columns:
                df = df[df['variant'].isna()].copy()

            t_raws_df, _ = calculate_t_statistics(df)

            # All t-statistics should be positive for baseline
            data_min = t_raws_df['t_raw'].min()
            assert data_min >= 0, f"Baseline should have all positive t-stats, got min={data_min}"

            # Y-min should be at or slightly below 0
            assert y_min <= 0, f"Y-min should be <= 0, got {y_min}"
            assert y_min >= -1, f"Y-min should not be too far below 0, got {y_min}"

            plt.close(fig)


class TestThresholdLineVisibility:
    """Test that threshold line is always visible."""

    @pytest.mark.parametrize("variant,data_file", [
        (None, 'data/model_results.pkl'),
        ('function', 'data/model_results_function.pkl'),
    ])
    def test_threshold_line_visible(self, variant, data_file):
        """
        Test that p<0.001 threshold line (t=3.291) is always visible.

        Uses REAL data for both baseline and function variant.
        """
        if not Path(data_file).exists():
            pytest.skip(f"Data file {data_file} not found")

        threshold = 3.291

        # Generate figure
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_fig.pdf"
            fig = generate_t_test_figure(
                data_path=data_file,
                output_path=str(output_path),
                variant=variant
            )

            # Extract axes and limits
            ax = fig.axes[0]
            y_min, y_max = ax.get_ylim()

            # Check threshold is within visible range
            assert y_min < threshold < y_max, \
                f"Threshold {threshold} should be within y-limits ({y_min}, {y_max})"

            # Check that a horizontal line at threshold exists
            found_threshold_line = False
            for line in ax.get_lines():
                # Check if this is a horizontal line at the threshold
                y_data = line.get_ydata()
                if len(y_data) >= 2 and np.allclose(y_data, threshold, atol=0.01):
                    found_threshold_line = True
                    break

            assert found_threshold_line, "Threshold line should be rendered"

            plt.close(fig)


class TestAverageFigure:
    """Test Figure 2B (average t-test) handles negatives correctly."""

    def test_avg_figure_handles_negatives(self):
        """
        Test that Figure 2B correctly displays negative t-statistics.

        Uses REAL function variant data where averaged t-stats may be negative.
        """
        data_path = 'data/model_results_function.pkl'

        if not Path(data_path).exists():
            pytest.skip(f"Data file {data_path} not found")

        # Generate average figure
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_avg_fig.pdf"
            fig = generate_t_test_avg_figure(
                data_path=data_path,
                output_path=str(output_path),
                variant='function'
            )

            # Extract axes
            ax = fig.axes[0]
            y_min, y_max = ax.get_ylim()

            # Load data and calculate t-statistics
            df = pd.read_pickle(data_path)
            df = df[df['variant'] == 'function'].copy()
            t_raws_df, _ = calculate_t_statistics(df)

            # Check y-limits encompass all averaged values
            data_min = t_raws_df['t_raw'].min()
            data_max = t_raws_df['t_raw'].max()

            assert y_min <= data_min, f"Y-min ({y_min}) should be <= data min ({data_min})"
            assert y_max >= data_max, f"Y-max ({y_max}) should be >= data max ({data_max})"

            # If there are negative values, y_min should be negative
            if data_min < 0:
                assert y_min < 0, f"Y-min should be negative when data has negatives, got {y_min}"

            plt.close(fig)


class TestEdgeCases:
    """Test edge cases with synthetic data."""

    def test_all_negative_t_statistics(self):
        """
        Test with synthetic data where all t-statistics are negative.

        Creates a modified dataset with all negative values to test
        edge case handling.
        """
        data_path = 'data/model_results_function.pkl'

        if not Path(data_path).exists():
            pytest.skip(f"Data file {data_path} not found")

        # Load real data and modify to make all t-stats negative
        df = pd.read_pickle(data_path)
        df = df[df['variant'] == 'function'].copy()

        # Swap train_author and loss_dataset to invert t-statistics
        # This makes models perform worse on their own author
        df['train_author'], df['loss_dataset'] = df['loss_dataset'], df['train_author']

        # Save to temporary file
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_data = Path(tmpdir) / "temp_data.pkl"
            df.to_pickle(temp_data)

            output_path = Path(tmpdir) / "test_negative.pdf"
            fig = generate_t_test_figure(
                data_path=str(temp_data),
                output_path=str(output_path),
                variant='function'
            )

            # Extract axes
            ax = fig.axes[0]
            y_min, y_max = ax.get_ylim()

            # Y-max should be slightly above 0 (for threshold visibility)
            assert y_max > 0, f"Y-max should be > 0 for threshold, got {y_max}"

            # Y-min should be well below 0
            assert y_min < 0, f"Y-min should be < 0 for negative data, got {y_min}"

            plt.close(fig)

    def test_all_positive_t_statistics(self):
        """
        Test with baseline data where all t-statistics are positive.

        Uses REAL baseline data.
        """
        data_path = 'data/model_results.pkl'

        if not Path(data_path).exists():
            pytest.skip(f"Data file {data_path} not found")

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_positive.pdf"
            fig = generate_t_test_figure(
                data_path=data_path,
                output_path=str(output_path),
                variant=None  # Baseline
            )

            # Extract axes
            ax = fig.axes[0]
            y_min, y_max = ax.get_ylim()

            # Y-min should be at or slightly negative (for padding)
            assert y_min <= 0, f"Y-min should be <= 0, got {y_min}"

            # Y-max should be well above 0
            assert y_max > 10, f"Y-max should be well above 0, got {y_max}"

            plt.close(fig)


class TestCLIIntegration:
    """Integration test with CLI."""

    def test_cli_function_variant_complete(self):
        """
        End-to-end test using the actual CLI.

        Runs: ./run_llm_stylometry.sh -f 2a --function-only --no-setup
        Verifies PDF is generated and contains expected content.
        """
        # Check if script exists
        script_path = Path('./run_llm_stylometry.sh')
        if not script_path.exists():
            pytest.skip("CLI script not found")

        # Check if function variant data exists
        if not Path('data/model_results_function.pkl').exists():
            pytest.skip("Function variant data not found")

        # Run CLI command
        result = subprocess.run(
            ['./run_llm_stylometry.sh', '-f', '2a', '--function-only', '--no-setup'],
            capture_output=True,
            text=True,
            timeout=120
        )

        # Check command succeeded
        assert result.returncode == 0, f"CLI failed with: {result.stderr}"

        # Check PDF was generated
        pdf_path = Path('paper/figs/source/t_test_function.pdf')
        assert pdf_path.exists(), "PDF should be generated"
        assert pdf_path.stat().st_size > 1000, "PDF should not be empty"

        # For deeper validation, we can re-generate and check
        fig = generate_t_test_figure(
            data_path='data/model_results_function.pkl',
            variant='function'
        )

        # Verify Austen line exists and has data
        ax = fig.axes[0]
        lines = ax.get_lines()
        assert len(lines) >= 8, "Should have at least 8 author lines"

        # Check for negative y-values (indicating Austen is visible)
        has_negative = False
        for line in lines:
            y_data = line.get_ydata()
            if np.any(y_data < 0):
                has_negative = True
                break
        assert has_negative, "Should have negative y-values (Austen's negative t-stats)"

        plt.close(fig)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
