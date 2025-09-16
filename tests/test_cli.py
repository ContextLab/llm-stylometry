#!/usr/bin/env python
"""Test CLI functionality."""

import pytest
import sys
import subprocess
import tempfile
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestCLI:
    """Test command-line interface."""

    @classmethod
    def setup_class(cls):
        """Set up test paths."""
        cls.script_path = Path(__file__).parent.parent / "code" / "generate_figures.py"
        cls.test_data_path = Path(__file__).parent / "data" / "test_model_results.pkl"
        cls.temp_dir = tempfile.mkdtemp()

        # Verify script exists
        if not cls.script_path.exists():
            pytest.skip("generate_figures.py not found")

        # Verify test data exists
        if not cls.test_data_path.exists():
            pytest.skip("Test data not found. Run create_test_data.py first.")

    def run_cli(self, args):
        """Helper to run CLI with arguments."""
        cmd = [sys.executable, str(self.script_path)] + args
        result = subprocess.run(cmd, capture_output=True, text=True)
        return result

    def test_cli_help(self):
        """Test help command."""
        result = self.run_cli(["--help"])
        assert result.returncode == 0, f"Help failed: {result.stderr}"
        assert "LLM Stylometry CLI" in result.stdout
        assert "--figure" in result.stdout
        assert "--train" in result.stdout

    def test_cli_list(self):
        """Test listing available figures."""
        result = self.run_cli(["--list"])
        assert result.returncode == 0, f"List failed: {result.stderr}"
        assert "Available figures:" in result.stdout
        assert "1a" in result.stdout
        assert "Figure 1A" in result.stdout
        assert "3d_MDS_plot.pdf" in result.stdout

    def test_cli_single_figure(self):
        """Test generating a single figure."""
        output_dir = Path(self.temp_dir) / "single_figure"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Generate figure 1a
        result = self.run_cli([
            "--figure", "1a",
            "--data", str(self.test_data_path),
            "--output", str(output_dir)
        ])

        assert result.returncode == 0, f"Figure generation failed: {result.stderr}"
        assert "Figure 1A" in result.stdout
        assert "Generated" in result.stdout

        # Check output file
        output_file = output_dir / "all_losses.pdf"
        assert output_file.exists(), f"Output file not created: {output_file}"
        assert output_file.stat().st_size > 1000, "Output file too small"

    def test_cli_all_figures(self):
        """Test generating all figures."""
        output_dir = Path(self.temp_dir) / "all_figures"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Generate all figures
        result = self.run_cli([
            "--data", str(self.test_data_path),
            "--output", str(output_dir)
        ])

        assert result.returncode == 0, f"All figures generation failed: {result.stderr}"
        assert "Generating Figures" in result.stdout

        # Check that multiple files were created
        pdf_files = list(output_dir.glob("*.pdf"))
        assert len(pdf_files) >= 5, f"Expected at least 5 PDFs, got {len(pdf_files)}"

    def test_cli_invalid_figure(self):
        """Test error handling for invalid figure."""
        result = self.run_cli([
            "--figure", "99z",
            "--data", str(self.test_data_path)
        ])

        assert result.returncode != 0, "Should fail for invalid figure"
        assert "Unknown figure" in result.stdout or "Unknown figure" in result.stderr

    def test_cli_missing_data(self):
        """Test error handling for missing data file."""
        result = self.run_cli([
            "--data", "nonexistent_file.pkl"
        ])

        assert result.returncode != 0, "Should fail for missing data"
        assert "not found" in result.stdout or "not found" in result.stderr

    @classmethod
    def teardown_class(cls):
        """Clean up temporary directory."""
        import shutil
        if hasattr(cls, 'temp_dir') and Path(cls.temp_dir).exists():
            shutil.rmtree(cls.temp_dir)


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])