#!/usr/bin/env python
"""
Generate all figures for the LLM Stylometry paper.

This script:
1. Sets up the conda environment if needed
2. Installs required dependencies
3. Runs all notebooks to generate figures
"""

import subprocess
import sys
import os
from pathlib import Path


def check_conda():
    """Check if conda is installed."""
    try:
        result = subprocess.run(['conda', '--version'], capture_output=True, text=True)
        return result.returncode == 0
    except FileNotFoundError:
        return False


def setup_environment():
    """Set up the conda environment with required packages."""
    print("=" * 60)
    print("Setting up environment for LLM Stylometry")
    print("=" * 60)

    # Check if conda is available
    if not check_conda():
        print("ERROR: Conda not found. Please install Anaconda or Miniconda first.")
        print("Visit: https://docs.conda.io/projects/conda/en/latest/user-guide/install/")
        sys.exit(1)

    # Check if we're already in the llm-stylometry environment
    current_env = os.environ.get('CONDA_DEFAULT_ENV', '')
    if current_env != 'llm-stylometry':
        print("\nTo use this script, please first activate the environment:")
        print("  conda activate llm-stylometry")
        print("\nIf the environment doesn't exist, create it with:")
        print("  conda create -n llm-stylometry python=3.10")
        print("  conda activate llm-stylometry")
        sys.exit(1)

    print("\n✓ Using llm-stylometry conda environment")

    # Install required packages
    print("\nInstalling required packages...")

    # Core packages
    packages_to_install = [
        ('conda', ['-c', 'pytorch', '-c', 'nvidia', 'pytorch>=2.2.0']),
        ('pip', ['numpy<2', 'scipy', 'transformers', 'matplotlib', 'seaborn',
                 'pandas', 'tqdm', 'cleantext', 'plotly', 'scikit-learn']),
        ('pip', ['jupyter', 'ipykernel', 'nbconvert']),
    ]

    for installer, pkgs in packages_to_install:
        if installer == 'conda':
            cmd = ['conda', 'install', '-y'] + pkgs
        else:
            cmd = ['pip', 'install'] + pkgs

        print(f"  Installing: {' '.join(pkgs[:3])}...")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"  Warning: Some packages may already be installed")

    # Install the llm_stylometry package in development mode
    print("\nInstalling llm_stylometry package...")
    result = subprocess.run(['pip', 'install', '-e', '.'], capture_output=True, text=True)
    if result.returncode == 0:
        print("  ✓ Package installed successfully")
    else:
        print("  ! Package may already be installed")

    print("\n✓ Environment setup complete!")


def check_data():
    """Check if the required data file exists."""
    data_file = Path('data/model_results.pkl')
    if not data_file.exists():
        print("\nERROR: Required data file not found: data/model_results.pkl")
        print("Please ensure you have the consolidated model results.")
        print("You may need to run: python consolidate_model_results.py")
        return False
    print("✓ Found model results data")
    return True


def run_notebooks():
    """Run all figure generation notebooks."""
    print("\n" + "=" * 60)
    print("Generating Figures")
    print("=" * 60)

    notebooks_dir = Path('notebooks')
    if not notebooks_dir.exists():
        print("ERROR: notebooks/ directory not found")
        return False

    notebooks = [
        ('figure_1_losses_and_distributions.ipynb', 'Figure 1: Loss curves and distributions'),
        ('figure_2_statistical_tests.ipynb', 'Figure 2: Statistical t-tests'),
        ('figure_3_confusion_matrix.ipynb', 'Figure 3: Confusion matrix heatmap'),
        ('figure_4_mds_visualization.ipynb', 'Figure 4: 3D MDS projection'),
        ('figure_5_oz_attribution.ipynb', 'Figure 5: Oz authorship analysis'),
    ]

    os.chdir(notebooks_dir)

    all_success = True
    for notebook_file, description in notebooks:
        print(f"\n{description}...")
        print(f"  Running {notebook_file}")

        # Use nbconvert to execute the notebook
        cmd = [
            'jupyter', 'nbconvert',
            '--to', 'notebook',
            '--execute',
            '--inplace',
            '--ExecutePreprocessor.timeout=300',
            '--ExecutePreprocessor.kernel_name=python3',
            notebook_file
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode == 0:
            print(f"  ✓ Successfully generated")
        else:
            print(f"  ✗ Error generating figure")
            print(f"    Error: {result.stderr[:200]}")
            all_success = False

    os.chdir('..')
    return all_success


def verify_outputs():
    """Verify that all expected figure files were created."""
    print("\n" + "=" * 60)
    print("Verifying Output Files")
    print("=" * 60)

    expected_files = [
        ('paper/figs/source/all_losses.pdf', 'Figure 1A: Training curves'),
        ('paper/figs/source/stripplot.pdf', 'Figure 1B: Strip plot'),
        ('paper/figs/source/t_test.pdf', 'Figure 2A: Individual t-tests'),
        ('paper/figs/source/t_test_avg.pdf', 'Figure 2B: Average t-test'),
        ('paper/figs/source/average_loss_heatmap.pdf', 'Figure 3: Loss heatmap'),
        ('paper/figs/source/3d_MDS_plot.pdf', 'Figure 4: MDS plot'),
        ('paper/figs/source/oz_losses.pdf', 'Figure 5: Oz losses'),
    ]

    all_found = True
    for file_path, description in expected_files:
        path = Path(file_path)
        if path.exists():
            size_kb = path.stat().st_size / 1024
            print(f"  ✓ {description}: {size_kb:.1f} KB")
        else:
            print(f"  ✗ {description}: NOT FOUND")
            all_found = False

    return all_found


def main():
    """Main execution function."""
    print("\n" + "╔" + "═" * 58 + "╗")
    print("║" + " " * 15 + "LLM Stylometry Figure Generator" + " " * 12 + "║")
    print("╚" + "═" * 58 + "╝")

    # Check environment and setup if needed
    try:
        setup_environment()
    except Exception as e:
        print(f"\nError setting up environment: {e}")
        print("Please set up the environment manually:")
        print("  conda create -n llm-stylometry python=3.10")
        print("  conda activate llm-stylometry")
        print("  pip install -r requirements.txt")
        sys.exit(1)

    # Check for required data
    if not check_data():
        sys.exit(1)

    # Run notebooks to generate figures
    success = run_notebooks()

    # Verify outputs
    if success:
        all_found = verify_outputs()

        if all_found:
            print("\n" + "=" * 60)
            print("✓ All figures generated successfully!")
            print("=" * 60)
            print("\nFigures are saved in: paper/figs/source/")
            print("\nYou can now:")
            print("  1. View the figures in the notebooks/ directory")
            print("  2. Use the PDFs in paper/figs/source/ for the paper")
        else:
            print("\n⚠ Some figures may not have been generated correctly.")
            print("Please check the notebooks for errors.")
    else:
        print("\n✗ Error generating some figures.")
        print("Please check the error messages above.")
        sys.exit(1)


if __name__ == "__main__":
    main()