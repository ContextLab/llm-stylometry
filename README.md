# LLM Stylometry

[![Tests](https://github.com/ContextLab/llm-stylometry/actions/workflows/tests.yml/badge.svg)](https://github.com/ContextLab/llm-stylometry/actions/workflows/tests.yml)


## Overview

This repository contains the code and data for our [paper](https://insert.link.when.ready) on using large language models (LLMs) for stylometric analysis. We demonstrate that GPT-2 models trained on individual authors' works can capture unique writing styles, enabling accurate authorship attribution through cross-entropy loss comparison.

## Repository Structure

```
llm-stylometry/
├── llm_stylometry/        # Python package with analysis tools
│   ├── core/             # Core experiment and configuration
│   ├── data/             # Data loading and tokenization
│   ├── models/           # Model utilities
│   ├── analysis/         # Statistical analysis
│   └── visualization/    # Plotting and visualization
├── code/                 # Original analysis scripts
├── data/                 # Datasets and results
│   ├── raw/             # Original texts from Project Gutenberg
│   ├── cleaned/         # Preprocessed texts by author
│   └── model_results.pkl # Consolidated model training results
├── models/              # Model configurations and logs
└── paper/               # LaTeX paper and figures
    ├── main.tex        # Paper source
    └── figs/           # Paper figures
```

## Installation

### Automatic Setup and Execution

The easiest way to get started is using the comprehensive CLI script:

```bash
# Clone the repository
git clone https://github.com/ContextLab/llm-stylometry.git
cd llm-stylometry

# Run the CLI (automatically sets up conda environment if needed)
./run_llm_stylometry.sh
```

The script will:
1. Check for conda and install Miniconda if needed (platform-specific)
2. Create and configure the conda environment
3. Install all dependencies including PyTorch with CUDA support
4. Generate all paper figures from pre-computed results

### Manual Setup

If you prefer manual setup:

```bash
# Create environment
conda create -n llm-stylometry python=3.10
conda activate llm-stylometry

# Install PyTorch (adjust for your CUDA version)
conda install -c pytorch -c nvidia pytorch pytorch-cuda=12.1

# Install other dependencies
pip install "numpy<2" scipy transformers matplotlib seaborn pandas tqdm
pip install cleantext plotly scikit-learn

# Install the package
pip install -e .
```

## Quick Start

### Using the CLI

```bash
# Generate all figures (default)
./run_llm_stylometry.sh

# Generate specific figure
./run_llm_stylometry.sh -f 1a    # Figure 1A only
./run_llm_stylometry.sh -f 4     # Figure 4 (MDS plot) only

# List available figures
./run_llm_stylometry.sh -l

# Train models from scratch (requires GPU)
./run_llm_stylometry.sh -t

# Custom data and output paths
./run_llm_stylometry.sh -d path/to/model_results.pkl -o path/to/output

# Get help
./run_llm_stylometry.sh -h
```

### Using Python Directly

```bash
conda activate llm-stylometry

# Generate all figures
python generate_figures.py

# Generate specific figure
python generate_figures.py --figure 1a

# Train models from scratch
python generate_figures.py --train

# List available figures
python generate_figures.py --list
```

**Note**: The t-test calculations (Figure 2) take approximately 2-3 minutes due to statistical computations across all epochs and authors.

### Using Pre-computed Results

The repository includes pre-computed results from training 80 models (8 authors × 10 random seeds). These results are consolidated in `data/model_results.pkl`.

```python
import pandas as pd
from llm_stylometry.visualization import generate_all_losses_figure

# Load consolidated results
df = pd.read_pickle('data/model_results.pkl')

# Generate a figure
fig = generate_all_losses_figure(
    data_path='data/model_results.pkl',
    output_path='my_figure.pdf'
)
```

### Available Figures

- **1a**: Figure 1A - Training curves (all_losses.pdf)
- **1b**: Figure 1B - Strip plot (stripplot.pdf)
- **2a**: Figure 2A - Individual t-tests (t_test.pdf)
- **2b**: Figure 2B - Average t-test (t_test_avg.pdf)
- **3**: Figure 3 - Confusion matrix heatmap (average_loss_heatmap.pdf)
- **4**: Figure 4 - 3D MDS plot (3d_MDS_plot.pdf)
- **5**: Figure 5 - Oz authorship analysis (oz_losses.pdf)

## Training Models from Scratch

**Note**: Training requires a CUDA-enabled GPU and takes significant time (~80 models total).

```bash
# Using the CLI (recommended)
./run_llm_stylometry.sh --train

# Or manually
conda activate llm-stylometry
python code/clean.py              # Clean data
python code/main.py               # Train models
python consolidate_model_results.py  # Consolidate results
```

### Model Configuration

Each model uses:
- GPT-2 architecture with custom dimensions
- 128 embedding dimensions
- 8 transformer layers
- 8 attention heads
- 1024 maximum sequence length
- Training on ~643,041 tokens per author
- Early stopping at loss ≤ 3.0 (after minimum 500 epochs)

## Data

### Authors Analyzed

We analyze texts from 8 authors:
- L. Frank Baum
- Ruth Plumly Thompson
- Jane Austen
- Charles Dickens
- F. Scott Fitzgerald
- Herman Melville
- Mark Twain
- H.G. Wells

### Special Evaluation Sets

For Baum and Thompson models, we include additional evaluation sets:
- **non_oz_baum**: Non-Oz works by Baum
- **non_oz_thompson**: Non-Oz works by Thompson
- **contested**: The 15th Oz book with disputed authorship

## Key Results

Our analysis shows that:
1. Models achieve lower cross-entropy loss on texts from the author they were trained on
2. The approach correctly attributes the contested 15th Oz book to Thompson
3. Stylometric distances between authors can be visualized using MDS

## Testing

### Running Tests Locally

The repository includes comprehensive tests that use real models and data (no mocks):

```bash
# Install test dependencies
pip install pytest pytest-timeout

# Run all tests
pytest tests/

# Run specific test modules
pytest tests/test_visualization.py  # Test figure generation
pytest tests/test_cli.py            # Test CLI functionality
pytest tests/test_model_training.py # Test model operations

# Run with verbose output
pytest -v tests/
```

### Test Coverage

Our test suite includes:

- **Visualization Tests**: Verify all figure generation functions work correctly with synthetic data
- **CLI Tests**: Test all command-line options and error handling
- **Model Training Tests**: Test model creation, training, and saving with tiny models
- **Data Tests**: Verify data loading and processing functions

### Continuous Integration

Tests run automatically on GitHub Actions for:
- **Platforms**: Linux, macOS, Windows
- **Python Version**: 3.10
- **Execution Time**: All tests complete in under 5 minutes

The CI pipeline:
1. Sets up Python environment
2. Installs dependencies (including CPU-only PyTorch)
3. Creates synthetic test data
4. Runs all test modules
5. Validates figure generation
6. Uploads artifacts on failure for debugging

### Writing New Tests

When adding new functionality, ensure tests:
- Use real data and models (no mocks)
- Complete quickly (use small datasets/models)
- Test actual functionality end-to-end
- Generate real outputs (PDFs, models, etc.)

Example test structure:
```python
def test_new_feature():
    # Use synthetic test data
    data = pd.read_pickle('tests/data/test_model_results.pkl')

    # Generate real output
    output_path = 'test_output.pdf'
    result = generate_figure(data, output_path)

    # Verify real file was created
    assert Path(output_path).exists()
    assert Path(output_path).stat().st_size > 1000
```

## Package API

The `llm_stylometry` package provides functions for all analyses:

```python
from llm_stylometry.visualization import (
    generate_all_losses_figure,      # Figure 1A: Training curves
    generate_stripplot_figure,       # Figure 1B: Loss distributions
    generate_t_test_figure,          # Figure 2A: Individual t-tests
    generate_t_test_avg_figure,      # Figure 2B: Average t-test
    generate_loss_heatmap_figure,    # Figure 3: Confusion matrix
    generate_3d_mds_figure,          # Figure 4: MDS visualization
    generate_oz_losses_figure        # Figure 5: Oz analysis
)
```

## Citation

If you use this code or data in your research, please cite:

```bibtex
@article{StroEtal25,
  title={A Stylometric Application of Large Language Models},
  author={Stropkay, Harrison F. and Chen, Jiayi and Rockmore, Daniel N. and Manning, Jeremy R.},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2025}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

For questions or issues, please open a GitHub issue or contact Jeremy R. Manning (jeremy.r.manning@dartmouth.edu).
