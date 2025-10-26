# LLM Stylometry

[![Tests](https://github.com/ContextLab/llm-stylometry/actions/workflows/tests.yml/badge.svg)](https://github.com/ContextLab/llm-stylometry/actions/workflows/tests.yml)


## Overview

This repository contains the code and data for our [paper](https://insert.link.when.ready) on using large language models (LLMs) for stylometric analysis. We demonstrate that GPT-2 models trained on individual authors' works can capture unique writing styles, enabling accurate authorship attribution through cross-entropy loss comparison.

## Repository Structure

```
llm-stylometry/
├── llm_stylometry/       # Python package (analysis, visualization, data loading)
├── code/                 # Scripts (training, figures, stats) - see code/README.md
├── data/                 # Texts and results - see data/README.md
├── models/               # 320 trained GPT-2 models - see models/README.md
├── paper/                # LaTeX source and figures - see paper/README.md
├── tests/                # Test suite
├── run_llm_stylometry.sh # Main CLI wrapper
├── remote_train.sh       # GPU cluster training
├── check_remote_status.sh # Monitor remote training
└── sync_models.sh        # Download trained models
```

See folder-specific README files for detailed documentation.

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

The easiest way to use the toolbox is via the CLI wrapper scripts:

```bash
# Generate all figures from pre-computed results
./run_llm_stylometry.sh

# Generate specific figure
./run_llm_stylometry.sh -f 1a    # Figure 1A only
./run_llm_stylometry.sh -l       # List available figures

# Compute statistical analyses
./run_stats.sh

# Get help
./run_llm_stylometry.sh -h
```

For training models from scratch, see [Training Models from Scratch](#training-models-from-scratch).

**Python API:** You can also use Python directly for programmatic access:

```python
from llm_stylometry.visualization import generate_all_losses_figure

# Generate a figure
fig = generate_all_losses_figure(
    data_path='data/model_results.pkl',
    output_path='figure.pdf'
)
```

See the [Package API](#package-api) section for all available functions.

**Note**: T-test calculations (Figure 2) take 2-3 minutes due to statistical computations across all epochs and authors.

**Downloading pre-trained weights (optional):** Model weight files are gitignored due to size. Download pre-trained weights to explore or use trained models:

```bash
./download_model_weights.sh --all    # Download all variants (~26.6GB)
./download_model_weights.sh -b       # Baseline only (~6.7GB)
```

See `models/README.md` for details. Pre-trained weights are not required for generating figures.

**Author datasets on HuggingFace:** Cleaned text corpora for all 8 authors are publicly available on HuggingFace at https://huggingface.co/contextlab (browse datasets). Each corpus includes verified book titles and can be loaded with `from datasets import load_dataset`.

## Analysis Variants

The paper analyzes three linguistic variants (Supplemental Figures S1-S8):

- **Content-only**: Function words masked → tests vocabulary/word choice (Supp. Figs. S1, S4, S7A, S8A)
- **Function-only**: Content words masked → tests grammatical structure (Supp. Figs. S2, S5, S7B, S8B)
- **Part-of-speech**: Words → POS tags → tests syntactic patterns (Supp. Figs. S3, S6, S7C, S8C)

**Generate supplemental figures:**
```bash
./run_llm_stylometry.sh -f s1a    # Supp. Fig. S1A (content-only, Fig 1A format)
./run_llm_stylometry.sh -f s4b    # Supp. Fig. S4B (content-only, Fig 2B format)
./run_llm_stylometry.sh -f s7c    # Supp. Fig. S7C (POS confusion matrix)
```

**Training variants:** Each trains 80 models (8 authors × 10 seeds)
```bash
./run_llm_stylometry.sh --train -co    # Content-only
./remote_train.sh -fo                  # Function-only on GPU cluster
```

**Statistical analysis:**
```bash
./run_stats.sh            # All variants (default)
```

**Fairness-based loss thresholding:** Automatically ensures fair comparison when variant models converge to different final losses. Disable with `--no-fairness` if needed.

## Training Models from Scratch

Training 320 models (baseline + 3 variants) requires a CUDA GPU. See `models/README.md` for details.

**Local training:**
```bash
./run_llm_stylometry.sh --train           # Baseline (80 models)
./run_llm_stylometry.sh --train -co       # Content-only variant
./run_llm_stylometry.sh -t -r             # Resume from checkpoints
```

**Remote training:**

Requires GPU cluster with SSH access. Create `.ssh/credentials_mycluster.json`:
```json
{"server": "hostname", "username": "user", "password": "pass"}
```

Then from local machine:
```bash
./remote_train.sh --cluster mycluster           # Train baseline
./remote_train.sh -co --cluster mycluster -r    # Resume content variant
./check_remote_status.sh --cluster mycluster    # Monitor progress
./sync_models.sh --cluster mycluster -a         # Download when complete
```

Trains in detached screen session on GPU server. See script help for full options.

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

The repository includes comprehensive tests for all functionality:

```bash
# Install test dependencies
pip install pytest pytest-timeout

# Run all tests
pytest tests/

# Run specific test modules
pytest tests/test_visualization.py  # Figure generation
pytest tests/test_cli.py            # CLI functionality
pytest tests/test_model_training.py # Model operations
```

Tests run automatically on GitHub Actions (Linux, macOS, Windows, Python 3.10). See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed testing guidelines and philosophy.

## Package API

The `llm_stylometry` package provides functions for all analyses:

```python
# Visualization functions
from llm_stylometry.visualization import (
    generate_all_losses_figure,      # Figure 1A: Training curves
    generate_stripplot_figure,       # Figure 1B: Loss distributions
    generate_t_test_figure,          # Figure 2A: Individual t-tests
    generate_t_test_avg_figure,      # Figure 2B: Average t-test
    generate_loss_heatmap_figure,    # Figure 3: Confusion matrix
    generate_3d_mds_figure,          # Figure 4: MDS visualization
    generate_oz_losses_figure        # Figure 5: Oz analysis
)

# Fairness-based loss thresholding (for variant comparisons)
from llm_stylometry.analysis.fairness import (
    compute_fairness_threshold,      # Compute fairness threshold
    apply_fairness_threshold         # Truncate data at threshold
)
```

All visualization functions support `variant` and `apply_fairness` parameters (except t-test figures). See the [Fairness-Based Loss Thresholding](#fairness-based-loss-thresholding) section for details.

## Citation

If you use this code or data in your research, please cite:

```bibtex
@article{StroEtal25,
  title={A Stylometric Application of Large Language Models},
  author={Stropkay, Harrison F. and Chen, Jiayi and Jabelli, Mohammad J. L. and Rockmore, Daniel N. and Manning, Jeremy R.},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2025}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

For questions or issues, please open a GitHub issue or contact Jeremy R. Manning (jeremy.r.manning@dartmouth.edu).
