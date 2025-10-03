# LLM Stylometry

[![Tests](https://github.com/ContextLab/llm-stylometry/actions/workflows/tests.yml/badge.svg)](https://github.com/ContextLab/llm-stylometry/actions/workflows/tests.yml)


## Overview

This repository contains the code and data for our [paper](https://insert.link.when.ready) on using large language models (LLMs) for stylometric analysis. We demonstrate that GPT-2 models trained on individual authors' works can capture unique writing styles, enabling accurate authorship attribution through cross-entropy loss comparison.

## Repository Structure

```
llm-stylometry/
├── .github/              # GitHub Actions CI/CD
│   └── workflows/       # Test automation workflows
├── llm_stylometry/       # Python package with analysis tools
│   ├── analysis/        # Statistical analysis utilities
│   ├── core/           # Core experiment and configuration
│   ├── data/           # Data loading and tokenization
│   ├── models/         # Model utilities
│   ├── utils/          # Helper utilities
│   ├── visualization/  # Plotting and visualization
│   └── cli_utils.py    # CLI helper functions
├── code/                # Training and CLI scripts
│   ├── generate_figures.py       # Main CLI entry point
│   ├── consolidate_model_results.py # Result consolidation
│   ├── main.py         # Model training orchestration
│   ├── clean.py        # Data preprocessing
│   └── ...             # Supporting training modules
├── data/                # Datasets and results
│   ├── raw/            # Original texts from Project Gutenberg
│   ├── cleaned/        # Preprocessed texts by author
│   └── model_results.pkl # Consolidated model training results
├── models/              # Trained models (80 baseline + 240 variants = 320 total)
│   └── {author}_tokenizer=gpt2_seed={0-9}/  # Baseline models
│   └── {author}_variant={variant}_tokenizer=gpt2_seed={0-9}/  # Variant models
├── paper/               # LaTeX paper and figures
│   ├── main.tex        # Paper source
│   ├── main.pdf        # Compiled paper
│   └── figs/           # Paper figures
├── tests/               # Test suite
│   ├── data/           # Test data and fixtures
│   ├── test_*.py       # Test modules
│   └── check_outputs.py # Output validation script
├── run_llm_stylometry.sh # Shell wrapper for easy setup
├── remote_train.sh     # Remote GPU server training script
├── sync_models.sh      # Download models from remote server
├── LICENSE             # MIT License
├── README.md           # This file
├── requirements-dev.txt # Development dependencies
├── pyproject.toml      # Package configuration
└── pytest.ini          # Test configuration
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

# Compute statistical analyses (Table 1 and key statistics)
./run_stats.sh

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

# Resume training from existing checkpoints
python generate_figures.py --train --resume

# List available figures
python generate_figures.py --list
```

**Note**: The t-test calculations (Figure 2) take approximately 2-3 minutes due to statistical computations across all epochs and authors.

## Analysis Variants

The project supports three linguistic analysis variants to understand what stylistic features models learn:

### Content-Only Variant
Masks function words with `<FUNC>` token, preserving only content words (nouns, verbs, adjectives, etc.)
- **Tests:** Whether models distinguish authors based on vocabulary and word choice
- **Example transformation:**
  - Original: "The quick brown fox jumps over the lazy dog"
  - Transformed: "<FUNC> quick brown fox jumps <FUNC> <FUNC> lazy dog"

### Function-Only Variant
Masks content words with `<CONTENT>` token, preserving only function words (articles, prepositions, conjunctions)
- **Tests:** Whether models distinguish authors based on grammatical structure
- **Example transformation:**
  - Original: "The quick brown fox jumps over the lazy dog"
  - Transformed: "The <CONTENT> <CONTENT> <CONTENT> <CONTENT> over the <CONTENT> <CONTENT>"

### Part-of-Speech (POS) Variant
Replaces all words with their POS tags using Universal Dependencies tagset
- **Tests:** Whether models distinguish authors based on syntactic patterns
- **Example transformation:**
  - Original: "The quick brown fox jumps over the lazy dog"
  - Transformed: "DET ADJ ADJ NOUN VERB ADP DET ADJ NOUN"

### Training Variants

```bash
# Train a single variant (8 authors × 10 seeds = 80 models per variant)
./run_llm_stylometry.sh --train --content-only
./run_llm_stylometry.sh --train --function-only
./run_llm_stylometry.sh --train --part-of-speech

# Short flags
./run_llm_stylometry.sh -t -co   # content-only
./run_llm_stylometry.sh -t -fo   # function-only
./run_llm_stylometry.sh -t -pos  # part-of-speech

# Train baseline (no variant flag)
./run_llm_stylometry.sh -t       # baseline (80 models)

# To train all conditions sequentially (baseline + 3 variants = 320 models total):
./run_llm_stylometry.sh -t                    # baseline
./run_llm_stylometry.sh -t --content-only     # content variant
./run_llm_stylometry.sh -t --function-only    # function variant
./run_llm_stylometry.sh -t --part-of-speech   # POS variant
```

### Generating Variant Figures

```bash
# Generate all figures for a single variant
./run_llm_stylometry.sh --content-only
./run_llm_stylometry.sh --function-only
./run_llm_stylometry.sh --part-of-speech

# Generate specific figure for a variant
./run_llm_stylometry.sh -f 1a --content-only
./run_llm_stylometry.sh -f 1a --function-only

# Generate baseline figures (no variant flag)
./run_llm_stylometry.sh           # all baseline figures
./run_llm_stylometry.sh -f 1a     # specific baseline figure

# To generate all figures for all conditions:
./run_llm_stylometry.sh                    # baseline
./run_llm_stylometry.sh --content-only     # content variant
./run_llm_stylometry.sh --function-only    # function variant
./run_llm_stylometry.sh --part-of-speech   # POS variant
```

### Computing Variant Statistics

```bash
# Single variant statistics
./run_stats.sh                    # baseline (default)
./run_stats.sh --content-only     # content variant
./run_stats.sh --function-only    # function variant
./run_stats.sh --part-of-speech   # POS variant

# All statistics (baseline + all 3 variants)
./run_stats.sh --all
```

### Remote Training with Variants

```bash
# Train a single variant on GPU server
./remote_train.sh --content-only
./remote_train.sh --function-only
./remote_train.sh --part-of-speech

# Resume variant training
./remote_train.sh --resume --content-only

# Train baseline on remote server (no variant flag)
./remote_train.sh

# To train all conditions on remote server, run sequentially:
./remote_train.sh                    # baseline
./remote_train.sh --content-only     # content variant
./remote_train.sh --function-only    # function variant
./remote_train.sh --part-of-speech   # POS variant
```

### Model Naming Convention

Models include variant in their directory names:
- Baseline: `{author}_tokenizer=gpt2_seed={0-9}/`
- Content: `{author}_variant=content_tokenizer=gpt2_seed={0-9}/`
- Function: `{author}_variant=function_tokenizer=gpt2_seed={0-9}/`
- POS: `{author}_variant=pos_tokenizer=gpt2_seed={0-9}/`

### Figure Output Paths

Figures include variant suffix:
- Baseline: `paper/figs/source/all_losses.pdf`
- Content: `paper/figs/source/all_losses_content.pdf`
- Function: `paper/figs/source/all_losses_function.pdf`
- POS: `paper/figs/source/all_losses_pos.pdf`

### Using Pre-computed Results

The repository includes pre-computed results from training 80 baseline models (8 authors × 10 random seeds). Results for analysis variants can be generated using the same pipeline. These results are consolidated in `data/model_results.pkl`.

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

### Statistical Analysis

Generate key statistics from the paper:

```bash
# Compute statistical analyses
./run_stats.sh
```

This produces:
- **Twain p-threshold analysis**: Epoch where Twain model first achieves p < 0.001
- **Average t-test**: t-test of average t-statistics across seeds, at 500th epoch
- **Table 1**: Individual author model t-tests comparing self vs. other losses

## Training Models from Scratch

**Note**: Training requires a CUDA-enabled GPU and takes significant time (~80 models total).

### Local Training

```bash
# Using the CLI (recommended - handles all steps automatically)
./run_llm_stylometry.sh --train

# Resume training from existing checkpoints
./run_llm_stylometry.sh --train --resume

# Limit GPU usage if needed
./run_llm_stylometry.sh --train --max-gpus 4
```

This command will:
1. Clean and prepare the data if needed
2. Train 80 models for the selected condition (8 authors × 10 seeds)
3. Consolidate results into `data/model_results.pkl`

**Note:** Training all conditions (baseline + 3 variants) requires training 320 models total.

**Resume Training**: The `--resume` flag allows you to continue training from existing checkpoints:
- Models that have already met training criteria are automatically skipped
- Partially trained models with saved weights resume from their last checkpoint
- Models without weights are trained from scratch (even if loss logs exist)
- Random states are restored from checkpoints to ensure consistent training continuation

The training pipeline automatically handles data preparation, model training across available GPUs, and result consolidation. Individual model checkpoints and loss logs are saved in the `models/` directory.

### Remote Training on GPU Server

#### Prerequisites: Setting up Git credentials on the server

Before using the remote training script, you need to set up Git credentials on your server once:

1. SSH into your server:
```bash
ssh username@server
```

2. Configure Git with your credentials:
```bash
# Set your Git user information (use your GitHub username)
git config --global user.name "your-github-username"
git config --global user.email "your.email@example.com"

# Enable credential storage
git config --global credential.helper store
```

3. Clone the repository with your Personal Access Token:
```bash
# Replace <username> and <token> with your GitHub username and Personal Access Token
# Get a token from: https://github.com/settings/tokens (grant 'repo' scope)
git clone https://<username>:<token>@github.com/ContextLab/llm-stylometry.git

# The credentials will be stored for future use
cd llm-stylometry
git pull  # This should work without prompting for credentials
```

#### Using the remote training script

Once Git credentials are configured on your server, run `remote_train.sh` **from your local machine** (not on the GPU server):

```bash
# From your local machine, start training on the remote GPU server
./remote_train.sh

# Resume training from existing checkpoints
./remote_train.sh --resume  # or -r

# Kill existing training sessions and optionally start new one
./remote_train.sh --kill  # or -k

# Kill and resume (restart interrupted training)
./remote_train.sh --kill --resume

# You'll be prompted for:
# - Server address (hostname or IP)
# - Username
```

**What this script does:** The `remote_train.sh` script connects to your GPU server via SSH and executes `run_llm_stylometry.sh --train -y` (or `--train --resume -y` if resuming) in a `screen` session. This allows you to disconnect your local machine while the GPU server continues training.

The script will:
1. SSH into your GPU server
2. Update the repository in `~/llm-stylometry` (or clone if it doesn't exist)
3. Start `run_llm_stylometry.sh --train -y` in a `screen` session
4. Exit, allowing your local machine to disconnect while training continues on the server

#### Monitoring training progress

To check on the training status, SSH into the server and reattach to the screen session:

```bash
# From your local machine
ssh username@server

# On the server, reattach to see live training output
screen -r llm_training

# To detach and leave training running, press Ctrl+A, then D
# To exit SSH while keeping training running
exit
```

#### Downloading results after training completes

Once training is complete, use `sync_models.sh` **from your local machine** to download the trained models and results:

```bash
# Download baseline models only (default)
./sync_models.sh

# Download specific variants
./sync_models.sh --content-only           # Content variant only
./sync_models.sh --baseline --content-only # Baseline + content
./sync_models.sh --all                    # All conditions (320 models)

# You'll be prompted for:
# - Server address
# - Username
```

**Variant Flags:**
- `-b, --baseline`: Sync baseline models (80 models)
- `-co, --content-only`: Sync content-only variant (80 models)
- `-fo, --function-only`: Sync function-only variant (80 models)
- `-pos, --part-of-speech`: Sync POS variant (80 models)
- `-a, --all`: Sync all conditions (320 models total)
- Flags are stackable: `./sync_models.sh -b -co` syncs baseline + content

**How it works:**
1. Checks which requested models are complete on remote server (80 per condition)
2. Only syncs complete model sets
3. Uses rsync to download models with progress indication
4. Backs up existing local models before replacing
5. Also syncs `model_results.pkl` if available

**Note**: The script verifies models are complete before downloading. If training is in progress, it will show which models are missing and skip incomplete conditions.

### Model Configuration

Each model uses the same architecture and hyperparameters (applies to baseline and all variants):
- GPT-2 architecture with custom dimensions
- 128 embedding dimensions
- 8 transformer layers
- 8 attention heads
- 1024 maximum sequence length
- Training on ~643,041 tokens per author
- Early stopping at loss ≤ 3.0 (after minimum 500 epochs)

**Note:** All analysis variants use identical training configurations, differing only in input text transformations. This ensures fair comparison across conditions.

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
