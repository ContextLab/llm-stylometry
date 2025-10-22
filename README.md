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

## Analysis Variants

The project supports three linguistic variants to understand what stylistic features models learn:

**Content-Only** (`-co`, `--content-only`): Masks function words with `<FUNC>`, preserving only content words (nouns, verbs, adjectives). Tests vocabulary and word choice.

**Function-Only** (`-fo`, `--function-only`): Masks content words with `<CONTENT>`, preserving only function words (articles, prepositions, conjunctions). Tests grammatical structure.

**Part-of-Speech** (`-pos`, `--part-of-speech`): Replaces words with POS tags (Universal Dependencies tagset). Tests syntactic patterns.

All CLI commands accept variant flags. Without a flag, the baseline condition is used. Each variant trains 80 models (8 authors × 10 seeds). See [Training Models from Scratch](#training-models-from-scratch) for training details.

```bash
# Generate figures for variants
./run_llm_stylometry.sh -f 1a -co           # Figure 1A, content variant
./run_llm_stylometry.sh --function-only     # All figures, function variant

# Compute statistics
./run_stats.sh --all                        # All variants at once
./run_stats.sh -co                          # Single variant
```

**Model directories:**
- Baseline: `{author}_tokenizer=gpt2_seed={0-9}/`
- Variants: `{author}_variant={content|function|pos}_tokenizer=gpt2_seed={0-9}/`

**Figure paths:**
- Baseline: `paper/figs/source/figure_name.pdf`
- Variants: `paper/figs/source/figure_name_{variant}.pdf`

### Fairness-Based Loss Thresholding

Variant models converge much faster than baseline models (all cross 3.0 loss by epochs 15-16) and may converge to different final losses. To ensure fair comparison, **fairness-based loss thresholding** is automatically applied to variant figures (1A, 1B, 3, 4, 5):

1. **Compute threshold**: Maximum of all models' minimum training losses within 500 epochs
2. **Truncate data**: Keep all epochs up to and including the first epoch where training loss ≤ threshold
3. **Fair comparison**: All models compared at the same training loss level (the fairness threshold)

This ensures models are not unfairly compared when some converged to higher losses than others. The feature is enabled by default for variants and can be disabled:

```bash
# Fairness enabled (default for variants)
./run_llm_stylometry.sh -f 1a -fo

# Fairness disabled
./run_llm_stylometry.sh -f 1a -fo --no-fairness
```

**Example results** (function-only variant):
- Fairness threshold: 1.2720 (Austen's minimum loss)
- Models truncated between epochs 88-500
- Data reduced: 360,640 rows → 170,659 rows (47.3%)

**Python API:**

```python
from llm_stylometry.analysis.fairness import (
    compute_fairness_threshold,
    apply_fairness_threshold
)

# Compute threshold for variant data
df = pd.read_pickle('data/model_results_function.pkl')
threshold = compute_fairness_threshold(df, min_epochs=500)
print(f"Fairness threshold: {threshold:.4f}")

# Truncate data at threshold
df_fair = apply_fairness_threshold(df, threshold, use_first_crossing=True)

# Generate figure with fairness
from llm_stylometry.visualization import generate_all_losses_figure
fig = generate_all_losses_figure(
    data_path='data/model_results_function.pkl',
    variant='function',
    apply_fairness=True  # default for variants
)
```

**Note**: T-test figures (2A, 2B) never apply fairness thresholding since they require all 500 epochs for statistical calculations.

## Training Models from Scratch

### Local Training

```bash
# Train baseline models
./run_llm_stylometry.sh --train

# Train analysis variants
./run_llm_stylometry.sh --train --content-only     # Content variant
./run_llm_stylometry.sh --train --function-only    # Function variant
./run_llm_stylometry.sh --train --part-of-speech   # POS variant

# Short flags
./run_llm_stylometry.sh -t -co              # Content variant
./run_llm_stylometry.sh -t -fo              # Function variant
./run_llm_stylometry.sh -t -pos             # POS variant

# Resume training from existing checkpoints
./run_llm_stylometry.sh --train --resume
./run_llm_stylometry.sh -t -r -co           # Resume content variant

# Limit GPU usage if needed
./run_llm_stylometry.sh --train --max-gpus 4
```

Each training run will:
1. Clean and prepare the data if needed
2. Train 80 models (8 authors × 10 seeds)
3. Consolidate results into `data/model_results.pkl`

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
# Train baseline models
./remote_train.sh

# Train analysis variants
./remote_train.sh --content-only        # Content variant
./remote_train.sh -fo                   # Function variant (short flag)
./remote_train.sh --part-of-speech      # POS variant

# Resume training from existing checkpoints
./remote_train.sh --resume              # Resume baseline
./remote_train.sh -r -co                # Resume content variant

# Kill existing training sessions
./remote_train.sh --kill                # Kill and exit
./remote_train.sh --kill --resume       # Kill and restart

# You'll be prompted for:
# - Server address (hostname or IP)
# - Username
```

**What this script does:** The `remote_train.sh` script connects to your GPU server via SSH and executes `run_llm_stylometry.sh --train -y` (with any variant flags you specify) in a `screen` session. This allows you to disconnect your local machine while the GPU server continues training.

The script will:
1. SSH into your GPU server
2. Update the repository in `~/llm-stylometry` (or clone if it doesn't exist)
3. Start training in a `screen` session with the specified options
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

#### Checking training status

Monitor training progress on your GPU server using `check_remote_status.sh` **from your local machine**:

```bash
# Check status on default cluster (tensor02)
./check_remote_status.sh

# Check status on specific cluster
./check_remote_status.sh --cluster tensor01
./check_remote_status.sh --cluster tensor02
```

The script provides a comprehensive status report including:

**For completed models:**
- Number of completed seeds per author (out of 10)
- Final training loss (mean ± std across all completed seeds)

**For in-progress models:**
- Current epoch and progress percentage
- Current training loss
- Estimated time to completion (based on actual runtime per epoch)

**Example output:**
```
================================================================================
POS VARIANT MODELS
================================================================================

AUSTEN
--------------------------------------------------------------------------------
  Completed: 2/10 seeds
  Final training loss: 1.1103 ± 0.0003 (mean ± std)
  In-progress: 1 seeds
    Seed 2: epoch 132/500 (26.4%) | loss: 1.2382 | ETA: 1d 1h 30m

--------------------------------------------------------------------------------
Summary: 16/80 complete, 8 in progress
Estimated completion: 1d 1h 30m (longest), 1d 0h 45m (average)
```

**How it works:**
1. Connects to your GPU server using saved credentials (`.ssh/credentials_{cluster}.json`)
2. Analyzes all model directories and loss logs
3. Calculates statistics for completed models
4. Estimates remaining time based on actual training progress
5. Reports status for baseline and all variant models

**Prerequisites:** The script uses the same credentials file as `remote_train.sh`. If credentials aren't saved, you'll be prompted to enter them interactively.

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
