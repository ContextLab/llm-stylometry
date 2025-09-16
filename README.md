# LLM Stylometry

A stylometric application of large language models for authorship attribution.

## Overview

This repository contains the code and data for our paper on using large language models (LLMs) for stylometric analysis. We demonstrate that GPT-2 models trained on individual authors' works can capture unique writing styles, enabling accurate authorship attribution through cross-entropy loss comparison.

## Repository Structure

```
llm-stylometry/
├── llm_stylometry/        # Python package (in development)
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
├── notebooks/           # Jupyter notebooks for figure generation
└── paper/               # LaTeX paper and figures
    ├── main.tex        # Paper source
    └── figs/           # Paper figures
```

## Installation

### Using Conda (Recommended)

```bash
# Create and activate environment
conda create -n llm-stylometry python=3.10
conda activate llm-stylometry

# Install PyTorch (adjust for your CUDA version)
conda install -c pytorch -c nvidia pytorch=2.2.2 pytorch-cuda=12.1

# Install other dependencies
conda install "numpy<2" scipy transformers matplotlib seaborn pandas tqdm
pip install cleantext plotly scikit-learn
```

### Using pip (Alternative)

```bash
# Clone the repository
git clone https://github.com/ContextLab/llm-stylometry.git
cd llm-stylometry

# Install package (in development)
pip install -e .
```

## Quick Start

### Using Pre-computed Results

The repository includes pre-computed results from training 80 models (8 authors × 10 random seeds). These results are consolidated in `data/model_results.pkl`.

```python
import pandas as pd

# Load consolidated results
df = pd.read_pickle('data/model_results.pkl')

# View summary
print(f"Models: {df['model_name'].nunique()}")
print(f"Authors: {sorted(df['author'].unique())}")
```

### Generating Figures

Figures can be generated using either the original scripts or the new Jupyter notebooks:

```bash
# Using notebooks (recommended)
jupyter notebook notebooks/figure_1_losses_and_distributions.ipynb

# Using original scripts
python code/all_losses.py
python code/stripplot.py
python code/t_test_figs.py
# ... etc
```

## Training Models from Scratch

**Note**: Training requires a CUDA-enabled GPU and takes significant time (~80 models total).

```bash
# Prepare data (if not already cleaned)
python code/clean.py

# Train all models (uses multiple GPUs if available)
python code/main.py
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

## Citation

If you use this code or data in your research, please cite:

```bibtex
@article{stropkay2024stylometry,
  title={A Stylometric Application of Large Language Models},
  author={Stropkay, Harrison F. and Chen, Jiayi and Rockmore, Daniel N. and Manning, Jeremy R.},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2024}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

For questions or issues, please open a GitHub issue or contact:
- Jeremy R. Manning (jeremy.r.manning@dartmouth.edu)