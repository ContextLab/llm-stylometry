# Code Directory

Python scripts for model training, analysis, and figure generation.

## Main Scripts

### Training and Figures
- **generate_figures.py** - Main CLI for training and figure generation
- **main.py** - Model training orchestration with parallel GPU support

### Data Processing
- **clean.py** - Preprocess Project Gutenberg texts (remove headers/footers)
- **create_analysis_variants.py** - Generate variant-transformed texts (content, function, POS)
- **consolidate_model_results.py** - Combine model loss logs into single pkl file

### Analysis
- **compute_stats.py** - Statistical analysis: t-tests, threshold crossings, cross-variant comparisons
- **check_training_status.py** - Remote training progress monitoring

### Utilities
- **constants.py** - Shared constants (authors, hyperparameters)

## Usage

Most scripts are called through shell wrappers (see main README):
- `./run_llm_stylometry.sh` → generate_figures.py
- `./run_stats.sh` → compute_stats.py
- `./check_remote_status.sh` → check_training_status.py

Run scripts directly for advanced usage:
```bash
python code/generate_figures.py --help
python code/compute_stats.py --help
```

See main README for complete documentation and examples.
