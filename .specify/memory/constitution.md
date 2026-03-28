# LLM Stylometry Constitution

## Core Principles

### I. Scientific Accuracy
All analyses must produce correct, verifiable results. Every statistical claim must be backed by reproducible computation. No results may be manually adjusted or cherry-picked. When code produces unexpected results, investigate the cause before proceeding — do not paper over anomalies.

### II. Replicability
Every experiment must be fully reproducible from the repository alone. This means: (a) all data processing steps are scripted, not manual; (b) random seeds are fixed and documented; (c) environment requirements (Python version, package versions) are pinned; (d) pre-computed results include the exact commands used to generate them. A new researcher should be able to clone the repo and reproduce every figure and table.

### III. Robust Documentation
Code, analyses, and results must be documented at three levels: (a) inline comments for non-obvious logic; (b) docstrings for all public functions with parameter descriptions; (c) README and paper text that explain the *why* behind methodological choices. Documentation must be updated whenever code changes — stale docs are worse than no docs.

### IV. Data Purity
Training data must be strictly separated from evaluation data. When using language models, training from scratch is preferred over fine-tuning pre-trained models to guarantee that held-out texts were never seen during pre-training. All data provenance must be traceable (e.g., Project Gutenberg IDs).

### V. Statistical Rigor
All quantitative claims must include appropriate uncertainty estimates (confidence intervals, p-values, or bootstrap ranges). Multiple random seeds (minimum 10) must be used to estimate variability. Effect sizes must accompany significance tests. When fitting models to data, report goodness-of-fit metrics and visualize residuals.

### VI. Backward Compatibility
New analyses must not break existing functionality. The dual codebase (`code/` for paper scripts, `llm_stylometry/` for the package) must remain consistent. Legacy model naming conventions must continue to work alongside new conventions (e.g., models without `_ntokens=` in their name default to the full token budget).

## Quality Gates

- All tests must pass before merging (`pytest tests/ -v`)
- Code must be formatted (`black .`) and linted (`ruff check .`)
- New analyses require at least one test verifying correctness
- Figures must be generated programmatically, never manually edited
- Pickle files should include the generation command in comments or companion scripts
- Sensitive data (credentials, API keys) must never be committed

## Development Workflow

- Work on feature branches; merge to main via pull request
- Commit frequently with descriptive messages during development
- Pre-computed results (`.pkl`, `.pkl.gz`) should be regenerable from raw data + code
- When pandas version constraints exist for serialized data, prefer format-stable alternatives (CSV, Parquet) or document the constraint prominently

## Governance

This constitution governs all code and analysis in the llm-stylometry repository. It supersedes informal conventions. Amendments require documentation of the rationale and must not weaken scientific rigor guarantees.

**Version**: 1.0.0 | **Ratified**: 2026-03-24 | **Last Amended**: 2026-03-24
