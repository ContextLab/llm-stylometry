# Quickstart: Reproducing Paper Revision Analyses

## Prerequisites

```bash
pip install -e .
pip install sentence-transformers
cd tests && python create_test_data.py && cd ..
```

## 1. Dataset-Size Sweep (PR #51 — already computed)

Results are in `data/model_results_ntokens.pkl.gz`. To regenerate stats:

```bash
uv run --no-project --python 3.11 --with pandas==2.3.3 --with numpy --with scipy --with tqdm \
  python code/compute_stats.py --data data/model_results_ntokens.pkl.gz --n-tokens
```

## 2. Sigmoid Fit

```bash
python code/fit_sigmoid.py
# Output: paper/figs/source/accuracy_vs_tokens_sigmoid.pdf
# Prints: fit parameters, R-squared, threshold for >=95% accuracy, bootstrap CI
```

## 3. Embedding Comparison

```bash
# Run all 3 models (estimated 6-12 hours on A6000 cluster)
python code/embedding_comparison.py

# Run a single model
python code/embedding_comparison.py --model nomic-ai/nomic-embed-text-v1.5

# Generate figures from cached results
python code/embedding_comparison.py --figures-only
```

## 4. Run Tests

```bash
pytest tests/test_sigmoid_fit.py -v
pytest tests/test_embedding_comparison.py -v
pytest tests/ -v  # full suite
```

## 5. Verify Paper Compiles

```bash
cd paper && pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex && cd ..
```
