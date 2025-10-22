# Data Directory

Contains text data and consolidated model results.

## Structure

```
data/
├── raw/                    # Original Project Gutenberg texts (with headers/footers)
├── cleaned/                # Preprocessed texts (headers/footers removed)
│   ├── {author}/          # One directory per author
│   ├── content_only/      # Function words masked as FUNC
│   ├── function_only/     # Content words masked as CONTENT
│   ├── pos_only/          # Words replaced with POS tags
│   ├── contested/         # Disputed authorship texts (Baum/Thompson)
│   ├── non_oz_baum/       # Non-Oz works by Baum
│   └── non_oz_thompson/   # Non-Oz works by Thompson
├── model_results.pkl               # Consolidated baseline results
├── model_results_content.pkl       # Content-only variant results
├── model_results_function.pkl      # Function-only variant results
├── model_results_pos.pkl           # POS variant results
└── classifier_results/             # Text classification results (gitignored)
    ├── baseline.pkl
    ├── content.pkl
    ├── function.pkl
    └── pos.pkl
```

## Authors

8 authors with 7-14 books each (84 books total):
- Austen (7 books)
- Baum (14 books)
- Dickens (14 books)
- Fitzgerald (8 books)
- Melville (10 books)
- Thompson (13 books)
- Twain (6 books)
- Wells (12 books)

## Creating Variant Data

Generate variant-transformed texts:
```bash
python code/create_analysis_variants.py all
```

## Consolidating Model Results

Combine loss logs from all models into single pkl file:
```bash
python code/consolidate_model_results.py              # Baseline
python code/consolidate_model_results.py --variant content  # Content-only
```

See main README for data sources and preprocessing details.
