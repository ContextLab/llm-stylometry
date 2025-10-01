# Test Suite

This directory contains test scripts and utilities for the LLM Stylometry project.

## Test Model Generation

### Quick Test Models (For Development)

Generate a small set of test models for quick verification:

```bash
python tests/create_test_models.py
```

**Default configuration:**
- 3 authors (fitzgerald, twain, austen)
- 3 seeds (42, 43, 44)
- 4 variants (baseline, content, function, pos)
- Total: 36 models (~1 hour)

**Custom configuration:**
```bash
python tests/create_test_models.py \
    --authors baum thompson \
    --seeds 0 1 2 \
    --variants baseline content
```

### Comprehensive Test Suite (For Full Validation)

Generate complete test dataset for statistical validation:

```bash
echo "y" | python tests/create_test_models.py \
    --authors baum thompson austen dickens fitzgerald melville twain wells \
    --seeds 0 1 2 3 4 5 6 7 8 9 \
    --variants baseline content function pos
```

**Configuration:**
- 8 authors (all from constants.py)
- 10 seeds (0-9, required for statistical tests)
- 4 variants (baseline + 3 analysis variants)
- Total: 320 models (~10-11 hours)

**Why 10 seeds?**
The `compute_stats.py` statistical tests require exactly 10 seeds:
- Average t-test: Tests mean t-statistic across 10 seeds
- Sufficient power: Provides robust statistics (n=10 per author)

## Monitoring Training Progress

Check training status in real-time:

```bash
python tests/check_test_model_progress.py
```

**Output:**
```
============================================================
Test Model Training Progress
============================================================
Total models: 82 / 320 (25.6%)

By variant:
  baseline  :  80 /  80 (100.0%)
  content   :   1 /  80 (1.2%)
  function  :   1 /  80 (1.2%)
  pos       :   0 /  80 (0.0%)

By author:
  baum      :  12 /  40 (30.0%)
  thompson  :  10 /  40 (25.0%)
  ...
============================================================
```

**Use case:** Run periodically to monitor long-running training jobs.

## Integration Tests

### Quick Integration Tests (~30 seconds)

Fast tests covering all major functionality:

```bash
python tests/test_variant_quick.py
```

**Tests:**
1. Data loading with variant column
2. Visualization function signatures
3. Variant filtering logic
4. compute_stats.py variant support
5. Shell script variant flags

### Full Integration Tests (~5 minutes)

Comprehensive tests including figure generation:

```bash
python tests/test_variant_integration.py
```

**Tests:** All quick tests plus actual figure generation for each variant.

## Statistical Validation

After generating comprehensive test suite, verify all statistical tests work:

```bash
# Baseline statistics
python code/compute_stats.py

# Content variant
python code/compute_stats.py --variant content

# Function variant
python code/compute_stats.py --variant function

# Part-of-speech variant
python code/compute_stats.py --variant pos
```

**Requirements for successful statistics:**
- **Twain threshold test**: ≥10 self-losses, ≥70 other-losses per epoch
- **Average t-test**: 10 seeds × 8 authors (exact)
- **Author comparison table**: ≥10 self-losses, ≥70 other-losses per author

The comprehensive test suite (320 models) provides sufficient data for all tests.

## Test Data Files

- `tests/data/test_model_results.pkl` - Initial test data (36 models)
- `tests/data/test_model_results_full.pkl` - Extended test data (116 models)
- `data/model_results.pkl` - Full/comprehensive test data (320 models when complete)

## Workflow for Issue Validation

1. **Generate test data** (if needed):
   ```bash
   # Quick test (1 hour)
   python tests/create_test_models.py

   # OR comprehensive test (10 hours)
   echo "y" | python tests/create_test_models.py --authors baum thompson austen dickens fitzgerald melville twain wells --seeds 0 1 2 3 4 5 6 7 8 9 --variants baseline content function pos
   ```

2. **Monitor progress**:
   ```bash
   watch -n 60 python tests/check_test_model_progress.py
   ```

3. **Consolidate results** (when complete):
   ```bash
   python code/consolidate_model_results.py
   ```

4. **Run integration tests**:
   ```bash
   python tests/test_variant_quick.py
   ```

5. **Validate statistics**:
   ```bash
   python code/compute_stats.py
   python code/compute_stats.py --variant content
   ```

6. **Verify figures**:
   ```bash
   ./run_llm_stylometry.sh -f 1a
   ./run_llm_stylometry.sh -f 1a -co
   ```

## Test Model Specifications

All test models use small configurations for fast training:

- **Architecture**: GPT-2
- **Layers**: 2
- **Embedding dim**: 64
- **Context length**: 128
- **Attention heads**: 2
- **Batch size**: 4
- **Training tokens**: 5,000-10,000
- **Epochs**: 3-50 (depending on test type)

These small models train in 2-3 minutes each while providing sufficient diversity for statistical testing.

## Troubleshooting

### "Insufficient data for t-test"

You need more seeds or authors. Use the comprehensive test suite:
```bash
# Requires 10 seeds × 8 authors
echo "y" | python tests/create_test_models.py --authors baum thompson austen dickens fitzgerald melville twain wells --seeds 0 1 2 3 4 5 6 7 8 9 --variants baseline content function pos
```

### Training takes too long

For quick development iteration:
- Use fewer seeds: `--seeds 0 1 2`
- Use fewer authors: `--authors fitzgerald twain`
- Use fewer variants: `--variants baseline content`

### "No variant column in data"

Run consolidation to add variant column:
```bash
python code/consolidate_model_results.py
```

## Notes

- Test models are stored in `models/` directory
- Each model takes ~2-3 minutes to train
- Use background training for large test suites: `nohup python ... &`
- Check progress anytime with `check_test_model_progress.py`
