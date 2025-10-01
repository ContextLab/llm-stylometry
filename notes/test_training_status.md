# Comprehensive Test Model Training Status

## Configuration

- **Total models**: 320 (8 authors × 10 seeds × 4 variants)
- **Model size**: 2 layers, 64 embedding dim, 5000 tokens, 50 epochs
- **Estimated time**: ~10-11 hours total (~2 min/model)
- **Started**: 2025-09-30 14:06 (background process ID: 7510ec)

## Purpose

Generate sufficient test data for all statistical tests in `compute_stats.py`:
1. Twain threshold test: requires ≥10 self-losses, ≥70 other-losses
2. Average t-test: requires 10 seeds × 8 authors
3. Author comparison table: requires ≥10 self-losses, ≥70 other-losses

## Progress Tracking

Run: `python tests/check_test_model_progress.py`

### Initial Status (14:06)
- Baseline: 80/80 complete
- Content: 1/80 starting
- Total: 81/320 (25%)

## When Complete

1. Run consolidation:
   ```bash
   python code/consolidate_model_results.py
   ```

2. Test statistics:
   ```bash
   python code/compute_stats.py
   python code/compute_stats.py --variant content
   python code/compute_stats.py --variant function
   python code/compute_stats.py --variant pos
   ```

3. Update test suite to use comprehensive data

4. Run integration tests:
   ```bash
   python tests/test_variant_quick.py
   ```

5. Close issue #16 if all tests pass

## Files

- Training script: `tests/create_test_models.py`
- Progress checker: `tests/check_test_model_progress.py`
- Background process: 7510ec
- Output location: `models/` directory
