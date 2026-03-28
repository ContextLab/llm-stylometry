# Implementation Plan: Paper Revision — New Analyses & Response Letter

**Branch**: `001-paper-revision-analyses` | **Date**: 2026-03-24 | **Spec**: [spec.md](spec.md)
**Input**: Feature specification from `/specs/001-paper-revision-analyses/spec.md`

## Summary

Address Computational Linguistics reviewer comments by adding three new analyses (dataset-size sweep integration with sigmoid fit, embedding-based comparison, and benchmark feasibility argument), updating the paper (methods, results, discussion, appendix), and drafting a point-by-point response letter. The dataset-size sweep is already complete (PR #51); remaining work is sigmoid fitting, embedding comparison, paper writing, and response letter.

## Technical Context

**Language/Version**: Python 3.10 (CI-authoritative), LaTeX for paper
**Primary Dependencies**: scipy (curve_fit), sentence-transformers, torch, numpy, pandas, matplotlib, seaborn, scikit-learn (cosine_similarity)
**Storage**: Pickle/gzip for model results, PDF for figures, CSV for embedding results
**Testing**: pytest (existing test suite)
**Target Platform**: macOS (96GB M2 Max), Linux (8xA6000 GPU cluster)
**Project Type**: Research scripts + installable package (dual codebase)
**Performance Goals**: Embedding comparison completes in <48 hours on available hardware
**Constraints**: 3 embedding models (137M–4B params); all results reproducible from documented commands
**Scale/Scope**: 8 authors, 84 books, ~30–1000 chunks per book, 3 embedding models

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Principle | Status | Notes |
|-|-|-|
| I. Scientific Accuracy | PASS | All analyses produce verifiable results with documented parameters |
| II. Replicability | PASS | Every analysis scripted with fixed seeds/deterministic protocols; commands documented |
| III. Robust Documentation | PASS | README updates, paper methods, inline docstrings all planned |
| IV. Data Purity | PASS | Embedding comparison uses leave-one-out; cross-entropy uses training-from-scratch |
| V. Statistical Rigor | PASS | Bootstrap CIs for sigmoid; purity/accuracy/confusion for embeddings; 10 seeds for sweep |
| VI. Backward Compatibility | PASS | New scripts don't modify existing code; PR #51 is backward-compatible |

No violations. No complexity tracking needed.

## Project Structure

### Documentation (this feature)

```text
specs/001-paper-revision-analyses/
├── plan.md              # This file
├── spec.md              # Feature specification
├── research.md          # Phase 0: resolved unknowns
├── data-model.md        # Phase 1: data entities
├── quickstart.md        # Phase 1: reproduction guide
└── checklists/
    └── requirements.md  # Spec quality checklist
```

### Source Code (repository root)

```text
code/
├── fit_sigmoid.py               # NEW: Sigmoid fit to accuracy vs tokens (already written)
├── embedding_comparison.py      # NEW: Chunk-level embedding authorship attribution
├── compute_stats.py             # MODIFIED (PR #51): dataset-size stats
├── consolidate_model_results.py # MODIFIED (PR #51): ntokens consolidation
├── experiment.py                # MODIFIED (PR #51): ntokens in model names
├── main.py                      # MODIFIED (PR #51): N_TRAIN_TOKENS env var

llm_stylometry/visualization/
├── t_tests.py                   # MODIFIED (PR #51): ntokens grid/avg figures
├── embedding_comparison.py      # NEW: comparison figure generation

paper/
├── main.tex                     # MODIFIED: methods, results, discussion updates
├── supplement.tex               # MODIFIED: embedding appendix
├── admin/response_letter.tex    # NEW: point-by-point response
├── figs/source/
│   ├── accuracy_vs_tokens_sigmoid.pdf  # NEW (already generated)
│   ├── embedding_comparison.pdf        # NEW: main paper comparison figure
│   ├── embedding_purity.pdf            # NEW: appendix figure
│   └── embedding_confusion.pdf         # NEW: appendix figure

tests/
├── test_dataset_size_support.py # EXISTS (PR #51)
├── test_sigmoid_fit.py          # NEW: verify sigmoid fit correctness
└── test_embedding_comparison.py # NEW: verify embedding pipeline correctness

data/
├── embedding_results/           # NEW: cached embedding classification results
│   ├── nomic-embed-text-v1.5/
│   ├── bge-m3/
│   └── Qwen3-Embedding-4B/

# Shell scripts (repo root)
run_llm_stylometry.sh            # MODIFIED: new figure flags, sentence-transformers dep, auto-run
remote_train_ntokens.sh          # NEW: launch dataset-size sweep on GPU cluster
check_ntokens_status.sh          # NEW: check sweep training progress
sync_ntokens.sh                  # NEW: download ntokens model results
remote_embedding.sh              # NEW: launch embedding comparison on GPU cluster
check_embedding_status.sh        # NEW: check embedding job progress
sync_embeddings.sh               # NEW: download embedding results
```

**Structure Decision**: New scripts go in `code/` (paper scripts layer). New visualization functions go in `llm_stylometry/visualization/` (package layer). Embedding results cached in `data/embedding_results/` per model for reproducibility.

## Phase 0: Research

### Resolved Unknowns

1. **Embedding models selected**: Qwen3-Embedding-4B (4.0B), bge-m3 (568M), nomic-embed-text-v1.5 (137M) — see spec clarifications
2. **Chunk strategy**: 1024 tokens, 128 overlap, chunk-level nearest-neighbor with modal vote
3. **LOO protocol**: Full deterministic leave-one-out (no seeds for embeddings)
4. **Huang et al. citation**: Use (2025) with one-time parenthetical in response letter
5. **Sigmoid fit**: Already implemented and verified (R²=0.978, threshold ≈51K tokens)
6. **Paper structure**: Embedding details in appendix, 1 comparison figure + brief results in main text

### Remaining Research Needed

- Huang et al. (2025) paper details for response letter arguments (fetch from PLoS ONE)
- Benchmark dataset token counts (Blogs50, CCAT50, Guardian, IMDB62) to quantify why they're infeasible

## Phase 1: Implementation Sequence

### Step 1: Merge/integrate PR #51 (P1)
- Merge PR #51 changes into this branch (prefer merge over cherry-pick for ~4300 files)
- Review and address PR #51 code issues (pandas version assertions → convert to Parquet or pin version, missing EOF newline)
- Verify pre-computed data files and model directories are present (no rerun needed)
- Re-generate PR #51 figures (t_test_ntokens_grid.pdf, t_test_avg_ntokens.pdf) to match established paper style; submit for PI review
- Run existing test suite to confirm integration didn't break anything

### Step 2: Update infrastructure (P1)
- Update `run_llm_stylometry.sh`:
  - Add `sentence-transformers` to dependency installation
  - Add new figure flags (6 = sigmoid fit, 7 = embedding comparison) to help text and dispatch
  - Auto-run sigmoid fit and embedding comparison if pre-computed results are missing
  - Only run baseline (intact text) variant for new analyses — no variant flag support needed
- Update `code/generate_figures.py` to support new figure types
- Create 6 new remote scripts (existing scripts unchanged):
  - `remote_train_ntokens.sh` — launch dataset-size sweep on GPU cluster
  - `check_ntokens_status.sh` — check sweep training progress
  - `sync_ntokens.sh` — download ntokens model results from cluster
  - `remote_embedding.sh` — launch embedding comparison on GPU cluster (installs sentence-transformers)
  - `check_embedding_status.sh` — check embedding job progress
  - `sync_embeddings.sh` — download embedding results from cluster
  - All follow existing patterns: SSH setup, conda/pip env, screen-based background execution
- **Deliverable**: Updated CLI, 6 new remote scripts

### Step 3: Finalize sigmoid fit (P1)
- `code/fit_sigmoid.py` already written and verified
- Write `tests/test_sigmoid_fit.py` to verify fit parameters and threshold
- Generate final figure, submit for PI review
- **Deliverable**: `paper/figs/source/accuracy_vs_tokens_sigmoid.pdf`, test

### Step 4: Implement embedding comparison pipeline (P2)
- Write `code/embedding_comparison.py`:
  - Book chunking (1024 tokens, 128 overlap, using GPT-2 tokenizer for consistency)
  - Per-model embedding (sentence-transformers API)
  - Full LOO: for each held-out book, classify each chunk by nearest training chunk
  - Book-level: modal vote, purity, runner-up, per-chunk accuracy
  - Confusion matrix across authors
  - Cache results per model in `data/embedding_results/`
- Write `tests/test_embedding_comparison.py`:
  - Test chunking produces correct number/size of chunks
  - Test nearest-neighbor classification with synthetic embeddings
  - Test modal vote logic
  - Test purity computation
- Run on cluster (estimated 6–12 hours for all 3 models)
- **Deliverable**: Per-model results cached, summary printed

### Step 5: Generate embedding figures (P2)
- Write `llm_stylometry/visualization/embedding_comparison.py`:
  - Main paper figure: bar chart of book-level accuracy per model + our 100% baseline
  - Appendix figures: purity distribution, confusion heatmap, per-author breakdown
- All figures: Helvetica, sns.despine, PDF output, consistent sizing
- Submit all figures for PI visual review
- **Deliverable**: 3 figures (1 main, 2 appendix)

### Step 6: Fetch Huang et al. (2025) details (P2)
- Read the PLoS ONE paper to extract:
  - Their benchmark dataset sizes (tokens per author)
  - Their accuracy numbers
  - Their fine-tuning methodology details
- Use findings to strengthen response letter arguments
- **Deliverable**: Research notes for response letter

### Step 7: Update paper (P3)
- **Methods**: Add subsections for dataset-size analysis, sigmoid fitting, embedding comparison
- **Results**: Add sigmoid figure + results paragraph, embedding comparison figure + ~1 paragraph
- **Discussion**: Expand Huang et al. comparison, add benchmark feasibility argument, data purity argument, unnecessary larger models argument
- **Supplement/Appendix**: Add embedding details section with detailed figures
- **Bibliography**: Verify Huang et al. (2025) PLoS ONE reference
- Verify paper compiles
- **Deliverable**: Updated main.tex, supplement.tex

### Step 8: Draft response letter (P3)
- Create `paper/admin/response_letter.tex`
- Structure: Editor response → R1 → R2 → R3, each point numbered
- Reference specific new figures and analyses
- Key arguments:
  - Training from scratch guarantees data purity (vs. fine-tuning)
  - ~51K tokens/author needed → benchmarks with hundreds of tokens are infeasible
  - GPT-2 achieves 100% → larger models unnecessary
  - Embedding comparison shows our method's advantage
  - Huang et al. (2025) parenthetical citation note
- **Deliverable**: Complete response letter draft

### Step 9: Update README (P3)
- Document dataset-size experiments
- Document sigmoid fit command
- Document embedding comparison (installation, running, expected output)
- **Deliverable**: Updated README.md

### Step 10: Verification pass
- Run full test suite (`pytest tests/ -v`)
- Run `black .` and `ruff check .`
- Verify paper compiles
- Check all new figures exist and are referenced
- Verify all commands in README work
- **Deliverable**: All checks pass

## Complexity Tracking

No constitution violations. No complexity justifications needed.
