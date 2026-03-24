# Tasks: Paper Revision — New Analyses & Response Letter

**Input**: Design documents from `/specs/001-paper-revision-analyses/`
**Prerequisites**: plan.md, spec.md, research.md, data-model.md, quickstart.md

**Tests**: Included per FR-010 (all new code must include at least one test).

**Organization**: Tasks grouped by user story for independent implementation.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story (US1–US5)
- Exact file paths included

---

## Phase 1: Setup & PR #51 Integration

**Purpose**: Integrate the dataset-size sweep from PR #51 into this branch, clean up issues, and ensure all pre-computed results are available. No rerunning of analyses needed — PR #51 contains all computed results.

### PR #51 Integration Strategy

PR #51 comes from a fork (`harrison-f-stropkay/llm-stylometry-2026`, branch `feature/vary-dataset-size`). It contains source code changes + ~4300 model config/loss log files + 2 pre-generated PDF figures + 2 compressed data files. Integration approach:

1. Merge the PR's changes into this branch
2. Clean up code issues
3. Verify pre-computed data files are present (no rerun needed)

### Tasks

- [x] T001 Fetch PR #51 and merge its changes into the `001-paper-revision-analyses` branch: `gh pr checkout 51` to inspect, then `git merge feature/vary-dataset-size` (prefer merge over cherry-pick for ~4300 files). Resolve any conflicts with main.
- [x] T002 Verify PR #51 pre-computed data files are present: data/model_results_ntokens.pkl.gz, data/t_test_ntokens_cache.pkl.gz, paper/figs/source/t_test_ntokens_grid.pdf, paper/figs/source/t_test_avg_ntokens.pdf
- [x] T003 Verify PR #51 model directories are present: spot-check that models/ contains ntokens model configs and loss logs (e.g., models/austen_tokenizer=gpt2_ntokens=10000_seed=0/config.json)
- [x] T004 Fix PR #51 issue: convert data/model_results_ntokens.pkl.gz to Parquet format (format-stable per constitution). Remove all `assert pd.__version__ == '2.3.3'` checks from code/compute_stats.py, code/consolidate_model_results.py, llm_stylometry/visualization/t_tests.py. Update all load calls to use pd.read_parquet. If Parquet conversion is infeasible, pin pandas version in requirements-dev.txt and document prominently.
- [x] T005 Fix PR #51 issue: add missing newline at EOF in .gitignore
- [x] T006 Fix PR #51 issue: remove `__main__` block from llm_stylometry/visualization/t_tests.py (library module); extract to a standalone script (e.g., code/generate_ntokens_figures.py)
- [x] T007 Fix PR #51 issue: move dead `uv run` command comments from inside `if __name__` blocks to module docstrings in code/compute_stats.py and code/consolidate_model_results.py
- [x] T008 Add PR #51's test file tests/test_dataset_size_support.py; run it to verify: `pytest tests/test_dataset_size_support.py -v`
- [ ] T009 Run existing test suite to confirm PR #51 integration didn't break anything: `pytest tests/ -v --tb=short` *(running)*

- [x] T009a [US1] Re-generate PR #51 figures (paper/figs/source/t_test_ntokens_grid.pdf, paper/figs/source/t_test_avg_ntokens.pdf) to ensure they match the established paper figure style (Helvetica, despine, color palette, sizing conventions). Compare against existing figures (e.g., t_test.pdf, t_test_avg.pdf) and adjust visualization code in llm_stylometry/visualization/t_tests.py if needed.
- [ ] T009b [US1] Visually inspect re-generated ntokens figures — submit for PI review before proceeding
- [x] T009c Update .gitignore to exclude data/embedding_results/ (cached embeddings can be large and should not be committed)

**Checkpoint**: PR #51 fully integrated, code cleaned up, figures style-matched and PI-approved, all tests pass

---

## Phase 2: Foundational (Infrastructure)

**Purpose**: CLI updates and remote scripts that all analyses depend on

- [ ] T010 Update run_llm_stylometry.sh: add `sentence-transformers` to dependency installation section
- [ ] T011 Update run_llm_stylometry.sh: add figure flags `6` (sigmoid fit) and `7` (embedding comparison) to help text, figure list, and dispatch logic
- [ ] T012 Update run_llm_stylometry.sh: add auto-run logic — if pre-computed results missing, run sigmoid fit and/or embedding comparison before generating figures
- [ ] T013 Update code/generate_figures.py: add `--figure 6` and `--figure 7` dispatch to call sigmoid fit and embedding comparison figure generators
- [ ] T014 [P] Create remote_train_ntokens.sh based on remote_train.sh — launch dataset-size sweep on GPU cluster with N_TRAIN_TOKENS env var support (for reproducibility; results already computed)
- [ ] T015 [P] Create check_ntokens_status.sh based on check_remote_status.sh — report progress of ntokens model training jobs
- [ ] T016 [P] Create sync_ntokens.sh based on sync_models.sh — download ntokens model results (configs + loss logs, not weights) from cluster
- [ ] T017 [P] Create remote_embedding.sh based on remote_train.sh — launch embedding comparison on GPU cluster, install sentence-transformers in remote env
- [ ] T018 [P] Create check_embedding_status.sh based on check_remote_status.sh — report embedding job progress per model
- [ ] T019 [P] Create sync_embeddings.sh based on sync_models.sh — download embedding results from data/embedding_results/ on cluster

**Checkpoint**: Infrastructure ready — all analyses can be run locally or on GPU cluster

---

## Phase 3: User Story 1 — Dataset-Size Analysis + Sigmoid Fit (Priority: P1) 🎯 MVP

**Goal**: Produce sigmoid fit with formal statistical analysis using pre-computed sweep results

**Independent Test**: Run `python code/fit_sigmoid.py` → verify R² > 0.95, threshold between 40K–60K tokens, PDF generated

### Tests for User Story 1

- [x] T020 [P] [US1] Write test for sigmoid fit parameters and convergence in tests/test_sigmoid_fit.py — verify R² > 0.95, parameter bounds, bootstrap CI produces valid range
- [x] T021 [P] [US1] Write test for inverse sigmoid threshold computation in tests/test_sigmoid_fit.py — verify find_threshold_tokens returns value between 30K–70K for 95% target

### Implementation for User Story 1

- [x] T022 [US1] Review and finalize code/fit_sigmoid.py — verify RAW_DATA matches PR #51 output, confirm figure style (Helvetica, despine, PDF, color conventions). Add residual diagnostics per constitution V: print residual summary (max, mean, RMSE) and optionally save a residual plot alongside the main figure.
- [x] T023 [US1] Generate paper/figs/source/accuracy_vs_tokens_sigmoid.pdf and visually inspect — submit for PI review
- [x] T024 [US1] Run tests: `pytest tests/test_sigmoid_fit.py -v` — 6/6 passed

**Checkpoint**: Sigmoid fit complete, verified, figure ready for paper

---

## Phase 4: User Story 2 — Embedding-Based Comparison (Priority: P2)

**Goal**: Implement chunk-level embedding attribution pipeline with 3 MTEB models and generate comparison figures

**Independent Test**: Run `python code/embedding_comparison.py --model nomic-ai/nomic-embed-text-v1.5` on local machine → verify per-book accuracy, purity, and confusion output

### Tests for User Story 2

- [ ] T025 [P] [US2] Write test for book chunking logic in tests/test_embedding_comparison.py — verify chunk count, overlap, token sizes for a sample book
- [ ] T026 [P] [US2] Write test for modal vote classification in tests/test_embedding_comparison.py — verify correct author prediction, purity calculation, runner-up identification with synthetic embeddings
- [ ] T027 [P] [US2] Write test for nearest-neighbor chunk classification in tests/test_embedding_comparison.py — verify cosine similarity ranking with known vectors

### Implementation for User Story 2

- [ ] T028 [US2] Implement book chunking in code/embedding_comparison.py — read books from data/cleaned/{author}/*.txt, tokenize with GPT-2 tokenizer, split into 1024-token chunks with 128-token overlap
- [ ] T029 [US2] Implement chunk embedding in code/embedding_comparison.py — load each model via sentence-transformers, embed all chunks, handle OOM gracefully (skip model with warning)
- [ ] T030 [US2] Implement LOO classification in code/embedding_comparison.py — for each held-out book, classify each chunk by nearest training chunk (cosine similarity), compute modal vote, purity, per-chunk accuracy, runner-up, confusion matrix. Tie-breaking: modal vote ties broken by highest average similarity score (fully deterministic, no random component).
- [ ] T031 [US2] Implement results caching in code/embedding_comparison.py — save per-model results to data/embedding_results/{model_name}/ as CSV; load from cache if exists
- [ ] T032 [US2] Implement CLI interface in code/embedding_comparison.py — `--model` for single model, `--figures-only` for figure generation from cached results, default runs all 3 models
- [ ] T033 [US2] Run tests: `pytest tests/test_embedding_comparison.py -v`
- [ ] T034 [US2] Run embedding comparison for nomic-embed-text-v1.5 locally (smallest model, ~30 min) — verify results are reasonable
- [ ] T035 [US2] Run all 3 models on GPU cluster using remote_embedding.sh, monitor with check_embedding_status.sh, sync with sync_embeddings.sh
- [ ] T036 [P] [US2] Implement main paper comparison figure in llm_stylometry/visualization/embedding_comparison.py — bar chart: book-level accuracy per model + our 100% baseline, Helvetica/despine/PDF style
- [ ] T037 [P] [US2] Implement appendix purity figure in llm_stylometry/visualization/embedding_comparison.py — distribution of purity scores per model per author
- [ ] T038 [P] [US2] Implement appendix confusion figure in llm_stylometry/visualization/embedding_comparison.py — heatmap of cross-author chunk confusion patterns per model
- [ ] T039 [US2] Generate all embedding figures and visually inspect — submit for PI review

**Checkpoint**: Embedding comparison complete, all figures ready

---

## Phase 5: User Story 3 — Response Letter (Priority: P2)

**Goal**: Draft point-by-point response addressing editor and 3 reviewers

**Independent Test**: Letter addresses every substantive reviewer point; references specific new analyses

### Implementation for User Story 3

- [ ] T040 [US3] Fetch and analyze Huang et al. (2025) from PLoS ONE — extract benchmark dataset sizes (tokens/author for Blogs50, CCAT50, Guardian, IMDB62), accuracy numbers, fine-tuning methodology details
- [ ] T041 [US3] Draft response to editor/action editor in paper/admin/response_letter.tex — address "how much data needed" question with sigmoid results (~51K tokens), explain why benchmarks are infeasible
- [ ] T042 [US3] Draft response to Reviewer 1 in paper/admin/response_letter.tex — clarify novelty vs Huang et al. (2025) (use parenthetical year note), explain training-from-scratch data purity advantage, address benchmark request with token requirement argument
- [ ] T043 [US3] Draft response to Reviewer 2 in paper/admin/response_letter.tex — push back on larger models: fine-tuning = same as Huang et al. with data purity concerns; GPT-2 already achieves 100% so larger models unnecessary
- [ ] T044 [US3] Draft response to Reviewer 3 in paper/admin/response_letter.tex — address novelty concerns, benchmark request (same argument as R1), reference embedding comparison as additional evidence
- [ ] T045 [US3] Review complete letter for consistency — ensure all new figures/analyses are referenced, arguments are coherent across reviewer responses

**Checkpoint**: Response letter complete, ready for PI review

---

## Phase 6: User Story 4 — Paper Updates (Priority: P3)

**Goal**: Update Methods, Results, Discussion, and Appendix with new analyses

**Independent Test**: Paper compiles without errors; all new figures referenced; discussion addresses Huang et al. and benchmarks

### Implementation for User Story 4

- [ ] T046 [US4] Update Methods section in paper/main.tex — add dataset-size sweep methodology, sigmoid fitting procedure (model form, bounded optimization, bootstrap CI), embedding comparison methodology (chunking, LOO, modal vote)
- [ ] T047 [US4] Update Results section in paper/main.tex — add sigmoid fit results paragraph with figure reference (R², threshold, CI), add ~1 paragraph embedding comparison results with main comparison figure reference
- [ ] T048 [US4] Update Discussion section in paper/main.tex — expand Huang et al. (2025) comparison (training-from-scratch vs fine-tuning, data purity), add benchmark feasibility argument (~51K tokens needed vs hundreds available), add argument that larger models are unnecessary
- [ ] T049 [US4] Add embedding appendix to paper/supplement.tex — detailed methods, chunk-level results, purity distributions, confusion patterns, per-author breakdowns with appendix figure references
- [ ] T050 [US4] Verify Huang et al. (2025) bibliography entry in paper/custom.bib — confirm PLoS ONE citation (not arXiv 2024)
- [ ] T051 [US4] Add figure includes for new figures in paper/main.tex — accuracy_vs_tokens_sigmoid.pdf, embedding_comparison.pdf; and in supplement.tex — embedding_purity.pdf, embedding_confusion.pdf
- [ ] T052 [US4] Compile paper and verify: `cd paper && pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex` — fix any errors or missing references

**Checkpoint**: Paper compiles, all new content integrated

---

## Phase 7: User Story 5 — README & Documentation (Priority: P3)

**Goal**: Update README so a new researcher can reproduce all analyses

**Independent Test**: Follow README instructions to reproduce sigmoid fit and verify output

### Implementation for User Story 5

- [ ] T053 [US5] Update README.md — document dataset-size sweep experiments (improve PR #51's additions with clearer instructions)
- [ ] T054 [US5] Update README.md — document sigmoid fit: command, expected output, figure location
- [ ] T055 [US5] Update README.md — document embedding comparison: installation (sentence-transformers), running (local + cluster), expected output, figure locations
- [ ] T056 [US5] Update README.md — document new remote scripts (remote_train_ntokens.sh, remote_embedding.sh, check/sync scripts) with usage examples

**Checkpoint**: README complete, reproduction instructions verified

---

## Phase 8: Polish & Verification

**Purpose**: Final validation across all work

- [ ] T057 Run formatter and linter: `black . && ruff check .`
- [ ] T058 Run full test suite after formatting/linting fixes: `pytest tests/ -v --tb=short`
- [ ] T059 [P] Validate all 6 new remote scripts pass syntax check: `bash -n remote_train_ntokens.sh check_ntokens_status.sh sync_ntokens.sh remote_embedding.sh check_embedding_status.sh sync_embeddings.sh`
- [ ] T060 Verify `python code/embedding_comparison.py --figures-only` works end-to-end from cached results (FR-011 single-command reproducibility)
- [ ] T061 Verify all new figures exist in paper/figs/source/ — accuracy_vs_tokens_sigmoid.pdf, t_test_ntokens_grid.pdf, t_test_avg_ntokens.pdf, embedding_comparison.pdf, embedding_purity.pdf, embedding_confusion.pdf
- [ ] T062 Verify paper compiles cleanly with no warnings about missing references
- [ ] T063 Verify all commands in quickstart.md work end-to-end
- [ ] T064 Final PI review of all figures and response letter

---

## Dependencies & Execution Order

### Phase Dependencies

- **Phase 1 (Setup)**: No dependencies — start immediately
- **Phase 2 (Infrastructure)**: Depends on Phase 1 completion
- **Phase 3 (US1 — Sigmoid)**: Depends on Phase 1 (needs PR #51 data); can start in parallel with Phase 4
- **Phase 4 (US2 — Embeddings)**: Depends on Phase 2 (needs remote scripts); can start in parallel with Phase 3
- **Phase 5 (US3 — Response Letter)**: Depends on Phase 3 + Phase 4 results
- **Phase 6 (US4 — Paper)**: Depends on Phase 3 + Phase 4 + Phase 5
- **Phase 7 (US5 — README)**: Depends on Phase 2 (infrastructure); can partially parallel with Phase 3–6
- **Phase 8 (Polish)**: Depends on all previous phases

### User Story Dependencies

- **US1 (Sigmoid)**: Independent after Phase 1 ← start here (data already available from PR #51)
- **US2 (Embeddings)**: Independent after Phase 2 ← can parallel with US1
- **US3 (Response Letter)**: Needs US1 + US2 results
- **US4 (Paper)**: Needs US1 + US2 + US3
- **US5 (README)**: Needs Phase 2; can partially parallel

### Parallel Opportunities

**After Phase 1 completes:**
- T020, T021 (sigmoid tests) can start immediately
- T014–T019 (all 6 remote scripts) can run in parallel

**After Phase 2 completes:**
- T025, T026, T027 (embedding tests) can run in parallel
- T022 (sigmoid finalize) and T028–T032 (embedding pipeline) can run in parallel
- T036, T037, T038 (all embedding figures) can run in parallel
- T053–T056 (all README sections) can run in parallel

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Phase 1: Integrate PR #51 + cleanup
2. Phase 3: Sigmoid fit + tests + figure
3. **STOP and VALIDATE**: R² > 0.95, threshold in expected range, figure looks good

### Full Delivery

1. Phase 1: Integrate PR #51
2. Phase 2: Infrastructure (CLI, 6 remote scripts) — in parallel with Phase 3
3. Phase 3: Sigmoid fit (fast, data already available)
4. Phase 4: Embeddings — launch GPU cluster job early (T035), write code while it runs
5. Phase 5: Response letter (once results are in)
6. Phase 6: Paper updates
7. Phase 7: README
8. Phase 8: Final verification

---

## Notes

- [P] tasks = different files, no dependencies
- [Story] label maps task to specific user story
- PI must visually inspect all figures before they're finalized (T009b, T023, T039, T064)
- GPU cluster work (T035) is the longest task (~6–12 hours); launch early
- Dataset-size sweep results are already computed (PR #51) — no rerun needed
- Commit after each completed phase
