# Session Notes: 2026-03-24 — Paper Revision Implementation

## Status
- On branch: `001-paper-revision-analyses`
- Speckit artifacts complete: spec.md, plan.md, tasks.md, research.md, data-model.md, quickstart.md, constitution.md
- Analysis complete: Reviewed all issues, 14 remediation edits applied
- Starting implementation of 67 tasks across 8 phases

## Completed Before Implementation
- PR #51 code review (findings documented)
- Sigmoid fit script written and verified (code/fit_sigmoid.py): R²=0.978, threshold ~51K tokens
- Speckit constitution created
- MTEB leaderboard research: selected 3 models (Qwen3-4B, bge-m3, nomic-v1.5)
- Full spec clarification (4 questions answered)

## Key Decisions
- Embedding: chunk-level nearest-neighbor with modal vote (not mean-pooling)
- LOO: full deterministic leave-one-out for embeddings
- Chunks: 1024 tokens, 128 overlap, min 3 chunks per book
- Pickle → Parquet conversion for version stability
- 6 new remote scripts (separate from existing scripts)
- Figures 6 (sigmoid) and 7 (embeddings) added to CLI
- Embedding details in appendix, 1 comparison figure in main paper
- Huang et al. cited as (2025) with parenthetical year note

## Progress
### Phase 1 (Setup & PR #51 Integration) — ~90% complete
- T001-T008: DONE (merge, verify, Parquet conversion, code cleanup)
- T009: Full test suite running (waiting)
- T009a: ntokens figures re-generated from Parquet cache ✓
- T009b: PI review of ntokens figures — WAITING FOR PI
- T009c: .gitignore updated ✓

### Phase 3 (US1 — Sigmoid Fit) — COMPLETE (redesigned)
- T020-T024: All done. 7/7 tests pass.
- Figure REDESIGNED: single panel with per-author colored dots, black sigmoid curve, 95% CI ribbon (bootstrap over authors), labeled threshold line
- Residual diagnostics: RMSE=1.55%, max=3.64%
- Bootstrap CI for threshold (author-resampled): [11,844, 70,098] tokens — wider than token-level bootstrap because it captures between-author variability
- Fit: y = 69.8 + 30.9 / (1 + exp(-5.43*(log10(x) - 4.43))), R²=0.978, threshold ≈51K tokens

### Phase 2 (Infrastructure) — NOT STARTED
- T010-T019: CLI updates and 6 remote scripts

### Phase 4 (US2 — Embeddings) — NOT STARTED
- Need to write embedding_comparison.py

### Figures — APPROVED
1. paper/figs/source/accuracy_vs_tokens_sigmoid.pdf — Panel A (approved)
2. paper/figs/source/t_test_ntokens.pdf — Panel B (approved)
3. paper/figs/n_tokens.pdf — Merged 2-panel figure (PI created manually)
- Old figures (t_test_ntokens_grid.pdf, t_test_avg_ntokens.pdf) removed/replaced

### Phase 4 (US2 — Embeddings) — Code written, nomic running
- code/embedding_comparison.py — complete (chunking, LOO, modal vote, caching, CLI)
- llm_stylometry/visualization/embedding_comparison.py — complete (3 figure types)
- tests/test_embedding_comparison.py — 7/7 pass
- nomic-embed-text-v1.5 running locally (~27 min, 51/84 books done)
- After nomic: launch bge-m3 (~1.8h) and Qwen3-4B (~13h) in background

### Phase 5 (US3 — Response Letter) — Research done
- Huang et al. (2025) analyzed: notes/huang_et_al_2025_notes.md
- Key arguments ready: data purity, benchmark infeasibility (~51K tokens needed vs 100-1000 in benchmarks)
- Waiting for embedding results before drafting

### Phase 6 (US4 — Paper) — Methods and partial results written
- Methods: 2 new subsections added (data requirements, embedding comparison)
- Results: Dataset-size results written with figure reference, embedding results left as PLACEHOLDER
- Discussion: Expanded Huang et al. comparison, benchmark feasibility argument, embedding placeholder
- Bibliography: MTEB citation added, Huang et al. (2025) verified as PLoS ONE
- Figure: \ref{fig:ntokens} referencing figs/n_tokens.pdf

### Remaining work
- Wait for nomic results → examine → update paper placeholders
- Launch bge-m3 and Qwen3-4B
- Draft response letter
- Write remote scripts (Phase 2)
- Update README
- Final verification pass

## Resubmission Deadline
2026-04-09 (~2 weeks)
