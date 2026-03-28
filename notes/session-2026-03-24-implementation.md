# Session Notes: 2026-03-24/25 — Paper Revision Implementation

## Branch: 001-paper-revision-analyses (13 commits)

## COMPLETED
- PR #51 merged + cleaned up (Parquet conversion, version assertions removed, __main__ extracted)
- Sigmoid fit: redesigned figure (per-author dots + bootstrap CI over authors), R²=0.979, ~51K threshold
- t-test ntokens figure: per-author curves with bootstrap CIs over seeds, no legend (panel B)
- Both figures approved by PI, merged into figs/n_tokens.pdf (Panel A + B)
- Embedding pipeline: code + tests (7/7) + per-book checkpoint/resume
- nomic-embed-text-v1.5: 81.0% (68/84), purity=0.666
- bge-m3: 76.2% (64/84), purity=0.694 (Dickens magnet problem)
- Remote scripts: 3 ntokens scripts tested on tensor02 (all 1520 models present)
- CLI: run_llm_stylometry.sh fig 6/7, run_stats.sh, generate_figures.py dispatch
- README: all new analyses documented
- Paper methods + results sections written (embedding PLACEHOLDERs remain)
- Paper discussion expanded (Huang et al. data purity, benchmark feasibility)
- Supplement: embedding appendix with table, purity/confusion figures
- Response letter: full verbatim reviewer comments, interleaved responses, pxfonts, yellow highlights
- Speckit: constitution, spec, plan, 67 tasks
- Black formatting applied across entire codebase
- 15/15 tests passing

## RUNNING
- Qwen3-Embedding-4B: 71/84 books cached, ~3h remaining (large Fitzgerald/Twain books)
- Monitor running in background (checks every 5 min)

## WHEN QWEN3-4B FINISHES
1. Check results: cat data/embedding_results/Qwen_Qwen3-Embedding-4B/summary.json | python3 -m json.tool
2. Fill PLACEHOLDERs:
   - paper/main.tex ~line 546: Qwen accuracy %
   - paper/supplement.tex ~line 324: Qwen accuracy + purity in table
   - paper/admin/response_letter.tex ~line 252: TBD%
3. Verify/update highlighted claims (search: colorbox{yellow}, NOTE.*Qwen, VERIFY):
   - "at most 81%" claims — update if Qwen beats nomic
   - "do not match" / "do not fully capture" — verify still true
   - Supplement interpretation — check if patterns hold for 3rd model
4. Regenerate embedding figures: python code/embedding_comparison.py --figures-only
5. PI review of embedding figures
6. Final: compile paper, run all tests, push

## KEY FILES
- code/fit_sigmoid.py — sigmoid fit + figure (reads from data/model_results_ntokens.pkl.gz)
- code/embedding_comparison.py — embedding pipeline (checkpointed per book)
- code/generate_ntokens_figures.py — t-test ntokens figure
- llm_stylometry/visualization/t_tests.py — generate_t_test_ntokens_figure (bootstrap CIs)
- llm_stylometry/visualization/embedding_comparison.py — 3 figure types
- paper/main.tex — updated Methods, Results, Discussion
- paper/supplement.tex — embedding appendix
- paper/admin/response_letter.tex — point-by-point response (pxfonts, yellow highlights)
- paper/custom.bib — MTEB citation added
- data/sigmoid_fit_results.json — shared between figures
- .ssh/credentials_tensor01.json, .ssh/credentials_tensor02.json — gitignored

## EMBEDDING RESULTS
| Model | Params | Accuracy | Purity | Status |
|-|-|-|-|-|
| nomic-embed-text-v1.5 | 137M | 81.0% (68/84) | 0.666 | COMPLETE |
| bge-m3 | 568M | 76.2% (64/84) | 0.694 | COMPLETE |
| Qwen3-Embedding-4B | 4.0B | TBD | TBD | 71/84 |
| Predictive comparison (ours) | 0.8M x 8 | 100% (84/84) | — | VERIFIED |
