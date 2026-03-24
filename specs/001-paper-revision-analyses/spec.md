# Feature Specification: Paper Revision — New Analyses & Response Letter

**Feature Branch**: `001-paper-revision-analyses`
**Created**: 2026-03-24
**Status**: Draft
**Input**: Address reviewer comments for Computational Linguistics resubmission (arXiv:2510.21958). Add dataset-size analysis, sigmoid fit, embedding comparison, response letter, and paper updates.

## User Scenarios & Testing *(mandatory)*

### User Story 1 — Dataset-Size Analysis Integration (Priority: P1)

A researcher runs the dataset-size sweep (PR #51) to determine how many training tokens per author are needed for reliable authorship attribution. The results are consolidated, visualized, and integrated into the paper with formal statistical analysis (sigmoid fit).

**Why this priority**: This directly answers the editor's top question ("how much data is needed?") and provides the quantitative foundation for pushing back on benchmark requests.

**Independent Test**: Run `python code/fit_sigmoid.py` and verify the sigmoid fit produces R² > 0.95, the 95% accuracy threshold is between 40K–60K tokens, and a publication-quality PDF figure is generated.

**Acceptance Scenarios**:

1. **Given** the accuracy-vs-tokens data from the sweep (2,500 to 643,041 tokens), **When** fitting a 4-parameter sigmoid to accuracy vs. log₁₀(tokens), **Then** the fit reports R² > 0.95, parameter confidence intervals, and the minimum tokens for ≥95% expected accuracy with bootstrap CI.
2. **Given** the sigmoid fit results, **When** generating the figure, **Then** the PDF matches the paper's visual style (Helvetica, despine, color conventions) and shows raw data points, fitted curve, and 95% threshold annotation.
3. **Given** the fit results, **When** the paper methods/results sections are updated, **Then** the sigmoid parameters, R², threshold estimate, and bootstrap CI are reported in the text and the figure is referenced.

---

### User Story 2 — Embedding-Based Comparison Method (Priority: P2)

A researcher runs an embedding-based authorship attribution pipeline as a comparison baseline. Multiple embedding models from the MTEB leaderboard are used to embed each book, and nearest-neighbor classification determines authorship. Results are compared against the cross-entropy approach.

**Why this priority**: Addresses reviewer request for comparison with existing methods (Issue #50, point 3). Provides concrete evidence of our method's effectiveness relative to a strong modern baseline.

**Independent Test**: Run the embedding comparison script on the 8-author corpus, verify it produces per-model accuracy tables and a comparison figure, and confirm results are reproducible across runs (fixed seeds).

**Acceptance Scenarios**:

1. **Given** the 8-author book corpus and a set of MTEB-leaderboard embedding models, **When** running leave-one-out embedding classification for each model, **Then** per-model accuracy is computed and reported.
2. **Given** multiple embedding model results, **When** generating figures, **Then**: (a) ONE comparison figure goes in the main paper showing book-level accuracy by model alongside our method's 100% baseline, and (b) detailed figures (purity distributions, confusion patterns, per-author breakdowns) go in a supplementary appendix.
3. **Given** the embedding results, **When** updating the paper, **Then** the main text includes: methods description, ~1 paragraph of results, and discussion of why our approach outperforms. Detailed chunk-level analyses, confusion matrices, and purity statistics go in the appendix.

---

### User Story 3 — Point-by-Point Response Letter (Priority: P2)

The PI drafts a response letter addressing the editor, action editor, and 3 reviewers. The letter incorporates results from the new analyses and presents clear arguments for the paper's novelty and contribution.

**Why this priority**: Required for resubmission (deadline: 2026-04-09). Must incorporate all new analysis results.

**Independent Test**: The response letter addresses every substantive point raised by each reviewer, references specific new figures/analyses, and can be read as a standalone document.

**Acceptance Scenarios**:

1. **Given** reviewer comments (Issue #50), **When** drafting the response, **Then** every substantive point from R1, R2, R3, and the editor is addressed with a numbered response.
2. **Given** the new dataset-size analysis, **When** responding to benchmark requests, **Then** the letter explains why standard benchmarks (Blogs50, CCAT50, Guardian, IMDB62) are infeasible: our method requires ~51K tokens/author but those benchmarks provide only hundreds of tokens/author.
3. **Given** R2's suggestion to use larger models (LLaMA, Qwen, DeepSeek), **When** drafting the response, **Then** the letter argues that: (a) fine-tuning = same approach as Huang et al. with data purity concerns, and (b) GPT-2 already achieves 100% accuracy, so larger models are unnecessary for this task.

---

### User Story 4 — Paper Section Updates (Priority: P3)

The paper's Methods, Results, and Discussion sections are updated to incorporate all new analyses, figures, and arguments.

**Why this priority**: Depends on completion of the analyses (P1, P2) and response letter framing (P2).

**Independent Test**: The paper compiles without errors, all new figures are referenced, and the discussion section addresses the relationship to Huang et al. (2025) and benchmark feasibility.

**Acceptance Scenarios**:

1. **Given** the sigmoid fit results, **When** updating the Methods section, **Then** the sigmoid fitting methodology is described (model form, optimization, bootstrap CI procedure).
2. **Given** the embedding comparison results, **When** updating the Results section, **Then** ~1 paragraph summarizes book-level accuracy across models with a reference to the comparison figure. Detailed chunk-level analyses go in a supplementary appendix.
3. **Given** all new analyses, **When** updating the Discussion, **Then** the text addresses: (a) data requirements vs. benchmark feasibility, (b) training-from-scratch vs. fine-tuning (data purity), (c) why larger models are unnecessary.

---

### User Story 5 — README and Documentation Updates (Priority: P3)

The repository README is updated to document the new analyses, their commands, and their outputs.

**Why this priority**: Important for reproducibility but not blocking the paper revision.

**Independent Test**: A new user can follow the README instructions to reproduce every new analysis and figure.

**Acceptance Scenarios**:

1. **Given** the updated README, **When** a researcher follows the dataset-size experiment instructions, **Then** they can reproduce the sweep, sigmoid fit, and figures.
2. **Given** the updated README, **When** a researcher follows the embedding comparison instructions, **Then** they can install dependencies, run the comparison, and generate the comparison figure.

---

### Edge Cases

- What happens when an embedding model cannot fit in available memory (96GB RAM or A6000 48GB)?
  - The pipeline should skip that model with a warning and continue with others.
- What happens when a book is too short to produce meaningful chunks for embedding?
  - Minimum chunk count threshold of 3; books producing fewer than 3 chunks (< ~2,800 tokens) are flagged with a warning but still included.
- What happens when the sigmoid fit fails to converge?
  - Report the failure with diagnostics rather than silently falling back.
- What happens when leave-one-out produces only 1 training book for an author (e.g., Twain has only 6 books)?
  - This is valid — the comparison still works with 5 training books. Document the minimum.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The sigmoid fit MUST use scipy.optimize.curve_fit with bounded parameters and report L, K, b, m with standard errors
- **FR-002**: The sigmoid fit MUST compute and report the minimum tokens for ≥95% accuracy with a bootstrap 95% confidence interval (≥1000 bootstrap iterations). Per constitution V, the fit MUST also report residual diagnostics (printed residual summary and optional residual plot) to verify goodness-of-fit beyond R².
- **FR-003**: The embedding comparison MUST use these 3 MTEB-leaderboard models spanning small/medium/large tiers: (1) **Qwen/Qwen3-Embedding-4B** (4.0B params, 32K context, MTEB 69.5), (2) **BAAI/bge-m3** (568M params, 8K context, MTEB 59.6), (3) **nomic-ai/nomic-embed-text-v1.5** (137M params, 8K context, MTEB 44.1). Estimated total compute: 6–12 hours on A6000 cluster.
- **FR-004**: The embedding comparison MUST use full leave-one-out cross-validation: for each author, hold out each book exactly once, embed all remaining books as the training set, classify the held-out book via nearest-neighbor. This is deterministic (no seed-based subsampling needed).
- **FR-005**: The embedding comparison MUST chunk books into 1024-token windows with 128-token overlap. For each held-out book, each chunk is classified by nearest-neighbor (cosine similarity) against all training-set chunks. The book-level prediction is the modal author across its chunks. Diagnostics MUST include: (a) purity (fraction of chunks assigned to modal author), (b) per-chunk accuracy, (c) runner-up author and margin, (d) confusion patterns between author pairs.
- **FR-006**: All new figures MUST match the paper's established visual style (Helvetica font, sns.despine, PDF output, consistent figure sizing)
- **FR-007**: All new figures MUST be manually examined by the PI before inclusion in the paper
- **FR-008**: The response letter MUST address every substantive point from the editor, action editor, and all 3 reviewers
- **FR-009**: The paper and response letter MUST cite Huang et al. (2025) with the correct PLoS ONE reference. The response letter MUST include a one-time parenthetical on first mention: "(referred to as Huang et al. 2024 in reviewer comments; now published as Huang et al. 2025 in PLoS ONE)"
- **FR-010**: All new code MUST include at least one test verifying correctness of core computations
- **FR-011**: All new analyses MUST be reproducible from a single script invocation with documented commands
- **FR-012**: `run_llm_stylometry.sh` MUST be updated to: (a) install sentence-transformers as a dependency during setup, (b) support new figure flags for sigmoid fit and embedding comparison figures, (c) auto-run sigmoid fit and embedding comparison if pre-computed results are not present, (d) only run baseline (intact text) variant for the new analyses
- **FR-013**: Six new remote scripts MUST be created (based on existing `remote_train.sh`, `check_remote_status.sh`, and `sync_models.sh` patterns). Existing scripts remain unchanged. New scripts: (a) `remote_train_ntokens.sh` — run dataset-size sweep on GPU cluster, (b) `check_ntokens_status.sh` — check sweep training progress, (c) `sync_ntokens.sh` — sync ntokens model results from cluster, (d) `remote_embedding.sh` — run embedding comparison on GPU cluster (must install sentence-transformers), (e) `check_embedding_status.sh` — check embedding job progress, (f) `sync_embeddings.sh` — sync embedding results from cluster. All scripts must follow existing patterns for SSH setup, environment configuration, and screen-based background execution.
- **FR-014**: The new figure flags in `run_llm_stylometry.sh` MUST follow the existing numbering convention (e.g., 6 for sigmoid fit, 7 for embedding comparison) and be listed in the help text. If a variant flag (-co, -fo, -pos) is passed with figure 6 or 7, the script MUST print a warning ("Figure N is baseline-only; variant flag ignored") and proceed with baseline.

### Key Entities

- **Accuracy-vs-Tokens Data**: Token counts (2,500–643,041), per-seed attribution accuracy (correct/total), grouped by n_train_tokens. Source: PR #51 results.
- **Sigmoid Model**: 4-parameter sigmoid fit to accuracy ~ log₁₀(tokens). Parameters: L (lower asymptote), K (range), b (steepness), m (midpoint).
- **Chunk Embeddings**: Per-chunk vector representations. Each book produces ~30–1000 chunks depending on length. All chunks from non-held-out books form the training pool for nearest-neighbor classification.
- **Embedding Classification Results**: Per-model, per-book results including: book-level accuracy (modal vote), chunk-level accuracy, purity (fraction of chunks voting for modal author), runner-up author and margin, and cross-author confusion patterns.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Sigmoid fit achieves R² > 0.95 and the 95% accuracy threshold falls between 30K–70K tokens with a bootstrap CI width < 20K tokens
- **SC-002**: At least 3 embedding models are evaluated, and results are presented in a comparison figure alongside the cross-entropy approach's 100% accuracy
- **SC-003**: The response letter addresses 100% of substantive reviewer points with specific references to new analyses
- **SC-004**: All new figures pass manual visual inspection by the PI (correct style, readable labels, no rendering artifacts)
- **SC-005**: Every new analysis can be reproduced from the repository using documented commands, producing identical numerical results (given fixed seeds)
- **SC-006**: The paper compiles without errors and all new figure references resolve correctly
- **SC-007**: Total compute time for the embedding comparison is under 48 hours on available hardware (96GB M2 Max or 8xA6000 cluster)

## Clarifications

### Session 2026-03-24

- Q: What chunk size and overlap for embedding books? → A: 1024 tokens with 128-token overlap
- Q: Embedding leave-one-out protocol? → A: Full LOO (hold out each book once per author, deterministic, no seeds)
- Q: How to handle Huang et al. citation year discrepancy? → A: Use "Huang et al. (2025)" with one-time parenthetical noting the 2024→2025 year change
- Q: Embedding classification strategy? → A: Chunk-level nearest-neighbor with modal vote for book-level prediction (not mean-pooling). Report purity, per-chunk accuracy, confusion patterns.
- Q: How to handle pandas version constraint for pkl.gz files? → A: Convert model_results_ntokens.pkl.gz to Parquet format (format-stable, per constitution Development Workflow). Remove all `assert pd.__version__` checks. If Parquet conversion is infeasible, pin pandas version in requirements-dev.txt and document prominently.
- Q: Is the embedding comparison fully deterministic? → A: Yes. Embedding forward passes are deterministic, LOO has no random selection, and nearest-neighbor ties are broken by highest similarity score (not random). No seeds needed.
- Q: Minimum chunk count for embedding? → A: N=3. Books producing fewer than 3 chunks (i.e., fewer than ~2,800 tokens) are flagged with a warning. This ensures modal vote has enough data points.

## Assumptions

- PR #51 (dataset-size sweep) will be merged or its results are available on the `feature/vary-dataset-size` branch. The accuracy data is treated as given input.
- The 8-author corpus in `data/cleaned/` is the authoritative dataset for all new analyses.
- The `sentence-transformers` library is available and sufficient for running MTEB-leaderboard models locally.
- The paper's LaTeX source in `paper/main.tex` is the current working draft and can be edited directly.
- Huang et al. (2025) is the correct citation year (PLoS ONE publication), not 2024 (arXiv preprint).
- The PI will manually review all figures before they are finalized — no figure is considered complete until visually inspected.
- The resubmission deadline is 2026-04-09, giving approximately 2 weeks for all work.
