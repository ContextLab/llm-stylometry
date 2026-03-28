# Research: Paper Revision Analyses

## Embedding Model Selection

**Decision**: Use 3 models spanning small/medium/large tiers
**Rationale**: Covers a range of model sizes to show how embedding quality scales for authorship attribution. All open-source, locally runnable, from MTEB leaderboard top performers.
**Alternatives considered**: 5-model set (rejected: longer compute, diminishing marginal insight), single model (rejected: can't show scaling trend), proprietary models (rejected: not reproducible)

| Model | Params | MTEB Mean | Max Tokens | Memory (fp16) |
|-|-|-|-|-|
| nomic-ai/nomic-embed-text-v1.5 | 137M | 44.1 | 8,192 | ~0.5GB |
| BAAI/bge-m3 | 568M | 59.6 | 8,194 | ~2.3GB |
| Qwen/Qwen3-Embedding-4B | 4.0B | 69.5 | 32,768 | ~16GB |

## Chunk Strategy

**Decision**: 1024 tokens, 128-token overlap
**Rationale**: Larger chunks preserve more context per passage. All 3 models support 8K+ tokens so 1024 is well within limits. Overlap prevents information loss at boundaries.
**Alternatives considered**: 512/64 (standard for MTEB benchmarks but less context), 256/32 (too granular), model-adaptive (unfair comparison)

## Classification Protocol

**Decision**: Chunk-level nearest-neighbor with modal vote for book-level prediction
**Rationale**: Preserves granularity — each chunk independently votes. Modal vote is robust to outlier chunks. Enables rich diagnostics (purity, confusion patterns, per-chunk accuracy).
**Alternatives considered**: Mean-pool embeddings then nearest-neighbor (too lossy — averages away distinguishing features), majority of k-nearest neighbors (adds hyperparameter without clear benefit for this comparison)

## Leave-One-Out Protocol

**Decision**: Full deterministic LOO (hold out each book exactly once)
**Rationale**: Embedding is deterministic (no training variance), so seed-based subsampling adds nothing. Full LOO gives maximum evaluation coverage (84 classification decisions).
**Alternatives considered**: Seed-matched 10-fold (matches cross-entropy protocol exactly but artificially limits evaluation)

## Sigmoid Fit

**Decision**: 4-parameter sigmoid y = L + K / (1 + exp(-b*(log₁₀(x) - m)))
**Rationale**: Natural model for accuracy saturation curves. log₁₀ transform linearizes the token scale. Bootstrap CI (1000 iterations) provides uncertainty estimate.
**Result**: R² = 0.978, threshold for ≥95% accuracy ≈ 51,070 tokens (95% CI: [43,922, 56,297])

## Huang et al. (2025) Citation

**Decision**: Use "(2025)" with one-time parenthetical in response letter
**Rationale**: Paper published in PLoS ONE July 2025. Reviewers cite arXiv 2024 version. Parenthetical note avoids confusion without being pedantic.

## Paper Structure for Embedding Results

**Decision**: 1 comparison figure in main text, detailed analyses in appendix
**Rationale**: Keeps the narrative focused on the core contribution (predictive comparison via trained-from-scratch models). Embedding comparison supports the argument but shouldn't dominate.
