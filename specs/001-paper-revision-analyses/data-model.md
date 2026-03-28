# Data Model: Paper Revision Analyses

## Entities

### Accuracy-vs-Tokens Record
- **Source**: PR #51 dataset-size sweep results (`data/model_results_ntokens.pkl.gz`)
- **Fields**: n_train_tokens (int), train_author (str), seed (int), predicted_author (str), correct (bool)
- **Granularity**: One record per model (8 authors × 10 seeds × 16 token levels = 1,280 models)
- **Aggregation**: Grouped by n_train_tokens → accuracy = sum(correct) / count

### Sigmoid Fit Parameters
- **Source**: `code/fit_sigmoid.py` output
- **Fields**: L (float, lower asymptote), K (float, range), b (float, steepness), m (float, midpoint on log₁₀ scale)
- **Derived**: R², threshold_tokens_95 (float), bootstrap_ci_lo (float), bootstrap_ci_hi (float)
- **Persistence**: Printed to stdout; figure saved as PDF. Parameters hardcoded in paper text.

### Book Chunk
- **Source**: `data/cleaned/{author}/*.txt`, chunked by embedding pipeline
- **Fields**: author (str), book_id (str, filename stem), chunk_index (int), text (str, 1024 tokens), token_count (int)
- **Chunk count formula**: ceil((book_tokens - 1024) / (1024 - 128)) + 1. A 30K-token book → ~33 chunks; a 200K-token book → ~223 chunks. Books with <3 chunks (<~2,800 tokens) are flagged.
- **Relationships**: Many chunks per book; each chunk inherits author from parent book

### Chunk Embedding
- **Source**: Embedding model forward pass
- **Fields**: model_name (str), author (str), book_id (str), chunk_index (int), embedding (float vector)
- **Dimensionality**: 768 (nomic), 1024 (bge-m3), 2560 (Qwen3-4B)

### Embedding Classification Result
- **Source**: Nearest-neighbor classification
- **Fields per chunk**: held_out_book (str), chunk_index (int), true_author (str), predicted_author (str), similarity_score (float), nearest_book (str)
- **Fields per book**: held_out_book (str), true_author (str), modal_author (str), correct (bool), purity (float), chunk_accuracy (float), runner_up_author (str), margin (float)
- **Persistence**: CSV per model in `data/embedding_results/{model_name}/`

## Relationships

```
Author (8) ──1:N──> Book (6-14 per author, 84 total)
Book ──1:N──> Chunk (~30-1000 per book)
Chunk ──1:1──> ChunkEmbedding (per model)
Book ──1:1──> EmbeddingClassificationResult (per model, via LOO)
n_train_tokens level ──1:N──> AccuracyRecord (80 per level)
AccuracyRecord set ──fit──> SigmoidParameters
```
