# Huang et al. (2025) Research Notes

## Citation
Huang, W., Murakami, A., & Grieve, J. (2025). Attributing authorship via the perplexity of authorial language models. PLoS ONE, 20(7), e0327081.

## Methodology — Authorial Language Models (ALMs)
- Fine-tune GPT-2 base (pre-trained) per author via continued pretraining
- Compute perplexity of test documents across all ALMs
- Assign to author with lowest perplexity
- Token-level annotations via comparative negative log-likelihood (CNLL)

## Key Difference from Our Approach
- They FINE-TUNE pre-trained GPT-2; we TRAIN FROM SCRATCH
- Fine-tuning inherits pre-trained representations → can't guarantee held-out texts weren't in GPT-2's training data (WebText, 40GB web scrape)
- Their defense: "WebText does not appear to contain any of our training data" — weak, can't prove absence
- Training from scratch = guaranteed data purity

## Benchmark Details

| Dataset | Authors | Avg Text Length | ALM Accuracy |
|-|-|-|-|
| Blogs50 | 50 | 112 tokens | 83.6% |
| CCAT50 | 50 | 506 tokens | 74.9% |
| Guardian | 13 | 1,052 tokens | 94.5% |
| IMDB62 | 62 | 349 tokens | 99.5% |

## Why Benchmarks Are Infeasible for Our Approach
- Our sigmoid analysis: need ~51K tokens per author for >=95% accuracy
- Blogs50: 112 tokens/text — orders of magnitude too few
- CCAT50: 506 tokens/text — still way too few
- Guardian: 1,052 tokens/text — still insufficient
- Even IMDB62 (349 tokens/text) is far below our threshold
- Training from scratch requires much more data than fine-tuning — this is expected

## Arguments for Response Letter
1. Training from scratch guarantees data purity; fine-tuning cannot
2. ~51K tokens needed → benchmarks with 100–1000 tokens/text are a different regime entirely
3. GPT-2 achieves 100% on our dataset → larger models unnecessary (addresses R2)
4. Our contribution is showing that from-scratch training captures author-specific patterns; theirs shows fine-tuning does — complementary, not redundant
