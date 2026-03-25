#!/usr/bin/env python
"""
Embedding-based authorship attribution comparison.

Compares text embedding nearest-neighbor classification against
our cross-entropy (predictive comparison) approach.

Protocol:
  1. Chunk each book into 1024-token windows with 128-token overlap
  2. Embed each chunk using a pre-trained model (sentence-transformers)
  3. Leave-one-out: for each held-out book, classify each chunk by
     nearest training chunk (cosine similarity)
  4. Book-level prediction = modal author across chunks
  5. Report: accuracy, purity, confusion, runner-up

Usage:
    python code/embedding_comparison.py                      # Run all 3 models
    python code/embedding_comparison.py --model nomic-ai/nomic-embed-text-v1.5  # Single model
    python code/embedding_comparison.py --figures-only        # Generate figures from cached results
"""

import argparse
import csv
import json
import logging
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from constants import AUTHORS, CLEANED_DATA_DIR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Models to evaluate (MTEB leaderboard selections)
MODELS = [
    "nomic-ai/nomic-embed-text-v1.5",  # 137M params, MTEB 44.1
    "BAAI/bge-m3",  # 568M params, MTEB 59.6
    "Qwen/Qwen3-Embedding-4B",  # 4.0B params, MTEB 69.5
]

CHUNK_SIZE = 1024  # tokens per chunk
CHUNK_OVERLAP = 128  # token overlap between chunks
MIN_CHUNKS = 3  # minimum chunks per book (warn if fewer)
RESULTS_DIR = Path("data/embedding_results")


def get_tokenizer():
    """Get GPT-2 tokenizer for consistent chunking."""
    from transformers import GPT2Tokenizer

    return GPT2Tokenizer.from_pretrained("gpt2")


def chunk_book(text, tokenizer, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """
    Split a book into overlapping token chunks.

    Args:
        text: Full book text
        tokenizer: Tokenizer for splitting into tokens
        chunk_size: Tokens per chunk
        overlap: Token overlap between consecutive chunks

    Returns:
        List of text chunks
    """
    original_max = tokenizer.model_max_length
    tokenizer.model_max_length = int(1e8)
    token_ids = tokenizer.encode(text)
    tokenizer.model_max_length = original_max

    stride = chunk_size - overlap
    chunks = []
    for start in range(0, len(token_ids), stride):
        end = start + chunk_size
        chunk_ids = token_ids[start:end]
        if len(chunk_ids) < overlap:
            break  # Skip tiny trailing fragments
        chunks.append(tokenizer.decode(chunk_ids))

    # Ensure at least 1 chunk for very short texts
    if not chunks and token_ids:
        chunks.append(tokenizer.decode(token_ids))

    return chunks


def load_books():
    """
    Load all books from the cleaned data directory.

    Returns:
        List of dicts: [{author, book_id, text, path}, ...]
    """
    books = []
    for author in AUTHORS:
        author_dir = CLEANED_DATA_DIR / author
        if not author_dir.exists():
            logger.warning(f"Author directory not found: {author_dir}")
            continue
        for txt_file in sorted(author_dir.glob("*.txt")):
            text = txt_file.read_text(encoding="utf-8")
            books.append(
                {
                    "author": author,
                    "book_id": txt_file.stem,
                    "text": text,
                    "path": str(txt_file),
                }
            )
    return books


def embed_chunks(chunks, model, batch_size=32):
    """
    Embed a list of text chunks using a sentence-transformers model.

    Args:
        chunks: List of text strings
        model: SentenceTransformer model instance
        batch_size: Batch size for encoding

    Returns:
        numpy array of shape (n_chunks, embedding_dim)
    """
    embeddings = model.encode(
        chunks,
        batch_size=batch_size,
        show_progress_bar=False,
        normalize_embeddings=True,  # L2-normalize for cosine similarity via dot product
    )
    return np.array(embeddings)


def classify_book_chunks(held_out_embeddings, train_embeddings, train_authors):
    """
    Classify each held-out chunk by nearest training chunk.

    Args:
        held_out_embeddings: (n_held_out, dim) array
        train_embeddings: (n_train, dim) array
        train_authors: list of author labels for each training chunk

    Returns:
        List of predicted authors (one per held-out chunk),
        List of similarity scores (one per held-out chunk)
    """
    # Cosine similarity via dot product (embeddings are L2-normalized)
    similarities = held_out_embeddings @ train_embeddings.T  # (n_held_out, n_train)

    nearest_idx = np.argmax(similarities, axis=1)
    predicted_authors = [train_authors[i] for i in nearest_idx]
    similarity_scores = [
        similarities[i, nearest_idx[i]] for i in range(len(nearest_idx))
    ]

    return predicted_authors, similarity_scores


def compute_book_result(true_author, chunk_predictions, chunk_similarities):
    """
    Compute book-level classification result from chunk predictions.

    Args:
        true_author: Ground truth author
        chunk_predictions: List of predicted authors per chunk
        chunk_similarities: List of similarity scores per chunk

    Returns:
        Dict with book-level results
    """
    counts = Counter(chunk_predictions)
    modal_author = counts.most_common(1)[0][0]

    # Tie-breaking: if multiple authors have same count, pick by highest avg similarity
    max_count = counts.most_common(1)[0][1]
    tied_authors = [a for a, c in counts.items() if c == max_count]
    if len(tied_authors) > 1:
        avg_sims = {}
        for author in tied_authors:
            author_sims = [
                s for p, s in zip(chunk_predictions, chunk_similarities) if p == author
            ]
            avg_sims[author] = np.mean(author_sims)
        modal_author = max(avg_sims, key=avg_sims.get)

    purity = counts[modal_author] / len(chunk_predictions)
    chunk_accuracy = sum(1 for p in chunk_predictions if p == true_author) / len(
        chunk_predictions
    )

    # Runner-up
    if len(counts) > 1:
        runner_up = counts.most_common(2)[1][0]
        runner_up_count = counts.most_common(2)[1][1]
        margin = (counts[modal_author] - runner_up_count) / len(chunk_predictions)
    else:
        runner_up = None
        margin = 1.0

    return {
        "true_author": true_author,
        "modal_author": modal_author,
        "correct": modal_author == true_author,
        "purity": purity,
        "chunk_accuracy": chunk_accuracy,
        "runner_up": runner_up,
        "margin": margin,
        "n_chunks": len(chunk_predictions),
        "vote_counts": dict(counts),
    }


def run_embedding_comparison(model_name, books=None, tokenizer=None):
    """
    Run full leave-one-out embedding comparison for a single model.

    Args:
        model_name: HuggingFace model name
        books: Pre-loaded books list (optional, loaded if None)
        tokenizer: GPT-2 tokenizer (optional, loaded if None)

    Returns:
        List of book-level result dicts
    """
    from sentence_transformers import SentenceTransformer

    if books is None:
        books = load_books()
    if tokenizer is None:
        tokenizer = get_tokenizer()

    logger.info(f"Loading model: {model_name}")
    try:
        model = SentenceTransformer(model_name, trust_remote_code=True)
    except Exception as e:
        logger.error(f"Failed to load model {model_name}: {e}")
        logger.warning(f"Skipping {model_name}")
        return []

    # Chunk all books
    logger.info(f"Chunking {len(books)} books...")
    book_chunks = []
    for book in books:
        chunks = chunk_book(book["text"], tokenizer)
        if len(chunks) < MIN_CHUNKS:
            logger.warning(
                f"Book {book['author']}/{book['book_id']} has only {len(chunks)} chunks "
                f"(< {MIN_CHUNKS} minimum)"
            )
        book_chunks.append(chunks)

    # Embed all chunks (with per-book checkpoint/resume)
    model_cache_dir = RESULTS_DIR / model_name.replace("/", "_") / "embeddings"
    model_cache_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Embedding all chunks (cached per book)...")
    book_embeddings = []
    for i, chunks in enumerate(book_chunks):
        cache_file = model_cache_dir / f"{books[i]['author']}_{books[i]['book_id']}.npy"
        if cache_file.exists():
            emb = np.load(cache_file)
            book_embeddings.append(emb)
            if i % 10 == 0:
                logger.info(
                    f"  Book {i+1}/{len(books)} ({books[i]['author']}/{books[i]['book_id']}) — loaded from cache"
                )
            continue

        if i % 10 == 0:
            logger.info(
                f"  Embedding book {i+1}/{len(books)} ({books[i]['author']}/{books[i]['book_id']}, {len(chunks)} chunks)"
            )
        try:
            emb = embed_chunks(chunks, model)
            np.save(cache_file, emb)
            book_embeddings.append(emb)
        except Exception as e:
            logger.error(
                f"OOM or error embedding {books[i]['author']}/{books[i]['book_id']}: {e}"
            )
            book_embeddings.append(None)

    # Leave-one-out classification
    logger.info("Running leave-one-out classification...")
    results = []
    for held_out_idx in range(len(books)):
        if book_embeddings[held_out_idx] is None:
            continue

        held_out_emb = book_embeddings[held_out_idx]
        held_out_book = books[held_out_idx]

        # Build training set (all chunks except held-out book)
        train_embs = []
        train_authors = []
        for j in range(len(books)):
            if j == held_out_idx or book_embeddings[j] is None:
                continue
            train_embs.append(book_embeddings[j])
            train_authors.extend([books[j]["author"]] * len(book_embeddings[j]))

        train_embeddings = np.vstack(train_embs)

        # Classify each held-out chunk
        chunk_preds, chunk_sims = classify_book_chunks(
            held_out_emb, train_embeddings, train_authors
        )

        # Compute book-level result
        result = compute_book_result(held_out_book["author"], chunk_preds, chunk_sims)
        result["book_id"] = held_out_book["book_id"]
        result["model"] = model_name
        results.append(result)

        if (held_out_idx + 1) % 10 == 0:
            correct_so_far = sum(1 for r in results if r["correct"])
            logger.info(
                f"  {held_out_idx+1}/{len(books)} books classified "
                f"({correct_so_far}/{len(results)} correct so far)"
            )

    return results


def save_results(results, model_name):
    """Save results to CSV in data/embedding_results/{model_name}/."""
    model_dir = RESULTS_DIR / model_name.replace("/", "_")
    model_dir.mkdir(parents=True, exist_ok=True)

    # Save book-level results
    csv_path = model_dir / "book_results.csv"
    fieldnames = [
        "model",
        "book_id",
        "true_author",
        "modal_author",
        "correct",
        "purity",
        "chunk_accuracy",
        "runner_up",
        "margin",
        "n_chunks",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)

    # Save summary
    summary = compute_summary(results, model_name)
    with open(model_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Results saved to {model_dir}")
    return csv_path


def load_cached_results(model_name):
    """Load cached results from CSV if available."""
    model_dir = RESULTS_DIR / model_name.replace("/", "_")
    csv_path = model_dir / "book_results.csv"
    if not csv_path.exists():
        return None

    import pandas as pd

    df = pd.read_csv(csv_path)
    results = df.to_dict("records")
    # Convert string booleans
    for r in results:
        if isinstance(r["correct"], str):
            r["correct"] = r["correct"].lower() == "true"
    return results


def compute_summary(results, model_name):
    """Compute summary statistics from results."""
    n_correct = sum(1 for r in results if r["correct"])
    n_total = len(results)
    accuracy = n_correct / n_total * 100 if n_total > 0 else 0

    # Per-author accuracy
    author_results = {}
    for author in AUTHORS:
        author_books = [r for r in results if r["true_author"] == author]
        if author_books:
            author_correct = sum(1 for r in author_books if r["correct"])
            author_results[author] = {
                "accuracy": author_correct / len(author_books) * 100,
                "correct": author_correct,
                "total": len(author_books),
                "avg_purity": np.mean([r["purity"] for r in author_books]),
            }

    # Confusion matrix
    confusion = {}
    for r in results:
        key = f"{r['true_author']}→{r['modal_author']}"
        confusion[key] = confusion.get(key, 0) + 1

    return {
        "model": model_name,
        "overall_accuracy": accuracy,
        "correct": n_correct,
        "total": n_total,
        "avg_purity": np.mean([r["purity"] for r in results]),
        "per_author": author_results,
        "confusion": confusion,
    }


def print_summary(summary):
    """Print a formatted summary."""
    print(f"\n{'='*60}")
    print(f"Model: {summary['model']}")
    print(f"{'='*60}")
    print(
        f"Overall accuracy: {summary['correct']}/{summary['total']} ({summary['overall_accuracy']:.1f}%)"
    )
    print(f"Average purity:   {summary['avg_purity']:.3f}")
    print("\nPer-author results:")
    print(
        f"{'Author':<12} {'Accuracy':>10} {'Correct':>8} {'Total':>6} {'Avg Purity':>12}"
    )
    print("-" * 50)
    for author in AUTHORS:
        if author in summary["per_author"]:
            a = summary["per_author"][author]
            print(
                f"{author.capitalize():<12} {a['accuracy']:>9.1f}% {a['correct']:>8}/{a['total']:<6} {a['avg_purity']:>11.3f}"
            )

    # Show misclassifications
    misclassifications = {
        k: v
        for k, v in summary["confusion"].items()
        if k.split("→")[0] != k.split("→")[1]
    }
    if misclassifications:
        print("\nMisclassifications:")
        for k, v in sorted(misclassifications.items(), key=lambda x: -x[1]):
            print(f"  {k}: {v}")


def main():
    parser = argparse.ArgumentParser(
        description="Embedding-based authorship attribution comparison"
    )
    parser.add_argument(
        "--model", type=str, default=None, help="Run a single model (HuggingFace name)"
    )
    parser.add_argument(
        "--figures-only",
        action="store_true",
        help="Generate figures from cached results only",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="paper/figs/source",
        help="Output directory for figures",
    )
    args = parser.parse_args()

    models_to_run = [args.model] if args.model else MODELS

    if args.figures_only:
        # Load cached results and generate figures
        all_results = {}
        for model_name in models_to_run:
            results = load_cached_results(model_name)
            if results is None:
                logger.error(
                    f"No cached results for {model_name}. Run without --figures-only first."
                )
                continue
            summary = compute_summary(results, model_name)
            all_results[model_name] = {"results": results, "summary": summary}
            print_summary(summary)

        if all_results:
            generate_figures(all_results, args.output)
        return

    # Load books and tokenizer once
    books = load_books()
    tokenizer = get_tokenizer()
    logger.info(f"Loaded {len(books)} books across {len(AUTHORS)} authors")

    all_results = {}
    for model_name in models_to_run:
        logger.info(f"\n{'='*60}")
        logger.info(f"Running: {model_name}")
        logger.info(f"{'='*60}")

        # Check cache first
        cached = load_cached_results(model_name)
        if cached is not None:
            logger.info(f"Using cached results for {model_name}")
            results = cached
        else:
            results = run_embedding_comparison(model_name, books, tokenizer)
            if results:
                save_results(results, model_name)

        if results:
            summary = compute_summary(results, model_name)
            all_results[model_name] = {"results": results, "summary": summary}
            print_summary(summary)

    if all_results:
        generate_figures(all_results, args.output)


def generate_figures(all_results, output_dir):
    """Generate all comparison figures."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(Path(__file__).parent.parent))
    from llm_stylometry.visualization.embedding_comparison import (
        generate_embedding_comparison_figure,
        generate_embedding_confusion_figure,
        generate_embedding_purity_figure,
    )

    summaries = [v["summary"] for v in all_results.values()]
    all_book_results = {k: v["results"] for k, v in all_results.items()}

    generate_embedding_comparison_figure(
        summaries=summaries,
        output_path=output_dir / "embedding_comparison.pdf",
    )
    logger.info(f"Saved: {output_dir / 'embedding_comparison.pdf'}")

    generate_embedding_purity_figure(
        all_book_results=all_book_results,
        output_path=output_dir / "embedding_purity.pdf",
    )
    logger.info(f"Saved: {output_dir / 'embedding_purity.pdf'}")

    generate_embedding_confusion_figure(
        all_book_results=all_book_results,
        output_path=output_dir / "embedding_confusion.pdf",
    )
    logger.info(f"Saved: {output_dir / 'embedding_confusion.pdf'}")


if __name__ == "__main__":
    main()
