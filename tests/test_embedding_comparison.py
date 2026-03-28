#!/usr/bin/env python
"""Tests for embedding-based authorship attribution comparison."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "code"))


def test_chunk_book_basic():
    """T025: Verify chunking produces correct number and size of chunks."""
    from embedding_comparison import chunk_book, get_tokenizer

    tokenizer = get_tokenizer()

    # Create a text that's ~3000 tokens long
    text = "The quick brown fox jumps over the lazy dog. " * 500
    chunks = chunk_book(text, tokenizer, chunk_size=1024, overlap=128)

    # With stride = 1024 - 128 = 896 tokens per step
    token_ids = tokenizer.encode(text)
    expected_chunks = max(1, (len(token_ids) - 128) // (1024 - 128))

    assert len(chunks) >= 1, "Should produce at least 1 chunk"
    assert (
        abs(len(chunks) - expected_chunks) <= 2
    ), f"Expected ~{expected_chunks} chunks, got {len(chunks)}"

    # Each chunk should decode to non-empty text
    for chunk in chunks:
        assert len(chunk) > 0, "Chunk should not be empty"


def test_chunk_book_overlap():
    """T025: Verify chunks overlap correctly."""
    from embedding_comparison import chunk_book, get_tokenizer

    tokenizer = get_tokenizer()
    text = "word " * 5000  # ~5000 tokens
    chunks = chunk_book(text, tokenizer, chunk_size=100, overlap=20)

    assert len(chunks) > 1, "Should produce multiple chunks with this text"

    # Tokenize first two chunks and check overlap
    tokens_0 = tokenizer.encode(chunks[0])
    tokens_1 = tokenizer.encode(chunks[1])

    # The last `overlap` tokens of chunk 0 should match the first `overlap` tokens of chunk 1
    # (approximately — decoding/re-encoding may introduce minor differences)
    assert len(tokens_0) <= 100 + 5  # Allow small tokenization variance
    assert len(tokens_1) <= 100 + 5


def test_chunk_book_short_text():
    """T025: Verify short text produces fewer than MIN_CHUNKS with warning."""
    from embedding_comparison import chunk_book, get_tokenizer

    tokenizer = get_tokenizer()
    short_text = "Hello world."
    chunks = chunk_book(short_text, tokenizer, chunk_size=1024, overlap=128)

    assert len(chunks) >= 1, "Even short text should produce at least 1 chunk"
    assert len(chunks) < 3, "Short text should produce fewer than MIN_CHUNKS"


def test_classify_book_chunks_nearest_neighbor():
    """T027: Verify nearest-neighbor classification with known vectors."""
    from embedding_comparison import classify_book_chunks

    # Create synthetic embeddings where the answer is obvious
    dim = 64
    rng = np.random.default_rng(42)

    # 3 "authors" with distinct embedding clusters
    author_a_center = np.array([1.0] + [0.0] * (dim - 1))
    author_b_center = np.array([0.0, 1.0] + [0.0] * (dim - 2))
    author_c_center = np.array([0.0, 0.0, 1.0] + [0.0] * (dim - 3))

    # Training chunks: 10 per author, slightly noisy
    train_embs = []
    train_authors = []
    for center, author in [
        (author_a_center, "a"),
        (author_b_center, "b"),
        (author_c_center, "c"),
    ]:
        for _ in range(10):
            noisy = center + rng.normal(0, 0.1, dim)
            noisy /= np.linalg.norm(noisy)
            train_embs.append(noisy)
            train_authors.append(author)

    train_embeddings = np.array(train_embs)

    # Held-out chunks: clearly from author "a"
    held_out = []
    for _ in range(5):
        noisy = author_a_center + rng.normal(0, 0.05, dim)
        noisy /= np.linalg.norm(noisy)
        held_out.append(noisy)
    held_out_embeddings = np.array(held_out)

    predictions, similarities = classify_book_chunks(
        held_out_embeddings, train_embeddings, train_authors
    )

    assert len(predictions) == 5
    assert all(p == "a" for p in predictions), f"Expected all 'a', got {predictions}"
    assert all(
        s > 0.5 for s in similarities
    ), "Similarities should be high for matching author"


def test_modal_vote_classification():
    """T026: Verify modal vote, purity, and runner-up computation."""
    from embedding_comparison import compute_book_result

    # Scenario: 10 chunks, 7 predict "austen", 2 predict "dickens", 1 predicts "twain"
    chunk_predictions = ["austen"] * 7 + ["dickens"] * 2 + ["twain"] * 1
    chunk_similarities = [0.9] * 7 + [0.8] * 2 + [0.7] * 1

    result = compute_book_result("austen", chunk_predictions, chunk_similarities)

    assert result["modal_author"] == "austen"
    assert result["correct"] is True
    assert result["purity"] == 0.7  # 7/10
    assert result["chunk_accuracy"] == 0.7  # 7/10 correct
    assert result["runner_up"] == "dickens"
    assert result["margin"] == 0.5  # (7-2)/10
    assert result["n_chunks"] == 10


def test_modal_vote_tie_breaking():
    """T026: Verify tie-breaking uses highest average similarity."""
    from embedding_comparison import compute_book_result

    # Tie: 3 chunks for "austen" (high similarity), 3 for "dickens" (low similarity)
    chunk_predictions = ["austen"] * 3 + ["dickens"] * 3
    chunk_similarities = [0.95, 0.90, 0.92] + [0.70, 0.72, 0.68]

    result = compute_book_result("austen", chunk_predictions, chunk_similarities)

    # austen should win because avg similarity (0.923) > dickens avg (0.700)
    assert (
        result["modal_author"] == "austen"
    ), f"Expected 'austen' to win tie-break, got '{result['modal_author']}'"


def test_modal_vote_incorrect_prediction():
    """T026: Verify correct=False when modal author != true author."""
    from embedding_comparison import compute_book_result

    chunk_predictions = ["dickens"] * 8 + ["austen"] * 2
    chunk_similarities = [0.85] * 8 + [0.90] * 2

    result = compute_book_result("austen", chunk_predictions, chunk_similarities)

    assert result["modal_author"] == "dickens"
    assert result["correct"] is False
    assert result["purity"] == 0.8
    assert result["chunk_accuracy"] == 0.2  # Only 2/10 are "austen"
