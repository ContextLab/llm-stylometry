#!/usr/bin/env python
"""
Test data loading for all variants.
Verifies that data loaders work correctly with variant data.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'code'))

from constants import AUTHORS, ANALYSIS_VARIANTS, get_data_dir
from data_utils import sample_book_path, tokenize_texts, sample_tokens
from tokenizer_utils import get_tokenizer

def test_variant_data_loading():
    """Test data loading for all variants."""
    print("\n" + "="*60)
    print("Test: Data Loading for All Variants")
    print("="*60)

    tokenizer = get_tokenizer("gpt2")
    test_seed = 42
    test_author = "fitzgerald"

    # Test each variant
    for variant in [None] + ANALYSIS_VARIANTS:
        print(f"\n--- Testing variant: {variant or 'baseline'} ---")

        # Get data directory
        data_dir = get_data_dir(variant)
        print(f"Data directory: {data_dir}")
        assert data_dir.exists()

        # Sample a book path
        book_path = sample_book_path(test_author, test_seed, variant)
        print(f"Sampled book: {book_path.name}")
        assert book_path.exists()
        assert book_path.parent.name == test_author

        # Verify it's in the right directory
        if variant:
            assert f"{variant}_only" in str(book_path.parent.parent)
        else:
            assert "cleaned" in str(book_path.parent.parent)
            assert "only" not in str(book_path.parent.parent)

        # Test tokenization
        author_dir = data_dir / test_author
        tokenized_texts = tokenize_texts(tokenizer, author_dir)
        print(f"Tokenized {len(tokenized_texts)} texts")
        assert len(tokenized_texts) > 0

        # Test sampling tokens
        n_tokens = 10000  # Small sample for testing
        sampled = sample_tokens(tokenized_texts, n_tokens, test_seed)
        print(f"Sampled {len(sampled)} tokens")
        assert len(sampled) == n_tokens

        # Verify content is different for variants
        sample_text = tokenizer.decode(sampled[:100])
        print(f"Sample text: {sample_text[:100]}...")

        if variant == 'content':
            assert '<FUNC>' in sample_text, "Expected <FUNC> tokens in content variant"
            print("✓ Found <FUNC> tokens in content variant")
        elif variant == 'function':
            assert '<CONTENT>' in sample_text, "Expected <CONTENT> tokens in function variant"
            print("✓ Found <CONTENT> tokens in function variant")
        elif variant == 'pos':
            # Check for POS tags (NOUN, VERB, etc.)
            has_pos_tags = any(tag in sample_text for tag in ['NOUN', 'VERB', 'ADJ', 'ADP'])
            assert has_pos_tags, "Expected POS tags in pos variant"
            print("✓ Found POS tags in pos variant")

        print(f"✓ Variant {variant or 'baseline'} data loading successful")

    print("\n" + "="*60)
    print("✓ ALL VARIANT DATA LOADING TESTS PASSED")
    print("="*60)

if __name__ == "__main__":
    test_variant_data_loading()