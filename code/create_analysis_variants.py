#!/usr/bin/env python
"""
Generate linguistic analysis variants of cleaned text data.

This script creates three variants:
1. content_only: Function words replaced with <FUNC>
2. function_only: Content words replaced with <CONTENT>
3. pos_only: All words replaced with POS tags

Usage:
    python create_analysis_variants.py content
    python create_analysis_variants.py function
    python create_analysis_variants.py pos
    python create_analysis_variants.py all
"""

import re
import argparse
import logging
from pathlib import Path
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import nltk

from constants import CLEANED_DATA_DIR

# Download required NLTK data
try:
    nltk.data.find('taggers/averaged_perceptron_tagger_eng')
except LookupError:
    print("Downloading NLTK averaged_perceptron_tagger...")
    nltk.download('averaged_perceptron_tagger_eng', quiet=True)

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    print("Downloading NLTK punkt tokenizer...")
    nltk.download('punkt_tab', quiet=True)

try:
    nltk.data.find('taggers/universal_tagset')
except LookupError:
    print("Downloading NLTK universal_tagset...")
    nltk.download('universal_tagset', quiet=True)


# Pattern that captures words and preserves all whitespace
TOKEN_PATTERN = r'(\w+|[^\w\s]+|\s+)'


def tokenize_preserving_structure(text):
    """
    Split text into tokens while preserving all whitespace.

    Args:
        text: Input text string

    Returns:
        List of tokens (words, punctuation, whitespace)
    """
    return re.findall(TOKEN_PATTERN, text)


class VariantProcessor:
    """Base class for variant processors."""

    def process_token(self, token):
        """Process a single token. Override in subclasses."""
        raise NotImplementedError

    def process_text(self, text):
        """Process entire text by tokenizing and processing each token."""
        tokens = tokenize_preserving_structure(text)
        processed_tokens = [self.process_token(t) for t in tokens]
        return ''.join(processed_tokens)


class ContentOnlyProcessor(VariantProcessor):
    """Replace function words with <FUNC>, keep content words."""

    def __init__(self):
        self.stop_words = set(ENGLISH_STOP_WORDS)

    def process_token(self, token):
        # Preserve non-word tokens (whitespace, punctuation)
        if not re.match(r'\w', token):
            return token

        # Replace function words with <FUNC>
        if token.lower() in self.stop_words:
            return '<FUNC>'

        # Keep content words unchanged
        return token


class FunctionOnlyProcessor(VariantProcessor):
    """Replace content words with <CONTENT>, keep function words."""

    def __init__(self):
        self.stop_words = set(ENGLISH_STOP_WORDS)

    def process_token(self, token):
        # Preserve non-word tokens
        if not re.match(r'\w', token):
            return token

        # Keep function words unchanged
        if token.lower() in self.stop_words:
            return token

        # Replace content words with <CONTENT>
        return '<CONTENT>'


class POSOnlyProcessor(VariantProcessor):
    """Replace all words with their POS tags."""

    def __init__(self):
        pass

    def process_text(self, text):
        """
        Process entire text to maintain sentence context for POS tagging.

        Override base class method since POS tagging needs sentence context.
        """
        # Tokenize preserving structure
        tokens = tokenize_preserving_structure(text)

        # Extract only word tokens for POS tagging
        word_tokens = [t for t in tokens if re.match(r'\w', t)]

        # Get POS tags using universal tagset
        pos_tags = nltk.pos_tag(word_tokens, tagset='universal')
        pos_dict = {word: tag for word, tag in pos_tags}

        # Replace words with POS tags, preserve other tokens
        result = []
        for token in tokens:
            if re.match(r'\w', token):
                result.append(pos_dict.get(token, 'X'))  # X for unknown
            else:
                result.append(token)

        return ''.join(result)


def process_directory(variant_type, force=False):
    """
    Process all books in data/cleaned/ to create variant.

    Args:
        variant_type: 'content', 'function', or 'pos'
        force: If True, overwrite existing files
    """
    # Set up processor
    processors = {
        'content': ContentOnlyProcessor(),
        'function': FunctionOnlyProcessor(),
        'pos': POSOnlyProcessor()
    }
    processor = processors[variant_type]

    # Set up output directory
    output_base = CLEANED_DATA_DIR / f"{variant_type}_only"

    # Get all input files
    input_files = list(CLEANED_DATA_DIR.glob('**/*.txt'))

    # Filter out variant directories if they exist
    input_files = [f for f in input_files if not any(
        variant in f.parts for variant in ['content_only', 'function_only', 'pos_only']
    )]

    logging.info(f"Processing {len(input_files)} files for {variant_type} variant")

    processed_count = 0
    skipped_count = 0

    for input_file in input_files:
        # Compute relative path
        rel_path = input_file.relative_to(CLEANED_DATA_DIR)
        output_file = output_base / rel_path

        # Skip if exists and not forcing
        if output_file.exists() and not force:
            logging.debug(f"Skipping {rel_path} (already exists)")
            skipped_count += 1
            continue

        # Create output directory
        output_file.parent.mkdir(parents=True, exist_ok=True)

        # Read input
        text = input_file.read_text(encoding='utf-8')

        # Process
        processed_text = processor.process_text(text)

        # Write output
        output_file.write_text(processed_text, encoding='utf-8')

        processed_count += 1
        logging.info(f"Processed: {rel_path}")

    logging.info(f"Complete: {processed_count} processed, {skipped_count} skipped")


def validate_output(variant_type):
    """
    Validate that variant output is correct.

    Checks:
    1. All input files have corresponding output files
    2. Output files have reasonable length (not too short/long)
    3. Variant-specific checks

    Returns:
        List of validation issues (empty if all valid)
    """
    input_dir = CLEANED_DATA_DIR
    output_dir = CLEANED_DATA_DIR / f"{variant_type}_only"

    issues = []

    # Check directory structure mirrors input
    input_files = list(input_dir.glob('**/*.txt'))
    input_files = [f for f in input_files if not any(
        variant in f.parts for variant in ['content_only', 'function_only', 'pos_only']
    )]

    for input_file in input_files:
        rel_path = input_file.relative_to(input_dir)
        output_file = output_dir / rel_path

        if not output_file.exists():
            issues.append(f"Missing output: {output_file}")
            continue

        input_text = input_file.read_text(encoding='utf-8')
        output_text = output_file.read_text(encoding='utf-8')

        # Check length is reasonable (within 50% of original)
        input_len = len(input_text)
        output_len = len(output_text)

        if output_len < input_len * 0.5 or output_len > input_len * 1.5:
            issues.append(f"Length mismatch in {rel_path}: {input_len} -> {output_len}")

        # Variant-specific checks
        if variant_type == 'content':
            if '<FUNC>' not in output_text:
                issues.append(f"No <FUNC> tokens in {rel_path}")
        elif variant_type == 'function':
            if '<CONTENT>' not in output_text:
                issues.append(f"No <CONTENT> tokens in {rel_path}")
        elif variant_type == 'pos':
            # Check that we have POS tags
            if not any(tag in output_text for tag in ['NOUN', 'VERB', 'ADJ']):
                issues.append(f"No POS tags found in {rel_path}")

    return issues


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Generate linguistic analysis variants of cleaned text data'
    )
    parser.add_argument(
        'variant',
        choices=['content', 'function', 'pos', 'all'],
        help='Which variant to generate'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Overwrite existing files'
    )
    parser.add_argument(
        '--validate',
        action='store_true',
        help='Validate output after processing'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Verbose logging'
    )

    args = parser.parse_args()

    # Set up logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # Process variants
    variants = ['content', 'function', 'pos'] if args.variant == 'all' else [args.variant]

    for variant in variants:
        logging.info(f"\n{'='*60}")
        logging.info(f"Processing {variant} variant")
        logging.info(f"{'='*60}")

        process_directory(variant, force=args.force)

        if args.validate:
            logging.info(f"Validating {variant} variant...")
            issues = validate_output(variant)
            if issues:
                logging.error(f"Validation issues for {variant}:")
                for issue in issues:
                    logging.error(f"  - {issue}")
            else:
                logging.info(f"✓ {variant} variant validated successfully")


if __name__ == '__main__':
    main()