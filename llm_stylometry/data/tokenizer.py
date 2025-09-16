"""Tokenizer utilities for text processing."""

from transformers import GPT2TokenizerFast
import logging

logger = logging.getLogger(__name__)


def get_tokenizer(tokenizer_name="gpt2", **kwargs):
    """
    Get a tokenizer by name.

    Args:
        tokenizer_name: Name of the tokenizer (currently only 'gpt2' supported)
        **kwargs: Additional arguments for tokenizer initialization

    Returns:
        Initialized tokenizer

    Raises:
        ValueError: If tokenizer name is not supported
    """
    logger.info(f"Creating tokenizer: {tokenizer_name}")

    if tokenizer_name == "gpt2":
        tokenizer = GPT2TokenizerFast.from_pretrained("gpt2", **kwargs)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer
    else:
        raise ValueError(f"Unsupported tokenizer: {tokenizer_name}")