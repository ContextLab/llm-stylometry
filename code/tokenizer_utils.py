from transformers import GPT2TokenizerFast
import logging

logger = logging.getLogger(__name__)


def get_tokenizer(tokenizer_name, **kwargs):
    logger.info(f"Creating tokenizer: {tokenizer_name}")

    if tokenizer_name == "gpt2":
        tokenizer = GPT2TokenizerFast.from_pretrained("gpt2", **kwargs)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer
    else:
        raise ValueError(f"Unsupported tokenizer: {tokenizer_name}")
