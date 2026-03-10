#!/usr/bin/env python
"""Tests for backward-compatible dataset-size support."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "code"))
sys.path.insert(0, str(Path(__file__).parent.parent))

from experiment import Experiment


def test_model_name_round_trip_with_ntokens():
    legacy = Experiment("austen", 0, "gpt2").name
    sized = Experiment("austen", 0, "gpt2", n_train_tokens=128608).name

    assert legacy == "austen_tokenizer=gpt2_seed=0"
    assert sized == "austen_tokenizer=gpt2_ntokens=128608_seed=0"
