#!/usr/bin/env python
"""
Fast integration test for variant training.
Tests training a minimal model (2 layers, 3 epochs) on all variants including baseline.
Should complete in 8-15 minutes total (4 variants × 2-4 min each).
"""

import os
import sys
import shutil
from pathlib import Path
import pandas as pd

# Add code to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'code'))

from experiment import Experiment
from constants import AUTHORS, MODELS_DIR, get_data_dir, ANALYSIS_VARIANTS

def train_variant_model(variant, test_author="fitzgerald", test_seed=42):
    """Train a single test model on specified variant (or None for baseline)."""
    variant_name = variant or "baseline"
    print("\n" + "="*60)
    print(f"Test: Training on {variant_name} variant")
    print("="*60)

    # Verify variant data exists
    variant_dir = get_data_dir(variant)
    assert variant_dir.exists(), f"Variant directory not found: {variant_dir}"
    print(f"✓ Variant directory exists: {variant_dir}")

    # Create test experiment with minimal model for fast testing
    exp = Experiment(
        train_author=test_author,
        seed=test_seed,
        tokenizer_name="gpt2",
        analysis_variant=variant,
        n_train_tokens=10000,  # Much smaller dataset for testing
        n_positions=128,       # Smaller context
        n_embd=64,            # Tiny model
        n_layer=2,            # Just 2 layers
        n_head=2,             # 2 attention heads
        batch_size=4,         # Smaller batch
        stop_criteria={
            "train_loss": 2.0,  # Realistic threshold that won't trigger early
            "min_epochs": 3,    # Run all 3 epochs
            "max_epochs": 3,    # Only 3 epochs for testing
        }
    )

    # Verify naming convention
    if variant:
        expected_name = f"{test_author}_variant={variant}_tokenizer=gpt2_seed={test_seed}"
    else:
        expected_name = f"{test_author}_tokenizer=gpt2_seed={test_seed}"
    assert exp.name == expected_name, f"Name mismatch: {exp.name} != {expected_name}"
    print(f"✓ Model name correct: {exp.name}")

    # Verify data directory
    assert exp.data_dir == variant_dir
    print(f"✓ Data directory correct: {exp.data_dir}")

    # Clean up any existing test model
    model_dir = MODELS_DIR / exp.name
    if model_dir.exists():
        print(f"Removing existing test model: {model_dir}")
        shutil.rmtree(model_dir)

    # Set environment variable to prevent main.py from running at import time
    os.environ['NO_MULTIPROCESSING'] = '1'

    # Import run_experiment function only (avoid module-level execution)
    print("\nStarting training (3 epochs, tiny model)...")
    print("Expected time: 2-5 minutes depending on hardware...")

    # Import required modules
    import torch
    import random
    import numpy as np
    from transformers import GPT2Config, GPT2LMHeadModel
    from data_utils import get_train_data_loader, get_eval_data_loader
    from model_utils import init_model, save_checkpoint, count_non_embedding_params
    from tokenizer_utils import get_tokenizer
    from eval_utils import evaluate_model
    from logging_utils import update_loss_log
    import logging

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Determine device
    device_type = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    device = torch.device(device_type)
    device_label = device_type.upper()

    print(f"Using device: {device_label}")

    # Initialize tokenizer
    tokenizer = get_tokenizer(exp.tokenizer_name)

    # Set random seeds
    random.seed(exp.seed)
    np.random.seed(exp.seed)
    torch.manual_seed(exp.seed)

    # Set up train dataloader
    train_dataloader = get_train_data_loader(
        path=exp.data_dir / exp.train_author,
        tokenizer=tokenizer,
        n_positions=exp.n_positions,
        batch_size=exp.batch_size,
        n_tokens=exp.n_train_tokens,
        seed=exp.seed,
        excluded_train_path=exp.excluded_train_path,
    )
    logger.info(f"Number of training batches: {len(train_dataloader)}")

    # Set up eval dataloaders
    eval_dataloaders = {
        name: get_eval_data_loader(
            path=path,
            tokenizer=tokenizer,
            n_positions=exp.n_positions,
            batch_size=exp.batch_size,
        )
        for name, path in exp.eval_paths.items()
    }

    # Set up model
    config = GPT2Config(
        n_positions=exp.n_positions,
        n_embd=exp.n_embd,
        n_layer=exp.n_layer,
        n_head=exp.n_head,
    )

    model, optimizer = init_model(
        model_class=GPT2LMHeadModel,
        model_name=exp.name,
        config=config,
        device=device,
        lr=exp.lr,
    )

    logger.info(f"Non-embedding parameters: {count_non_embedding_params(model)}")

    # Initial evaluation
    for name, eval_dataloader in eval_dataloaders.items():
        eval_loss = evaluate_model(model=model, eval_dataloader=eval_dataloader, device=device)
        update_loss_log(
            log_file_path=MODELS_DIR / exp.name / "loss_logs.csv",
            epochs_completed=0,
            loss_dataset=name,
            loss_value=eval_loss,
            seed=exp.seed,
            train_author=exp.train_author,
        )

    # Training loop
    for epoch in range(exp.stop_criteria["max_epochs"]):
        total_train_loss = 0.0
        model.train()

        for batch in train_dataloader:
            input_ids = batch["input_ids"].to(device)
            outputs = model(input_ids=input_ids, labels=input_ids)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            del outputs, loss

        epochs_completed = epoch + 1
        train_loss = total_train_loss / len(train_dataloader)

        # Log training loss
        update_loss_log(
            log_file_path=MODELS_DIR / exp.name / "loss_logs.csv",
            train_author=exp.train_author,
            loss_dataset="train",
            loss_value=train_loss,
            epochs_completed=epochs_completed,
            seed=exp.seed,
        )

        # Evaluate
        eval_losses = {}
        for name, eval_dataloader in eval_dataloaders.items():
            eval_loss = evaluate_model(model=model, eval_dataloader=eval_dataloader, device=device)
            eval_losses[name] = eval_loss
            update_loss_log(
                log_file_path=MODELS_DIR / exp.name / "loss_logs.csv",
                epochs_completed=epochs_completed,
                loss_dataset=name,
                loss_value=eval_loss,
                seed=exp.seed,
                train_author=exp.train_author,
            )

        log_message = f"Epoch {epochs_completed}/{exp.stop_criteria['max_epochs']}: train={train_loss:.4f}"
        for name, loss in eval_losses.items():
            log_message += f", {name}={loss:.4f}"
        logger.info(log_message)

        # Save checkpoint
        save_checkpoint(model=model, optimizer=optimizer, model_name=exp.name, epochs_completed=epochs_completed)

        # Early stopping
        if train_loss <= exp.stop_criteria["train_loss"] and epochs_completed >= exp.stop_criteria["min_epochs"]:
            logger.info(f"Stopping: train loss {train_loss:.4f} <= {exp.stop_criteria['train_loss']}")
            break

    logger.info(f"Training complete for {exp.name}")

    # Verify model was created
    assert model_dir.exists(), f"Model directory not created: {model_dir}"
    print(f"✓ Model directory created: {model_dir}")

    # Verify model files exist
    config_file = model_dir / "config.json"
    weights_file = model_dir / "model.safetensors"
    training_state = model_dir / "training_state.pt"
    loss_log = model_dir / "loss_logs.csv"

    assert config_file.exists(), "config.json not found"
    assert weights_file.exists() or (model_dir / "pytorch_model.bin").exists(), "Model weights not found"
    assert training_state.exists(), "training_state.pt not found"
    assert loss_log.exists(), "loss_logs.csv not found"
    print("✓ All model files created")

    # Verify loss logs
    df = pd.read_csv(loss_log)
    assert not df.empty, "Loss log is empty"
    assert 'train_author' in df.columns
    assert 'loss_dataset' in df.columns
    assert 'epochs_completed' in df.columns

    # Check that training ran for exactly 3 epochs
    max_epoch = df['epochs_completed'].max()
    assert max_epoch == 3, f"Expected 3 epochs, got: {max_epoch}"
    print(f"✓ Training completed {max_epoch} epochs")

    # Verify train loss was logged
    train_losses = df[df['loss_dataset'] == 'train']
    assert not train_losses.empty, "No training losses logged"
    assert len(train_losses) == 3, f"Expected 3 train loss entries, got {len(train_losses)}"
    print(f"✓ Training losses logged: {len(train_losses)} entries")

    # Verify eval losses were logged for all authors
    for author in AUTHORS:
        author_losses = df[df['loss_dataset'] == author]
        assert not author_losses.empty, f"No eval losses for {author}"
    print(f"✓ Eval losses logged for all {len(AUTHORS)} authors")

    print(f"✓ {variant_name.upper()} VARIANT TEST PASSED")

    # Clean up test model
    print(f"Cleaning up test model...")
    shutil.rmtree(model_dir)


def test_all_variants():
    """Test training on baseline and all three variants."""
    print("\n" + "="*60)
    print("INTEGRATION TEST: All Variants")
    print("="*60)
    print("Testing: baseline, content, function, pos")
    print("Expected time: 8-15 minutes total")
    print("="*60)

    # Test baseline
    train_variant_model(None)

    # Test each variant
    for variant in ANALYSIS_VARIANTS:
        train_variant_model(variant)

    print("\n" + "="*60)
    print("✓ ALL VARIANT TESTS PASSED")
    print("="*60)


if __name__ == "__main__":
    test_all_variants()