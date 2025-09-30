#!/usr/bin/env python
"""
Create test models for variant testing (without cleanup).
Uses same logic as test_variant_training.py but keeps models for consolidation testing.
"""

import os
import sys
import shutil
from pathlib import Path

# Add code to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'code'))

from experiment import Experiment
from constants import AUTHORS, MODELS_DIR, get_data_dir, ANALYSIS_VARIANTS

def train_variant_model(variant, test_author="fitzgerald", test_seed=42, cleanup=False):
    """Train a single test model on specified variant (or None for baseline)."""
    variant_name = variant or "baseline"
    print("\n" + "="*60)
    print(f"Creating test model: {variant_name} variant")
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

    model_dir = MODELS_DIR / exp.name

    # If model already exists, skip training
    if model_dir.exists():
        print(f"✓ Model already exists: {model_dir}")
        return model_dir

    # Set environment variable to prevent main.py from running at import time
    os.environ['NO_MULTIPROCESSING'] = '1'

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
    print(f"✓ {variant_name.upper()} model created: {model_dir}")

    return model_dir


def main():
    """Create test models for all variants with multiple authors and seeds."""
    import argparse

    parser = argparse.ArgumentParser(description='Create test models for variant testing')
    parser.add_argument('--authors', nargs='+', default=['fitzgerald', 'twain', 'austen'],
                       help='Authors to create models for (default: fitzgerald, twain, austen)')
    parser.add_argument('--seeds', nargs='+', type=int, default=[42, 43, 44],
                       help='Seeds to use (default: 42, 43, 44)')
    parser.add_argument('--variants', nargs='+', default=['baseline', 'content', 'function', 'pos'],
                       help='Variants to create (default: all)')

    args = parser.parse_args()

    # Convert 'baseline' to None in variants list
    variants_to_create = [None if v == 'baseline' else v for v in args.variants]

    total_models = len(args.authors) * len(args.seeds) * len(variants_to_create)

    print("\n" + "="*60)
    print("Creating Comprehensive Test Models for Variant Testing")
    print("="*60)
    print(f"Authors: {', '.join(args.authors)}")
    print(f"Seeds: {', '.join(map(str, args.seeds))}")
    print(f"Variants: {', '.join(args.variants)}")
    print(f"Total models to create: {total_models}")
    print(f"Estimated time: ~{total_models * 2.5:.0f} minutes (2-3 min per model)")
    print("="*60)

    response = input("\nProceed? [y/N]: ")
    if response.lower() != 'y':
        print("Cancelled.")
        return

    models_created = []
    models_skipped = []

    for author in args.authors:
        for seed in args.seeds:
            for variant in variants_to_create:
                variant_name = variant or 'baseline'
                print(f"\n[{len(models_created)+len(models_skipped)+1}/{total_models}] {author}, seed={seed}, variant={variant_name}")

                try:
                    model_dir = train_variant_model(variant, test_author=author, test_seed=seed)
                    models_created.append(model_dir)
                except Exception as e:
                    print(f"✗ ERROR: {e}")
                    models_skipped.append((author, seed, variant_name))

    print("\n" + "="*60)
    print("✓ MODEL CREATION COMPLETE")
    print("="*60)
    print(f"Successfully created: {len(models_created)} models")
    if models_skipped:
        print(f"Skipped (errors): {len(models_skipped)} models")
        for author, seed, variant in models_skipped:
            print(f"  - {author}, seed={seed}, variant={variant}")
    print("\nThese models are ready for consolidation and testing.")
    print("="*60)


if __name__ == "__main__":
    main()