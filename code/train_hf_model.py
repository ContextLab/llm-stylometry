#!/usr/bin/env python3
"""
Train high-quality models for HuggingFace deployment.

This script continues training from existing seed=0 models to much lower loss (0.1)
for better text generation quality.

Strategy:
1. Creates temporary model name: {author}_hf_temp_tokenizer=gpt2_seed=0
2. Trains in models/ directory (to work with existing infrastructure)
3. After training completes, renames to models_hf/{author}_tokenizer=gpt2

This approach avoids modifying core training code while achieving HF-compatible naming.
"""

import argparse
import sys
from pathlib import Path
import shutil

# Add code directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from experiment import Experiment
from constants import AUTHORS, MODELS_DIR
from main import run_experiment
import torch.multiprocessing as mp


def create_hf_experiment(
    author: str,
    target_loss: float = 0.1,
    max_epochs: int = 50000
):
    """
    Create an experiment configuration for HF model training.

    Strategy: Use temporary name with seed suffix during training,
    then rename to HF format after completion.

    Args:
        author: Author name (e.g., 'baum', 'austen')
        target_loss: Target training loss (default: 0.1)
        max_epochs: Maximum epochs (default: 50000, effectively unlimited)

    Returns:
        Experiment object configured for HF training
    """
    # Verify source model exists
    source_model = MODELS_DIR / f"{author}_tokenizer=gpt2_seed=0"
    if not source_model.exists():
        raise FileNotFoundError(
            f"Source model not found: {source_model}\n"
            f"Please ensure seed=0 baseline model exists for {author}"
        )

    # Verify source model has training_state.pt
    if not (source_model / "training_state.pt").exists():
        raise FileNotFoundError(
            f"Training state not found: {source_model}/training_state.pt\n"
            f"Cannot resume training without checkpoint"
        )

    # Use temporary name with _hf_ marker
    # This will train in models/ directory, then we'll move it
    temp_name = f"{author}_hf_temp_tokenizer=gpt2_seed=0"

    # Copy source model to temporary location for training
    temp_model_dir = MODELS_DIR / temp_name
    if temp_model_dir.exists():
        print(f"Removing existing temporary model: {temp_model_dir}")
        shutil.rmtree(temp_model_dir)

    print(f"Copying seed=0 model to temporary location: {temp_model_dir}")
    shutil.copytree(source_model, temp_model_dir)

    # Create experiment with HF-specific settings
    exp = Experiment(
        train_author=author,
        seed=0,  # Use same seed for consistency
        tokenizer_name="gpt2",
        stop_criteria={
            'train_loss': target_loss,
            'min_epochs': 0,  # No minimum (already trained 500+ epochs)
            'max_epochs': max_epochs
        },
        resume_training=True,  # Always resume from copied checkpoint
    )

    # Override experiment name to use temporary name
    exp.name = temp_name

    return exp


def main():
    parser = argparse.ArgumentParser(
        description='Train high-quality models for HuggingFace deployment'
    )
    parser.add_argument(
        '--author',
        required=True,
        choices=AUTHORS,
        help='Author to train'
    )
    parser.add_argument(
        '--target-loss',
        type=float,
        default=0.1,
        help='Target training loss (default: 0.1)'
    )
    parser.add_argument(
        '--output-dir',
        default='models_hf',
        help='Output directory for HF models (default: models_hf)'
    )
    parser.add_argument(
        '--max-epochs',
        type=int,
        default=50000,
        help='Maximum epochs (default: 50000)'
    )

    args = parser.parse_args()

    # Create experiment
    exp = create_hf_experiment(
        author=args.author,
        target_loss=args.target_loss,
        max_epochs=args.max_epochs
    )

    # Temporary model will be saved in models/
    temp_model_path = MODELS_DIR / exp.name

    # Final HF model will be in models_hf/
    output_dir = Path(args.output_dir)
    final_model_name = f"{args.author}_tokenizer=gpt2"
    final_model_path = output_dir / final_model_name

    print("="*60)
    print(f"Training HuggingFace Model: {args.author.capitalize()}")
    print("="*60)
    print(f"Author: {args.author}")
    print(f"Target loss: {args.target_loss}")
    print(f"Max epochs: {args.max_epochs}")
    print(f"Resume from: models/{args.author}_tokenizer=gpt2_seed=0")
    print(f"Temporary location: {temp_model_path}")
    print(f"Final location: {final_model_path}")
    print("="*60)
    print()

    # Determine device type
    import torch
    if torch.cuda.is_available():
        device_type = "cuda"
        print(f"Using CUDA GPU: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device_type = "mps"
        print("Using Apple Metal Performance Shaders (MPS)")
    else:
        device_type = "cpu"
        print("WARNING: No GPU available, training will be very slow")

    print(f"\nStarting training...")
    print(f"This will continue until train_loss <= {args.target_loss}")
    print(f"(May require 5,000-10,000+ epochs)\n")

    # Run training (no device queue needed for single model)
    device_queue = None
    run_experiment(exp, device_queue, device_type)

    # After training completes, move to final location
    print("\n" + "="*60)
    print(f"Training complete for {args.author}")
    print("="*60)

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Remove final destination if it exists
    if final_model_path.exists():
        print(f"Removing existing HF model: {final_model_path}")
        shutil.rmtree(final_model_path)

    # Move to final location and rename
    print(f"Moving {temp_model_path} -> {final_model_path}")
    shutil.move(str(temp_model_path), str(final_model_path))

    print(f"Model saved to: {final_model_path}")
    print("="*60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
