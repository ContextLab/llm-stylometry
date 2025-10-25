#!/usr/bin/env python3
"""
Train all 8 HuggingFace models in parallel using multiprocessing.

Uses the same parallel training pattern as the main experiments,
but with HF-specific configuration (target loss 0.1, only training loss evaluation).
"""

import sys
import os
import logging
from pathlib import Path
import torch
import torch.multiprocessing as mp

# Add code directory to path
sys.path.insert(0, str(Path(__file__).parent))

from experiment import Experiment
from constants import AUTHORS, MODELS_DIR
from main import run_experiment
import shutil

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def prepare_hf_experiments(authors, target_loss=0.1, max_epochs=50000):
    """
    Prepare HF experiments for specified authors.

    Args:
        authors: List of author names to train
        target_loss: Target training loss
        max_epochs: Maximum epochs

    Creates temporary copies of seed=0 models for HF training.
    """
    experiments = []

    for author in authors:
        # Verify source model exists
        source_model = MODELS_DIR / f"{author}_tokenizer=gpt2_seed=0"
        if not source_model.exists():
            logger.error(f"Source model not found for {author}: {source_model}")
            continue

        if not (source_model / "training_state.pt").exists():
            logger.error(f"Training state not found for {author}")
            continue

        # Create temporary name for HF training
        temp_name = f"{author}_hf_temp_tokenizer=gpt2_seed=0"
        temp_model_dir = MODELS_DIR / temp_name

        # Copy source model to temp location if not exists
        if not temp_model_dir.exists():
            logger.info(f"Copying {author} seed=0 model to {temp_name}")
            shutil.copytree(source_model, temp_model_dir)
        else:
            logger.info(f"Using existing temp model: {temp_name}")

        # Create experiment with HF settings
        exp = Experiment(
            train_author=author,
            seed=0,
            tokenizer_name="gpt2",
            stop_criteria={
                'train_loss': target_loss,
                'min_epochs': 0,  # No minimum, already trained
                'max_epochs': max_epochs
            },
            resume_training=True
        )

        # Override name to use temp model
        exp.name = temp_name

        # Override eval_paths to only include training author
        # This skips evaluation on other authors to save time
        exp.eval_paths = {author: exp.eval_paths[author]}

        experiments.append(exp)

    return experiments


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description='Train HuggingFace models in parallel using multiprocessing'
    )
    parser.add_argument(
        'authors',
        nargs='+',
        choices=AUTHORS,
        help='Authors to train (one or more)'
    )
    parser.add_argument(
        '--target-loss',
        type=float,
        default=0.1,
        help='Target training loss (default: 0.1)'
    )
    parser.add_argument(
        '--max-epochs',
        type=int,
        default=50000,
        help='Maximum epochs (default: 50000)'
    )
    parser.add_argument(
        '--max-gpus',
        type=int,
        default=8,
        help='Maximum GPUs to use (default: 8)'
    )

    args = parser.parse_args()

    print("="*60)
    print("Training HuggingFace Models (Parallel)")
    print("="*60)
    print(f"Authors: {', '.join(args.authors)} ({len(args.authors)} total)")
    print(f"Target loss: {args.target_loss}")
    print(f"Max epochs: {args.max_epochs}")
    print(f"Max GPUs: {args.max_gpus}")
    print("="*60)
    print()

    # Prepare experiments
    experiments = prepare_hf_experiments(
        authors=args.authors,
        target_loss=args.target_loss,
        max_epochs=args.max_epochs
    )

    if len(experiments) == 0:
        print("ERROR: No experiments prepared")
        return 1

    print(f"Prepared {len(experiments)} experiments")
    for exp in experiments:
        print(f"  - {exp.train_author}: {exp.name}")
    print()

    # Detect device
    if torch.cuda.is_available():
        device_type = "cuda"
        device_count = torch.cuda.device_count()
        gpu_count = min(device_count, args.max_gpus)
        print(f"Using {gpu_count} GPUs out of {device_count} available")
    elif torch.backends.mps.is_available():
        device_type = "mps"
        gpu_count = 1
        print("Using Apple Metal Performance Shaders (MPS)")
    else:
        device_type = "cpu"
        gpu_count = 1
        print("Using CPU (training will be slow)")

    print()
    print("Starting parallel training...")
    print()

    # Set up multiprocessing for parallel training
    if device_type == "cuda" and gpu_count > 1:
        mp.set_start_method("spawn", force=True)
        manager = mp.Manager()
        device_queue = manager.Queue()

        # Populate device queue
        for gpu in range(gpu_count):
            device_queue.put(gpu)

        # Create process pool
        pool = mp.Pool(processes=gpu_count)

        def error_callback(e):
            logger.exception("Error in worker process")
            pool.terminate()
            sys.exit(1)

        # Start all experiments
        for exp in experiments:
            pool.apply_async(
                run_experiment,
                (exp, device_queue, device_type),
                error_callback=error_callback
            )

        pool.close()
        pool.join()
    else:
        # Sequential mode
        print("Running in sequential mode")
        device_queue = None
        for exp in experiments:
            run_experiment(exp, device_queue, device_type)

    print()
    print("="*60)
    print("Training Complete - Moving to models_hf/")
    print("="*60)

    # Move completed models to models_hf/
    models_hf_dir = Path('models_hf')
    models_hf_dir.mkdir(exist_ok=True)

    for exp in experiments:
        temp_model_path = MODELS_DIR / exp.name
        final_model_name = f"{exp.train_author}_tokenizer=gpt2"
        final_model_path = models_hf_dir / final_model_name

        if temp_model_path.exists():
            if final_model_path.exists():
                print(f"Removing existing: {final_model_path}")
                shutil.rmtree(final_model_path)

            print(f"Moving: {exp.train_author} -> {final_model_path}")
            shutil.move(str(temp_model_path), str(final_model_path))
        else:
            print(f"WARNING: Temp model not found: {temp_model_path}")

    print()
    print("="*60)
    print("All models saved to models_hf/")
    print("="*60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
