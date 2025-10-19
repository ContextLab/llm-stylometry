#!/usr/bin/env python
"""
Check training status for baseline and variant models.

This script analyzes model directories and loss logs to provide a comprehensive
status report including completed models, in-progress training, and estimated
time to completion.
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict
import json


def parse_model_name(dir_name):
    """Parse model directory name to extract components."""
    parts = dir_name.split('_')

    author = parts[0]
    variant = None
    tokenizer = None
    seed = None

    for part in parts[1:]:
        if part.startswith('variant='):
            variant = part.split('=')[1]
        elif part.startswith('tokenizer='):
            tokenizer = part.split('=')[1]
        elif part.startswith('seed='):
            seed = int(part.split('=')[1])

    return author, variant, tokenizer, seed


def get_model_status(model_dir):
    """Get status of a single model from its loss logs and checkpoints."""
    loss_logs_path = model_dir / 'loss_logs.csv'

    if not loss_logs_path.exists():
        return None

    try:
        df = pd.read_csv(loss_logs_path)

        if len(df) == 0:
            return None

        # Get latest epoch
        max_epoch = df['epochs_completed'].max()

        # Get training loss at max epoch
        train_loss_row = df[(df['epochs_completed'] == max_epoch) &
                            (df['loss_dataset'] == 'train')]

        if len(train_loss_row) == 0:
            current_loss = None
        else:
            current_loss = train_loss_row.iloc[0]['loss_value']

        # Get final epoch losses (last epoch with data)
        final_epoch_data = df[df['epochs_completed'] == max_epoch].copy()

        # Check if complete (500 epochs or more)
        is_complete = max_epoch >= 500

        # Get timestamp from loss_logs.csv modification time
        last_modified = datetime.fromtimestamp(loss_logs_path.stat().st_mtime)

        # Estimate start time from the most recent training log file
        # Look in the logs directory for training logs
        logs_dir = model_dir.parent.parent / 'logs'
        start_time = datetime.fromtimestamp(loss_logs_path.stat().st_ctime)  # fallback

        if logs_dir.exists():
            # Find the most recent training log
            log_files = sorted(logs_dir.glob('training_*.log'), key=lambda f: f.stat().st_mtime, reverse=True)
            if log_files:
                # Parse the first line to get actual start time
                # Format: "Training started at Thu Oct 16 14:47:54 EDT 2025"
                try:
                    with open(log_files[0], 'r') as f:
                        first_line = f.readline().strip()
                        if first_line.startswith('Training started at '):
                            time_str = first_line.replace('Training started at ', '')
                            # Parse the timestamp
                            start_time = datetime.strptime(time_str, '%a %b %d %H:%M:%S %Z %Y')
                except Exception:
                    # Fall back to file modification time
                    start_time = datetime.fromtimestamp(log_files[0].stat().st_mtime)

        return {
            'current_epoch': max_epoch,
            'current_loss': current_loss,
            'is_complete': is_complete,
            'final_epoch_data': final_epoch_data,
            'last_modified': last_modified,
            'start_time': start_time,
            'total_epochs': len(df['epochs_completed'].unique())
        }
    except Exception as e:
        print(f"Error reading {model_dir}: {e}")
        return None


def format_timedelta(td):
    """Format timedelta as human-readable string."""
    total_seconds = int(td.total_seconds())

    days = total_seconds // 86400
    hours = (total_seconds % 86400) // 3600
    minutes = (total_seconds % 3600) // 60

    if days > 0:
        return f"{days}d {hours}h {minutes}m"
    elif hours > 0:
        return f"{hours}h {minutes}m"
    else:
        return f"{minutes}m"


def analyze_training_status(models_dir='models'):
    """Analyze training status for all models."""
    models_path = Path(models_dir)

    if not models_path.exists():
        print(f"Error: Models directory not found: {models_dir}")
        return

    # Organize by variant and author
    baseline_models = defaultdict(list)  # author -> list of (seed, status)
    variant_models = defaultdict(lambda: defaultdict(list))  # variant -> author -> list of (seed, status)

    # Scan all model directories
    for model_dir in sorted(models_path.iterdir()):
        if not model_dir.is_dir():
            continue

        author, variant, tokenizer, seed = parse_model_name(model_dir.name)

        if author is None or seed is None:
            continue

        status = get_model_status(model_dir)

        if status is None:
            continue

        if variant is None:
            baseline_models[author].append((seed, status))
        else:
            variant_models[variant][author].append((seed, status))

    # Print report
    print("=" * 80)
    print("TRAINING STATUS REPORT")
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    # Baseline models
    if baseline_models:
        print("\n" + "=" * 80)
        print("BASELINE MODELS")
        print("=" * 80)
        print_variant_status(baseline_models, "baseline")

    # Variant models
    for variant in sorted(variant_models.keys()):
        print("\n" + "=" * 80)
        print(f"{variant.upper()} VARIANT MODELS")
        print("=" * 80)
        print_variant_status(variant_models[variant], variant)


def print_variant_status(author_models, variant_name):
    """Print status for a variant (baseline or specific variant)."""

    all_complete = True
    in_progress_count = 0

    for author in sorted(author_models.keys()):
        models = author_models[author]

        # Separate complete and in-progress
        complete = [(seed, status) for seed, status in models if status['is_complete']]
        in_progress = [(seed, status) for seed, status in models if not status['is_complete']]

        if in_progress:
            all_complete = False
            in_progress_count += len(in_progress)

        print(f"\n{author.upper()}")
        print("-" * 80)

        # Completed models
        if complete:
            # Calculate mean and std of final epoch training losses
            final_losses = []
            for seed, status in complete:
                train_row = status['final_epoch_data'][
                    status['final_epoch_data']['loss_dataset'] == 'train'
                ]
                if len(train_row) > 0:
                    final_losses.append(train_row.iloc[0]['loss_value'])

            if final_losses:
                mean_loss = np.mean(final_losses)
                std_loss = np.std(final_losses)
                print(f"  Completed: {len(complete)}/10 seeds")
                print(f"  Final training loss: {mean_loss:.4f} ± {std_loss:.4f} (mean ± std)")
            else:
                print(f"  Completed: {len(complete)}/10 seeds (no loss data)")
        else:
            print(f"  Completed: 0/10 seeds")

        # In-progress models
        if in_progress:
            print(f"  In-progress: {len(in_progress)} seeds")

            for seed, status in sorted(in_progress):
                epoch = status['current_epoch']
                loss = status['current_loss']

                # Estimate time to completion
                # Use current time instead of last_modified for accurate elapsed time
                elapsed = datetime.now() - status['start_time']

                if epoch > 0:
                    avg_time_per_epoch = elapsed / epoch
                    remaining_epochs = 500 - epoch
                    eta = avg_time_per_epoch * remaining_epochs

                    progress_pct = (epoch / 500) * 100

                    loss_str = f"{loss:.4f}" if loss is not None else "N/A"
                    print(f"    Seed {seed}: epoch {epoch}/500 ({progress_pct:.1f}%) | "
                          f"loss: {loss_str} | ETA: {format_timedelta(eta)}")
                else:
                    print(f"    Seed {seed}: epoch {epoch}/500 (starting...)")

    # Summary
    total_expected = len(author_models) * 10  # 10 seeds per author
    total_complete = sum(len([s for s, st in models if st['is_complete']])
                        for models in author_models.values())
    total_in_progress = sum(len([s for s, st in models if not st['is_complete']])
                           for models in author_models.values())

    print("\n" + "-" * 80)
    print(f"Summary: {total_complete}/{total_expected} complete, "
          f"{total_in_progress} in progress")

    if total_in_progress > 0:
        # Overall ETA based on all in-progress models
        all_etas = []
        for author, models in author_models.items():
            for seed, status in models:
                if not status['is_complete'] and status['current_epoch'] > 0:
                    # Use current time for accurate elapsed calculation
                    elapsed = datetime.now() - status['start_time']
                    avg_time_per_epoch = elapsed / status['current_epoch']
                    remaining_epochs = 500 - status['current_epoch']
                    eta = avg_time_per_epoch * remaining_epochs
                    all_etas.append(eta)

        if all_etas:
            max_eta = max(all_etas)
            avg_eta = sum(all_etas, timedelta()) / len(all_etas)
            print(f"Estimated completion: {format_timedelta(max_eta)} (longest), "
                  f"{format_timedelta(avg_eta)} (average)")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Check training status for baseline and variant models'
    )
    parser.add_argument(
        '--models-dir',
        default='models',
        help='Directory containing trained models (default: models)'
    )

    args = parser.parse_args()

    analyze_training_status(args.models_dir)


if __name__ == '__main__':
    main()
