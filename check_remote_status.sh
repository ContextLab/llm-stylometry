#!/bin/bash
#
# Check training status on remote GPU server
#
# This script connects to a remote GPU server and analyzes the training
# status of all models (baseline + variants), providing statistics on
# completed models and estimates for in-progress training.
#
# Usage:
#   ./check_remote_status.sh --cluster CLUSTER_NAME
#

set -e

# Color output helpers
print_info() {
    echo -e "\033[0;34m[INFO]\033[0m $1"
}

print_error() {
    echo -e "\033[0;31m[ERROR]\033[0m $1" >&2
}

print_warning() {
    echo -e "\033[0;33m[WARNING]\033[0m $1"
}

print_success() {
    echo -e "\033[0;32m[SUCCESS]\033[0m $1"
}

# Default cluster
CLUSTER=""  # Must be specified with --cluster flag

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --cluster)
            CLUSTER="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Check training status on remote GPU server"
            echo ""
            echo "Options:"
            echo "  --cluster NAME          Select cluster (required)"
            echo "  -h, --help             Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0 --cluster mycluster          # Check status on mycluster"
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Validate cluster is specified
if [ -z "$CLUSTER" ]; then
    print_error "Cluster must be specified with --cluster flag"
    echo "Example: $0 --cluster mycluster"
    exit 1
fi

# Read credentials from config file
CRED_FILE=".ssh/credentials_${CLUSTER}.json"
if [ -f "$CRED_FILE" ]; then
    print_info "Found credentials file for $CLUSTER, using saved credentials..."
    SERVER_ADDRESS=$(python3 -c "import json; print(json.load(open('$CRED_FILE'))['server'])" 2>/dev/null)
    USERNAME=$(python3 -c "import json; print(json.load(open('$CRED_FILE'))['username'])" 2>/dev/null)
    PASSWORD=$(python3 -c "import json; print(json.load(open('$CRED_FILE'))['password'])" 2>/dev/null)

    if [ -z "$SERVER_ADDRESS" ] || [ -z "$USERNAME" ] || [ -z "$PASSWORD" ]; then
        print_error "Failed to read credentials from $CRED_FILE"
        exit 1
    fi

    USE_SSHPASS=true
else
    print_warning "No credentials file found at $CRED_FILE"
    read -p "Enter GPU server address (hostname or IP): " SERVER_ADDRESS
    if [ -z "$SERVER_ADDRESS" ]; then
        print_error "Server address cannot be empty"
        exit 1
    fi

    read -p "Enter username for $SERVER_ADDRESS: " USERNAME
    if [ -z "$USERNAME" ]; then
        print_error "Username cannot be empty"
        exit 1
    fi

    USE_SSHPASS=false
fi

print_info "Connecting to $USERNAME@$SERVER_ADDRESS..."

# Build SSH command based on authentication method
if [ "$USE_SSHPASS" = true ]; then
    # Use sshpass for password authentication
    if ! command -v sshpass &> /dev/null; then
        print_error "sshpass is required but not installed. Please install it: brew install hudochenkov/sshpass/sshpass"
        exit 1
    fi
    SSH_CMD="sshpass -p '$PASSWORD' ssh -o StrictHostKeyChecking=no"
else
    # Test SSH connection first with interactive authentication
    if ! ssh -o ConnectTimeout=5 -o BatchMode=yes "$USERNAME@$SERVER_ADDRESS" "echo 'Connection test successful'" 2>/dev/null; then
        print_warning "Initial connection test failed. Trying with interactive authentication..."
    fi
    echo
    SSH_CMD="ssh"
fi

print_info "Checking training status on $CLUSTER..."
echo ""

# Transfer Python script to remote server and execute it
# We'll use a heredoc to send the Python script content
eval "$SSH_CMD \"$USERNAME@$SERVER_ADDRESS\" 'bash -s'" << 'ENDSSH'
#!/bin/bash

# Change to project directory
cd ~/llm-stylometry || { echo "ERROR: Project directory ~/llm-stylometry not found"; exit 1; }

# Activate conda environment (assumes it was set up by remote_train.sh)
if ! command -v conda &> /dev/null; then
    echo "ERROR: conda not found. Please run remote_train.sh first to set up the environment."
    exit 1
fi

eval "$(conda shell.bash hook)" 2>/dev/null || { echo "ERROR: Failed to initialize conda"; exit 1; }
conda activate llm-stylometry 2>/dev/null || { echo "ERROR: llm-stylometry environment not found. Please run remote_train.sh first."; exit 1; }

# Create temporary Python script
cat > /tmp/check_training_status.py << 'ENDPYTHON'
#!/usr/bin/env python
"""
Check training status for baseline and variant models.

This script analyzes model directories and loss logs to provide a comprehensive
status report including completed models, in-progress training, and estimated
time to completion.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict


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


# Run analysis
analyze_training_status('models')
ENDPYTHON

# Execute the Python script
python3 /tmp/check_training_status.py

# Clean up
rm -f /tmp/check_training_status.py
ENDSSH

if [ $? -eq 0 ]; then
    echo ""
    print_success "Status check complete!"
else
    print_error "Failed to check training status"
    exit 1
fi
