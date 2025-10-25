#!/bin/bash
#
# Check HuggingFace model training status on remote GPU server
#
# Adapted from check_remote_status.sh pattern
#

set -e

# Color output helpers
print_info() {
    echo -e "\033[0;34m[INFO]\033[0m $1"
}

print_error() {
    echo -e "\033[0;31m[ERROR]\033[0m $1" >&2
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
            echo "Check HuggingFace model training status on remote GPU server"
            echo ""
            echo "Options:"
            echo "  --cluster NAME          Select cluster (required)"
            echo "  -h, --help             Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0 --cluster tensor02"
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
if [ ! -f "$CRED_FILE" ]; then
    print_error "Credentials file not found: $CRED_FILE"
    exit 1
fi

SERVER_ADDRESS=$(python3 -c "import json; print(json.load(open('$CRED_FILE'))['server'])" 2>/dev/null)
USERNAME=$(python3 -c "import json; print(json.load(open('$CRED_FILE'))['username'])" 2>/dev/null)
PASSWORD=$(python3 -c "import json; print(json.load(open('$CRED_FILE'))['password'])" 2>/dev/null)

if [ -z "$SERVER_ADDRESS" ] || [ -z "$USERNAME" ] || [ -z "$PASSWORD" ]; then
    print_error "Failed to read credentials from $CRED_FILE"
    exit 1
fi

# Setup SSH command
if ! command -v sshpass &> /dev/null; then
    print_error "sshpass is required but not installed"
    exit 1
fi

SSH_CMD="sshpass -p '$PASSWORD' ssh -o StrictHostKeyChecking=no"

print_info "Connecting to $USERNAME@$SERVER_ADDRESS..."
print_info "Checking HF training status on $CLUSTER..."
echo ""

# Execute status check on remote server
eval "$SSH_CMD \"$USERNAME@$SERVER_ADDRESS\" 'bash -s'" << 'ENDSSH'
#!/bin/bash

# Change to project directory
cd ~/llm-stylometry || { echo "ERROR: Project directory ~/llm-stylometry not found"; exit 1; }

# Activate conda environment
if ! command -v conda &> /dev/null; then
    echo "ERROR: conda not found"
    exit 1
fi

eval "$(conda shell.bash hook)" 2>/dev/null || { echo "ERROR: Failed to initialize conda"; exit 1; }
conda activate llm-stylometry 2>/dev/null || { echo "ERROR: llm-stylometry environment not found"; exit 1; }

# Create temporary Python script
cat > /tmp/check_hf_status.py << 'ENDPYTHON'
#!/usr/bin/env python
"""Check HuggingFace training status."""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

AUTHORS = ['austen', 'baum', 'dickens', 'fitzgerald', 'melville', 'thompson', 'twain', 'wells']
TARGET_LOSS = 0.1  # HF target loss
PAPER_LOSS = 3.0   # Paper stopping point

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

def check_author_status(author):
    """Check HF training status for a single author."""
    # Check seed=0 model (HF training location)
    model_dir = Path(f'models/{author}_tokenizer=gpt2_seed=0')
    loss_log = model_dir / 'loss_logs.csv'

    print(f"[DEBUG] Checking {author}: {loss_log}", file=sys.stderr)

    if not loss_log.exists():
        print(f"[DEBUG] Loss log not found for {author}", file=sys.stderr)
        return None

    try:
        df = pd.read_csv(loss_log)
        print(f"[DEBUG] {author}: {len(df)} rows in loss log", file=sys.stderr)
        if len(df) == 0:
            return None

        # Get latest epoch
        max_epoch = df['epochs_completed'].max()
        train_rows = df[(df['epochs_completed'] == max_epoch) & (df['loss_dataset'] == 'train')]

        if len(train_rows) == 0:
            return None

        current_loss = train_rows.iloc[0]['loss_value']

        # Check if we're in HF training (loss < PAPER_LOSS)
        hf_rows = df[(df['loss_dataset'] == 'train') & (df['loss_value'] < PAPER_LOSS)]
        print(f"[DEBUG] {author}: HF rows (loss < {PAPER_LOSS}): {len(hf_rows)}", file=sys.stderr)

        if len(hf_rows) == 0:
            # Not yet started HF training
            print(f"[DEBUG] {author}: Returning None - no HF training yet", file=sys.stderr)
            return None

        # Find when HF training started
        hf_start_epoch = int(hf_rows.iloc[0]['epochs_completed'])
        epochs_since_start = int(max_epoch - hf_start_epoch)
        print(f"[DEBUG] {author}: HF start epoch {hf_start_epoch}, epochs since: {epochs_since_start}", file=sys.stderr)

        # Estimate elapsed time (rough: 10 sec/epoch with eval skipped)
        elapsed = timedelta(seconds=int(epochs_since_start * 10))
        print(f"[DEBUG] {author}: About to return status dict", file=sys.stderr)

        # Check if complete
        is_complete = current_loss <= TARGET_LOSS

        result = {
            'current_epoch': max_epoch,
            'current_loss': current_loss,
            'target_loss': TARGET_LOSS,
            'is_complete': is_complete,
            'hf_start_epoch': hf_start_epoch,
            'epochs_since_start': epochs_since_start,
            'elapsed': elapsed
        }

        print(f"[DEBUG] {author}: Returning status - epoch {max_epoch}, loss {current_loss:.4f}", file=sys.stderr)
        return result

    except Exception as e:
        print(f"[DEBUG] {author}: Exception: {e}", file=sys.stderr)
        return None

# Print report
print("=" * 80)
print("HUGGINGFACE MODEL TRAINING STATUS")
print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 80)

completed_count = 0
in_progress_count = 0
not_started_count = 0

for author in AUTHORS:
    status = check_author_status(author)

    print(f"\n{author.upper()}")
    print("-" * 80)

    if status is None:
        print("  Status: Not started")
        not_started_count += 1
    elif status['is_complete']:
        print(f"  Status: Complete ✓")
        print(f"  Final loss: {status['current_loss']:.4f}")
        print(f"  Total epochs: {status['current_epoch']:,}")
        print(f"  HF epochs: {status['epochs_since_start']:,} (from epoch {status['hf_start_epoch']})")
        completed_count += 1
    else:
        print(f"  Status: Training...")
        print(f"  Current epoch: {status['current_epoch']:,}")
        print(f"  Current loss: {status['current_loss']:.4f}")
        print(f"  Target loss: {status['target_loss']:.4f}")
        print(f"  HF epochs completed: {status['epochs_since_start']:,}")
        print(f"  Elapsed: {format_timedelta(status['elapsed'])}")

        # Estimate remaining time based on loss decay
        if status['epochs_since_start'] > 10:
            # Rough estimate: assume exponential decay
            # loss goes from ~3.0 to ~0.1 (factor of 30)
            # Current progress
            loss_ratio = (PAPER_LOSS - status['current_loss']) / (PAPER_LOSS - TARGET_LOSS)
            progress_pct = loss_ratio * 100

            # Estimate total HF epochs needed (very rough)
            if loss_ratio > 0:
                estimated_total_hf_epochs = int(status['epochs_since_start'] / loss_ratio)
                remaining_epochs = estimated_total_hf_epochs - status['epochs_since_start']
                eta = timedelta(seconds=remaining_epochs * 10)
                print(f"  Progress: {progress_pct:.1f}%")
                print(f"  Estimated remaining: {format_timedelta(eta)}")

        in_progress_count += 1

# Summary
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"Completed: {completed_count}/8")
print(f"In progress: {in_progress_count}/8")
print(f"Not started: {not_started_count}/8")

if in_progress_count > 0 or completed_count < 8:
    print("\nTo download completed models:")
    print("  ./sync_hf_models.sh --cluster CLUSTER")

ENDPYTHON

# Execute the Python script
python3 /tmp/check_hf_status.py

# Clean up
rm -f /tmp/check_hf_status.py
ENDSSH

if [ $? -eq 0 ]; then
    echo ""
    print_success "Status check complete!"
else
    print_error "Failed to check training status"
    exit 1
fi
