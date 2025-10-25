#!/bin/bash

# Check HuggingFace Model Training Status
# Monitor training progress on remote GPU server

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
            echo "  $0 --cluster mycluster"
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

# Extract credentials using Python
SERVER_ADDRESS=$(python3 -c "import json; print(json.load(open('$CRED_FILE'))['server'])" 2>/dev/null)
USERNAME=$(python3 -c "import json; print(json.load(open('$CRED_FILE'))['username'])" 2>/dev/null)
PASSWORD=$(python3 -c "import json; print(json.load(open('$CRED_FILE'))['password'])" 2>/dev/null)

if [ -z "$SERVER_ADDRESS" ] || [ -z "$USERNAME" ] || [ -z "$PASSWORD" ]; then
    print_error "Failed to read credentials from $CRED_FILE"
    exit 1
fi

# Setup SSH command with password authentication
if ! command -v sshpass &> /dev/null; then
    print_error "sshpass is required but not installed. Please install it: brew install hudochenkov/sshpass/sshpass"
    exit 1
fi

SSH_CMD="sshpass -p '$PASSWORD' ssh -o StrictHostKeyChecking=no"

print_info "Connecting to $USERNAME@$SERVER_ADDRESS..."
print_info "Checking HF training status on $CLUSTER..."
echo ""

# Transfer Python script to remote server and execute it
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
"""Check HF training status."""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

AUTHORS = ['austen', 'baum', 'dickens', 'fitzgerald', 'melville', 'thompson', 'twain', 'wells']
TARGET_LOSS = 0.1  # Default target

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
    """Check training status for a single author."""
    # Check all possible HF training locations
    model_names = [
        f"{author}_tokenizer=gpt2_seed=0",  # Training directly on seed=0 (current approach)
        f"{author}_hf_temp_tokenizer=gpt2_seed=0",  # Legacy temp model approach
        f"{author}_tokenizer=gpt2"  # Final HF model (rare - usually in models_hf/)
    ]

    # Check in models/ directory
    models_dir = Path('models')
    for model_name in model_names:
        model_dir = models_dir / model_name
        loss_log = model_dir / 'loss_logs.csv'

        if loss_log.exists():
            try:
                df = pd.read_csv(loss_log)
                if len(df) == 0:
                    continue

                # Get latest epoch
                max_epoch = df['epochs_completed'].max()
                train_rows = df[(df['epochs_completed'] == max_epoch) & (df['loss_dataset'] == 'train')]

                if len(train_rows) == 0:
                    continue

                current_loss = train_rows.iloc[0]['loss_value']

                # Check if complete
                is_complete = current_loss <= TARGET_LOSS

                # Estimate time
                last_modified = datetime.fromtimestamp(loss_log.stat().st_mtime)
                start_time = datetime.fromtimestamp(loss_log.stat().st_ctime)
                elapsed = datetime.now() - start_time

                status = {
                    'model_name': model_name,
                    'current_epoch': max_epoch,
                    'current_loss': current_loss,
                    'target_loss': TARGET_LOSS,
                    'is_complete': is_complete,
                    'last_modified': last_modified,
                    'start_time': start_time,
                    'elapsed': elapsed
                }

                return status

            except Exception as e:
                continue

    # Check in models_hf/ directory
    models_hf_dir = Path('models_hf')
    if models_hf_dir.exists():
        hf_model_dir = models_hf_dir / f"{author}_tokenizer=gpt2"
        hf_loss_log = hf_model_dir / 'loss_logs.csv'

        if hf_loss_log.exists():
            try:
                df = pd.read_csv(hf_loss_log)
                max_epoch = df['epochs_completed'].max()
                train_rows = df[(df['epochs_completed'] == max_epoch) & (df['loss_dataset'] == 'train')]
                current_loss = train_rows.iloc[0]['loss_value']

                return {
                    'model_name': f"{author}_tokenizer=gpt2 (in models_hf/)",
                    'current_epoch': max_epoch,
                    'current_loss': current_loss,
                    'target_loss': TARGET_LOSS,
                    'is_complete': True,  # Already in final location
                    'last_modified': datetime.fromtimestamp(hf_loss_log.stat().st_mtime),
                    'start_time': datetime.fromtimestamp(hf_loss_log.stat().st_ctime),
                    'elapsed': datetime.now() - datetime.fromtimestamp(hf_loss_log.stat().st_ctime)
                }
            except:
                pass

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
        print(f"  Model: {status['model_name']}")
        print(f"  Final loss: {status['current_loss']:.4f}")
        print(f"  Epochs: {status['current_epoch']:,}")
        completed_count += 1
    else:
        print(f"  Status: Training...")
        print(f"  Model: {status['model_name']}")
        print(f"  Current epoch: {status['current_epoch']:,}")
        print(f"  Current loss: {status['current_loss']:.4f}")
        print(f"  Target loss: {status['target_loss']:.4f}")

        # Progress calculation
        if status['current_loss'] > status['target_loss']:
            # Estimate based on loss decay
            epochs = status['current_epoch']
            if epochs > 0:
                avg_time_per_epoch = status['elapsed'] / epochs
                # Rough estimate: assume exponential decay
                loss_ratio = np.log(status['current_loss'] / status['target_loss'])
                estimated_epochs_needed = int(epochs * loss_ratio)
                eta = avg_time_per_epoch * estimated_epochs_needed
                print(f"  Elapsed: {format_timedelta(status['elapsed'])}")
                print(f"  Estimated remaining: {format_timedelta(eta)}")
            else:
                print(f"  Elapsed: {format_timedelta(status['elapsed'])}")

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
