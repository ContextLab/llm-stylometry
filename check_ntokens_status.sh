#!/bin/bash

# Check ntokens sweep training status on remote GPU server
#
# Usage: ./check_ntokens_status.sh --cluster tensor02

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
print_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
print_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
print_error() { echo -e "${RED}[ERROR]${NC} $1"; }

CLUSTER=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --cluster)
            CLUSTER="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 --cluster NAME"
            echo "Check ntokens sweep training status on remote GPU server"
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

if [ -z "$CLUSTER" ]; then
    print_error "Cluster must be specified with --cluster flag"
    exit 1
fi

CRED_FILE=".ssh/credentials_${CLUSTER}.json"
if [ -f "$CRED_FILE" ]; then
    SERVER_ADDRESS=$(python3 -c "import json; print(json.load(open('$CRED_FILE'))['server'])" 2>/dev/null)
    USERNAME=$(python3 -c "import json; print(json.load(open('$CRED_FILE'))['username'])" 2>/dev/null)
    PASSWORD=$(python3 -c "import json; print(json.load(open('$CRED_FILE'))['password'])" 2>/dev/null)

    if [ -z "$SERVER_ADDRESS" ] || [ -z "$USERNAME" ] || [ -z "$PASSWORD" ]; then
        print_error "Failed to read credentials from $CRED_FILE"
        exit 1
    fi
    USE_SSHPASS=true
else
    print_warning "No credentials file at $CRED_FILE"
    read -p "Enter server address: " SERVER_ADDRESS
    read -p "Enter username: " USERNAME
    USE_SSHPASS=false
fi

if [ "$USE_SSHPASS" = true ]; then
    if ! command -v sshpass &> /dev/null; then
        print_error "sshpass required: brew install hudochenkov/sshpass/sshpass"
        exit 1
    fi
    SSH_CMD="sshpass -p '$PASSWORD' ssh -o StrictHostKeyChecking=no"
else
    SSH_CMD="ssh"
fi

print_info "Checking ntokens sweep status on $CLUSTER..."
echo ""

eval "$SSH_CMD \"$USERNAME@$SERVER_ADDRESS\" 'bash -s'" << 'ENDSSH'
#!/bin/bash

cd ~/llm-stylometry || { echo "ERROR: Project directory not found"; exit 1; }

if ! command -v conda &> /dev/null; then
    echo "ERROR: conda not found"
    exit 1
fi

eval "$(conda shell.bash hook)" 2>/dev/null || true
conda activate llm-stylometry 2>/dev/null || true

python3 << 'ENDPYTHON'
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import defaultdict

models_dir = Path("models")
if not models_dir.exists():
    print("ERROR: models/ directory not found")
    exit(1)

# Token levels we expect
TOKEN_LEVELS = [2500, 5000, 10000, 20000, 40000, 45000, 50000, 55000,
                60000, 65000, 70000, 75000, 80000, 128608, 160000,
                257216, 385825, 514433, 643041]
AUTHORS = ["austen", "baum", "dickens", "fitzgerald", "melville",
           "thompson", "twain", "wells"]
SEEDS = list(range(10))

print("=" * 70)
print("NTOKENS SWEEP TRAINING STATUS")
print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 70)

total_expected = len(TOKEN_LEVELS) * len(AUTHORS) * len(SEEDS)
total_complete = 0
total_in_progress = 0
total_missing = 0

for n_tokens in TOKEN_LEVELS:
    complete = 0
    in_progress = 0
    missing = 0
    max_epoch = 0
    min_loss = float('inf')

    for author in AUTHORS:
        for seed in SEEDS:
            if n_tokens == 643041:
                # Legacy baseline naming (no ntokens in name)
                model_name = f"{author}_tokenizer=gpt2_seed={seed}"
            else:
                model_name = f"{author}_tokenizer=gpt2_ntokens={n_tokens}_seed={seed}"

            model_dir = models_dir / model_name
            loss_file = model_dir / "loss_logs.csv"

            if not model_dir.exists():
                missing += 1
                continue

            if not loss_file.exists():
                missing += 1
                continue

            try:
                df = pd.read_csv(loss_file)
                if df.empty:
                    missing += 1
                    continue

                epoch = df["epochs_completed"].max()
                train_loss = df[(df["epochs_completed"] == epoch) &
                               (df["loss_dataset"] == "train")]["loss_value"]

                if not train_loss.empty:
                    loss = train_loss.iloc[0]
                    if loss <= 3.0 and epoch >= 500:
                        complete += 1
                    else:
                        in_progress += 1
                        max_epoch = max(max_epoch, epoch)
                        min_loss = min(min_loss, loss)
                else:
                    in_progress += 1
            except Exception:
                missing += 1

    total = len(AUTHORS) * len(SEEDS)
    total_complete += complete
    total_in_progress += in_progress
    total_missing += missing

    status = "✓" if complete == total else "..." if in_progress > 0 else "✗"
    print(f"\n{n_tokens:>7,} tokens: {complete:>2}/{total} complete", end="")
    if in_progress > 0:
        loss_str = f", best loss: {min_loss:.3f}" if min_loss < float('inf') else ""
        print(f", {in_progress} in progress (max epoch: {max_epoch}{loss_str})", end="")
    if missing > 0:
        print(f", {missing} missing", end="")
    print(f"  [{status}]")

print("\n" + "-" * 70)
print(f"Total: {total_complete}/{total_expected} complete, "
      f"{total_in_progress} in progress, {total_missing} missing")

# Check screen session
import subprocess
result = subprocess.run(["screen", "-list"], capture_output=True, text=True)
if "ntokens_training" in result.stdout:
    print("\n✓ Screen session 'ntokens_training' is active")
else:
    print("\n✗ No active 'ntokens_training' screen session")

# Check latest log
import glob
logs = sorted(glob.glob("logs/ntokens_training_*.log"))
if logs:
    latest = logs[-1]
    print(f"Latest log: {latest}")
    # Show last 3 lines
    with open(latest) as f:
        lines = f.readlines()
        for line in lines[-3:]:
            print(f"  {line.rstrip()}")
ENDPYTHON
ENDSSH

if [ $? -eq 0 ]; then
    echo ""
    print_success "Status check complete!"
else
    print_error "Failed to check status"
    exit 1
fi
