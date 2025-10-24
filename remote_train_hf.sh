#!/bin/bash

# Remote Training Script for HuggingFace Models
# Train high-quality models on GPU cluster

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

# Default values
CLUSTER=""  # Must be specified with --cluster flag
TRAIN_AUTHOR=""
TRAIN_ALL=false
TARGET_LOSS=0.1

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --cluster)
            CLUSTER="$2"
            shift 2
            ;;
        --author)
            TRAIN_AUTHOR="$2"
            shift 2
            ;;
        --all)
            TRAIN_ALL=true
            shift
            ;;
        --target-loss)
            TARGET_LOSS="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Train HuggingFace models on remote GPU cluster"
            echo ""
            echo "Options:"
            echo "  --cluster NAME      GPU cluster name (required)"
            echo "  --author NAME       Train single author"
            echo "  --all               Train all 8 authors"
            echo "  --target-loss LOSS  Target training loss (default: 0.1)"
            echo "  -h, --help          Show this help"
            echo ""
            echo "Examples:"
            echo "  $0 --cluster mycluster --author baum"
            echo "  $0 --cluster mycluster --all"
            echo ""
            echo "Prerequisites:"
            echo "  - Credentials file: .ssh/credentials_CLUSTERNAME.json"
            echo "  - Format: {\"server\": \"hostname\", \"username\": \"user\", \"password\": \"pass\"}"
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
    echo "Example: $0 --cluster mycluster --all"
    exit 1
fi

# Validate author arguments
if [ "$TRAIN_ALL" = false ] && [ -z "$TRAIN_AUTHOR" ]; then
    print_error "Must specify --author or --all"
    echo "Use --help for usage information"
    exit 1
fi

# Read credentials from config file
CRED_FILE=".ssh/credentials_${CLUSTER}.json"
if [ ! -f "$CRED_FILE" ]; then
    print_error "Credentials file not found: $CRED_FILE"
    print_info "Please create credentials file with server, username, and password"
    exit 1
fi

# Extract credentials using Python
if ! command -v python3 &> /dev/null; then
    print_error "python3 is required to parse credentials file"
    exit 1
fi

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

# Build training flags
TRAIN_FLAGS=""
if [ "$TRAIN_ALL" = true ]; then
    TRAIN_FLAGS="--all"
else
    TRAIN_FLAGS="--author $TRAIN_AUTHOR"
fi

TRAIN_FLAGS="$TRAIN_FLAGS --target-loss $TARGET_LOSS"

echo
print_info "Training configuration:"
if [ "$TRAIN_ALL" = true ]; then
    echo "  Authors: All 8 authors"
else
    echo "  Author: $TRAIN_AUTHOR"
fi
echo "  Target loss: $TARGET_LOSS"
echo "  Cluster: $CLUSTER"
echo

# Connect and start training
eval "$SSH_CMD \"$USERNAME@$SERVER_ADDRESS\" 'TRAIN_FLAGS=\"$TRAIN_FLAGS\" bash -s'" << 'ENDSSH'
#!/bin/bash

# Change to project directory
cd ~/llm-stylometry || { echo "ERROR: Project directory ~/llm-stylometry not found"; exit 1; }

# Update repository with aggressive conflict resolution
echo "[INFO] Updating repository..."
# Clear any existing git state
git reset --hard HEAD
git clean -fd  # Remove untracked files
# Fetch and reset to latest
git fetch origin
git reset --hard origin/main
echo "[INFO] Repository updated to latest version"

# Activate conda environment
if ! command -v conda &> /dev/null; then
    echo "ERROR: conda not found on remote server"
    exit 1
fi

eval "$(conda shell.bash hook)" 2>/dev/null || { echo "ERROR: Failed to initialize conda"; exit 1; }
conda activate llm-stylometry 2>/dev/null || { echo "ERROR: llm-stylometry environment not found"; exit 1; }

# Create logs directory
mkdir -p logs

# Kill existing screen session if it exists
if screen -list | grep -q "hf_training"; then
    echo "[INFO] Killing existing hf_training screen session..."
    screen -S hf_training -X quit || true
    sleep 2
fi

# Create training script
cat > /tmp/hf_train.sh << TRAINSCRIPT
#!/bin/bash
cd ~/llm-stylometry
eval "\$(conda shell.bash hook)"
conda activate llm-stylometry

# Log start time
echo "HF model training started at \$(date)" > logs/hf_training.log

# Run training
./train_hf_models.sh $TRAIN_FLAGS 2>&1 | tee -a logs/hf_training.log

echo "HF model training completed at \$(date)" >> logs/hf_training.log
TRAINSCRIPT

chmod +x /tmp/hf_train.sh

# Start training in screen session
echo "[INFO] Starting training in screen session 'hf_training'..."
screen -dmS hf_training bash /tmp/hf_train.sh

echo "[SUCCESS] Training started on remote server"
echo ""
echo "To monitor progress:"
echo "  ./check_hf_status.sh --cluster $CLUSTER"
echo ""
echo "To view live training:"
echo "  ssh $USERNAME@$SERVER_ADDRESS"
echo "  screen -r hf_training"
echo "  (Press Ctrl+A then D to detach)"
ENDSSH

if [ $? -eq 0 ]; then
    echo
    print_success "Remote training started successfully!"
    echo
    echo "Training is running in a screen session on $CLUSTER"
    echo "It will continue even if you disconnect."
    echo
    echo "Monitor progress with:"
    echo "  ./check_hf_status.sh --cluster $CLUSTER"
else
    print_error "Failed to start remote training"
    exit 1
fi
