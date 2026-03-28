#!/bin/bash

# Sync ntokens model results from remote GPU server
# Downloads configs, loss logs, and generated configs (NOT model weights)
#
# Usage: ./sync_ntokens.sh --cluster tensor02

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
SYNC_WEIGHTS=false
SYNC_DATA=true

while [[ $# -gt 0 ]]; do
    case $1 in
        --cluster)
            CLUSTER="$2"
            shift 2
            ;;
        --include-weights)
            SYNC_WEIGHTS=true
            shift
            ;;
        --data-only)
            SYNC_DATA=true
            SYNC_WEIGHTS=false
            shift
            ;;
        -h|--help)
            echo "Usage: $0 --cluster NAME [options]"
            echo ""
            echo "Options:"
            echo "  --cluster NAME       Specify cluster (required)"
            echo "  --include-weights    Also download model weight files (large!)"
            echo "  --data-only          Only download consolidated results (default)"
            echo "  -h, --help           Show this help"
            echo ""
            echo "By default, syncs:"
            echo "  - Model configs (config.json, generation_config.json)"
            echo "  - Loss logs (loss_logs.csv)"
            echo "  - Consolidated results (model_results_ntokens.pkl.gz)"
            echo "  - NOT model weights (model.safetensors, training_state.pt)"
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

echo "=================================================="
echo "   LLM Stylometry — Sync Ntokens Results"
echo "=================================================="

CRED_FILE=".ssh/credentials_${CLUSTER}.json"
if [ ! -f "$CRED_FILE" ]; then
    print_error "Credentials file not found: $CRED_FILE"
    exit 1
fi

SERVER_ADDRESS=$(python3 -c "import json; print(json.load(open('$CRED_FILE'))['server'])")
USERNAME=$(python3 -c "import json; print(json.load(open('$CRED_FILE'))['username'])")
PASSWORD=$(python3 -c "import json; print(json.load(open('$CRED_FILE'))['password'])")

if [ -z "$SERVER_ADDRESS" ] || [ -z "$USERNAME" ]; then
    print_error "Failed to load credentials"
    exit 1
fi

if [ -n "$PASSWORD" ]; then
    if ! command -v sshpass &> /dev/null; then
        print_error "sshpass required: brew install hudochenkov/sshpass/sshpass"
        exit 1
    fi
    SSH_CMD="sshpass -p '$PASSWORD' ssh -o StrictHostKeyChecking=no"
    RSYNC_CMD="sshpass -p '$PASSWORD' rsync -e 'ssh -o StrictHostKeyChecking=no'"
else
    SSH_CMD="ssh"
    RSYNC_CMD="rsync"
fi

print_info "Connecting to $USERNAME@$SERVER_ADDRESS..."

# Count remote ntokens models
print_info "Checking remote ntokens models..."
REMOTE_COUNT=$(eval $SSH_CMD "$USERNAME@$SERVER_ADDRESS" "ls -d ~/llm-stylometry/models/*_ntokens=* 2>/dev/null | wc -l")
print_info "Found $REMOTE_COUNT ntokens model directories on remote"

if [ "$REMOTE_COUNT" -eq 0 ]; then
    print_warning "No ntokens models found on remote server"
    exit 0
fi

# Sync model directories (configs + loss logs, exclude weights by default)
LOCAL_MODELS="$PWD/models"
mkdir -p "$LOCAL_MODELS"

print_info "Syncing ntokens model configs and loss logs..."

EXCLUDE_FLAGS=""
if [ "$SYNC_WEIGHTS" = false ]; then
    EXCLUDE_FLAGS="--exclude='model.safetensors' --exclude='pytorch_model.bin' --exclude='model.pth' --exclude='training_state.pt'"
fi

eval $RSYNC_CMD -avz --progress \
    $EXCLUDE_FLAGS \
    --include="'*_ntokens=*/'" \
    --include="'*_ntokens=*/***'" \
    --exclude="'*'" \
    "'$USERNAME@$SERVER_ADDRESS:~/llm-stylometry/models/'" "'$LOCAL_MODELS/'"

if [ $? -eq 0 ]; then
    SYNCED=$(ls -d "$LOCAL_MODELS"/*_ntokens=* 2>/dev/null | wc -l)
    print_success "Synced $SYNCED ntokens model directories"
else
    print_error "Failed to sync model directories"
    exit 1
fi

# Sync consolidated results files
print_info "Checking for consolidated results..."
mkdir -p "$PWD/data"

for RFILE in model_results_ntokens.pkl.gz model_results_ntokens.pkl.gz t_test_ntokens_cache; do
    REMOTE_EXISTS=$(eval $SSH_CMD "$USERNAME@$SERVER_ADDRESS" "[ -e \"\$HOME/llm-stylometry/data/$RFILE\" ] && echo yes || echo no")
    if [ "$REMOTE_EXISTS" = "yes" ]; then
        print_info "Downloading data/$RFILE..."
        eval $RSYNC_CMD -avz \
            "'$USERNAME@$SERVER_ADDRESS:~/llm-stylometry/data/$RFILE'" \
            "'$PWD/data/$RFILE'"
        print_success "Downloaded $RFILE"
    fi
done

# Sync sigmoid fit results if available
SIGMOID_EXISTS=$(eval $SSH_CMD "$USERNAME@$SERVER_ADDRESS" "[ -f \"\$HOME/llm-stylometry/data/sigmoid_fit_results.json\" ] && echo yes || echo no")
if [ "$SIGMOID_EXISTS" = "yes" ]; then
    print_info "Downloading sigmoid_fit_results.json..."
    eval $RSYNC_CMD -avz \
        "'$USERNAME@$SERVER_ADDRESS:~/llm-stylometry/data/sigmoid_fit_results.json'" \
        "'$PWD/data/sigmoid_fit_results.json'"
fi

# Summary
echo
echo "=================================================="
echo "              Sync Complete!"
echo "=================================================="
LOCAL_COUNT=$(ls -d "$LOCAL_MODELS"/*_ntokens=* 2>/dev/null | wc -l)
echo "✓ Ntokens model directories: $LOCAL_COUNT"
echo "✓ Results in: $PWD/data/"
echo
echo "Next steps:"
echo "  python code/fit_sigmoid.py                  # Generate sigmoid figure"
echo "  python code/generate_ntokens_figures.py      # Generate t-test figure"
