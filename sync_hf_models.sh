#!/bin/bash

# Sync HuggingFace Models from Remote Cluster
# Download completed HF models from GPU server to local machine

set -e

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
print_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
print_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
print_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Default values
CLUSTER=""  # Must be specified
SYNC_AUTHOR=""
SYNC_ALL=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --cluster)
            CLUSTER="$2"
            shift 2
            ;;
        --author)
            SYNC_AUTHOR="$2"
            shift 2
            ;;
        --all)
            SYNC_ALL=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Download HuggingFace models from remote cluster"
            echo ""
            echo "Options:"
            echo "  --cluster NAME      Cluster name (required)"
            echo "  --author NAME       Sync single author"
            echo "  --all               Sync all completed authors"
            echo "  -h, --help          Show this help"
            echo ""
            echo "Examples:"
            echo "  $0 --cluster mycluster --all"
            echo "  $0 --cluster mycluster --author baum"
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Validate arguments
if [ -z "$CLUSTER" ]; then
    print_error "Cluster must be specified with --cluster flag"
    exit 1
fi

if [ "$SYNC_ALL" = false ] && [ -z "$SYNC_AUTHOR" ]; then
    print_error "Must specify --author or --all"
    exit 1
fi

echo "=================================================="
echo "       HuggingFace Model Sync"
echo "=================================================="
echo
print_info "Cluster: $CLUSTER"
echo

# Load credentials
CRED_FILE=".ssh/credentials_${CLUSTER}.json"
if [ ! -f "$CRED_FILE" ]; then
    print_error "Credentials file not found: $CRED_FILE"
    exit 1
fi

SERVER_ADDRESS=$(python3 -c "import json; print(json.load(open('$CRED_FILE'))['server'])")
USERNAME=$(python3 -c "import json; print(json.load(open('$CRED_FILE'))['username'])")
PASSWORD=$(python3 -c "import json; print(json.load(open('$CRED_FILE'))['password'])")

if [ -z "$SERVER_ADDRESS" ] || [ -z "$USERNAME" ]; then
    print_error "Failed to load credentials from $CRED_FILE"
    exit 1
fi

# Setup SSH/rsync commands with password authentication
if [ -n "$PASSWORD" ]; then
    if ! command -v sshpass &> /dev/null; then
        print_error "sshpass is required but not installed"
        exit 1
    fi
    SSH_CMD="sshpass -p '$PASSWORD' ssh -o StrictHostKeyChecking=no"
    RSYNC_CMD="sshpass -p '$PASSWORD' rsync -e 'ssh -o StrictHostKeyChecking=no'"
else
    SSH_CMD="ssh"
    RSYNC_CMD="rsync"
fi

print_info "Connecting to: $USERNAME@$SERVER_ADDRESS"

# Check which models are complete on remote
print_info "Checking model status on remote server..."

AUTHORS_TO_SYNC=()

if [ "$SYNC_ALL" = true ]; then
    # Check all authors
    ALL_AUTHORS="austen baum dickens fitzgerald melville thompson twain wells"

    for author in $ALL_AUTHORS; do
        # Check if model exists and is complete on remote
        IS_COMPLETE=$(eval $SSH_CMD "$USERNAME@$SERVER_ADDRESS" \
            "cd ~/llm-stylometry && \
             [ -d models_hf/${author}_tokenizer=gpt2 ] && \
             [ -f models_hf/${author}_tokenizer=gpt2/model.safetensors ] && \
             echo 'yes' || echo 'no'")

        if [ "$IS_COMPLETE" = "yes" ]; then
            print_success "$author model complete"
            AUTHORS_TO_SYNC+=("$author")
        else
            print_warning "$author model not complete (skipping)"
        fi
    done
else
    # Single author
    IS_COMPLETE=$(eval $SSH_CMD "$USERNAME@$SERVER_ADDRESS" \
        "cd ~/llm-stylometry && \
         [ -d models_hf/${SYNC_AUTHOR}_tokenizer=gpt2 ] && \
         [ -f models_hf/${SYNC_AUTHOR}_tokenizer=gpt2/model.safetensors ] && \
         echo 'yes' || echo 'no'")

    if [ "$IS_COMPLETE" = "yes" ]; then
        print_success "$SYNC_AUTHOR model complete"
        AUTHORS_TO_SYNC+=("$SYNC_AUTHOR")
    else
        print_error "$SYNC_AUTHOR model not complete on remote"
        exit 1
    fi
fi

# Check if we have anything to sync
if [ ${#AUTHORS_TO_SYNC[@]} -eq 0 ]; then
    echo
    print_error "No complete models found to sync"
    exit 1
fi

echo
print_info "Will sync: ${AUTHORS_TO_SYNC[*]}"
echo

# Ask for confirmation
read -p "Continue with download? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    print_info "Download cancelled"
    exit 0
fi

# Create local models_hf directory
mkdir -p models_hf

# Sync each author
SYNCED_COUNT=0

for author in "${AUTHORS_TO_SYNC[@]}"; do
    print_info "Syncing $author model..."

    # Use rsync to download
    eval $RSYNC_CMD -avz --progress \
        "$USERNAME@$SERVER_ADDRESS:~/llm-stylometry/models_hf/${author}_tokenizer=gpt2/" \
        "models_hf/${author}_tokenizer=gpt2/"

    if [ $? -eq 0 ]; then
        print_success "$author model synced"
        ((SYNCED_COUNT++))
    else
        print_error "Failed to sync $author model"
    fi
done

# Verify synced models
echo
print_info "Verifying synced models..."

for author in "${AUTHORS_TO_SYNC[@]}"; do
    MODEL_DIR="models_hf/${author}_tokenizer=gpt2"

    if [ -f "$MODEL_DIR/model.safetensors" ] && [ -f "$MODEL_DIR/config.json" ]; then
        print_success "Verified: $author"
    else
        print_warning "Incomplete: $author (missing files)"
    fi
done

# Summary
echo
echo "=================================================="
echo "              Sync Complete"
echo "=================================================="
echo "✓ Synced: $SYNCED_COUNT models"
echo
echo "Models available in: models_hf/"
echo
echo "Next steps:"
echo "  1. Verify model quality (generate text samples)"
echo "  2. Upload to HuggingFace: ./upload_to_huggingface.sh"
