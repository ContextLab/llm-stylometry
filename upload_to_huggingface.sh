#!/bin/bash

# Upload Models to HuggingFace
# Upload trained HF models to public HuggingFace repositories

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
UPLOAD_AUTHOR=""
UPLOAD_ALL=false
DRY_RUN=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --author)
            UPLOAD_AUTHOR="$2"
            shift 2
            ;;
        --all)
            UPLOAD_ALL=true
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Upload HuggingFace models to public repositories"
            echo ""
            echo "Options:"
            echo "  --author NAME       Upload single author"
            echo "  --all               Upload all completed authors"
            echo "  --dry-run           Generate model cards without uploading"
            echo "  -h, --help          Show this help"
            echo ""
            echo "Examples:"
            echo "  $0 --author baum --dry-run      # Test model card generation"
            echo "  $0 --author baum                # Upload Baum model"
            echo "  $0 --all                        # Upload all completed models"
            echo ""
            echo "Prerequisites:"
            echo "  - Credentials: .huggingface/credentials.json"
            echo "  - Format: {\"username\": \"contextlab\", \"token\": \"hf_...\"}"
            echo "  - Models in models_hf/{author}_tokenizer=gpt2/"
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Validate arguments
if [ "$UPLOAD_ALL" = false ] && [ -z "$UPLOAD_AUTHOR" ]; then
    print_error "Must specify --author or --all"
    exit 1
fi

echo "=================================================="
echo "     HuggingFace Model Upload"
echo "=================================================="
echo

# Check credentials
CRED_FILE=".huggingface/credentials.json"
if [ ! -f "$CRED_FILE" ]; then
    print_error "Credentials file not found: $CRED_FILE"
    echo ""
    echo "Please create credentials file:"
    echo "  mkdir -p .huggingface"
    echo "  echo '{\"username\": \"contextlab\", \"token\": \"hf_...\"}' > $CRED_FILE"
    exit 1
fi

print_info "Loading HuggingFace credentials..."

# Activate conda environment
if ! command -v conda &> /dev/null; then
    print_error "conda not found"
    exit 1
fi

eval "$(conda shell.bash hook)" 2>/dev/null || {
    print_error "Failed to initialize conda"
    exit 1
}

conda activate llm-stylometry 2>/dev/null || {
    print_error "Failed to activate llm-stylometry environment"
    exit 1
}

# Check if huggingface_hub is installed
if ! python -c "import huggingface_hub" 2>/dev/null; then
    print_warning "huggingface_hub not installed. Installing..."
    pip install huggingface_hub
fi

# Build author list
if [ "$UPLOAD_ALL" = true ]; then
    # Find all completed models in models_hf/
    AUTHORS=()
    for model_dir in models_hf/*_tokenizer=gpt2; do
        if [ -d "$model_dir" ] && [ -f "$model_dir/model.safetensors" ]; then
            author=$(basename "$model_dir" | cut -d'_' -f1)
            AUTHORS+=("$author")
        fi
    done

    if [ ${#AUTHORS[@]} -eq 0 ]; then
        print_error "No completed models found in models_hf/"
        exit 1
    fi

    print_info "Found ${#AUTHORS[@]} completed models: ${AUTHORS[*]}"
else
    AUTHORS=("$UPLOAD_AUTHOR")

    # Verify model exists
    MODEL_DIR="models_hf/${UPLOAD_AUTHOR}_tokenizer=gpt2"
    if [ ! -d "$MODEL_DIR" ]; then
        print_error "Model directory not found: $MODEL_DIR"
        exit 1
    fi

    if [ ! -f "$MODEL_DIR/model.safetensors" ]; then
        print_error "Model weights not found: $MODEL_DIR/model.safetensors"
        exit 1
    fi

    print_info "Found model: $UPLOAD_AUTHOR"
fi

echo

# Upload each author
UPLOADED_COUNT=0
FAILED_COUNT=0

for author in "${AUTHORS[@]}"; do
    print_info "Processing $author..."

    MODEL_DIR="models_hf/${author}_tokenizer=gpt2"

    # Generate model card
    print_info "  Generating model card..."
    if python code/generate_model_card.py --author "$author" --model-dir "$MODEL_DIR"; then
        print_success "  Model card generated"
    else
        print_error "  Failed to generate model card"
        ((FAILED_COUNT++))
        continue
    fi

    if [ "$DRY_RUN" = true ]; then
        print_warning "  [DRY RUN] Skipping upload"
        print_info "  Model card preview: $MODEL_DIR/README.md"
        ((UPLOADED_COUNT++))
        continue
    fi

    # Upload to HuggingFace
    print_info "  Uploading to HuggingFace..."

    python3 << ENDPYTHON
import json
from pathlib import Path
from huggingface_hub import HfApi, create_repo, upload_folder

# Load credentials
with open('.huggingface/credentials.json') as f:
    creds = json.load(f)

api = HfApi(token=creds['token'])

# Create or update repo
repo_id = f"contextlab/gpt2-$author"
model_dir = "$MODEL_DIR"

try:
    # Create repo if doesn't exist
    create_repo(repo_id, exist_ok=True, token=creds['token'], repo_type="model", private=False)
    print(f"  Repository ready: {repo_id}")

    # Upload model directory
    api.upload_folder(
        folder_path=model_dir,
        repo_id=repo_id,
        repo_type="model",
        commit_message="Upload $author stylometry model"
    )

    print(f"  Upload complete: https://huggingface.co/{repo_id}")

except Exception as e:
    print(f"  ERROR: Upload failed: {e}")
    exit(1)
ENDPYTHON

    if [ $? -eq 0 ]; then
        print_success "  Uploaded: $author"
        ((UPLOADED_COUNT++))
    else
        print_error "  Failed: $author"
        ((FAILED_COUNT++))
    fi

    echo
done

# Summary
echo "=================================================="
echo "                Summary"
echo "=================================================="
echo "✓ Uploaded: $UPLOADED_COUNT"
if [ "$FAILED_COUNT" -gt 0 ]; then
    echo "✗ Failed: $FAILED_COUNT"
fi
echo

if [ "$UPLOADED_COUNT" -gt 0 ]; then
    if [ "$DRY_RUN" = true ]; then
        print_success "Dry run complete! Model cards generated."
        echo "Review model cards in models_hf/*/README.md"
        echo "Run without --dry-run to upload to HuggingFace"
    else
        print_success "Upload complete!"
        echo "View models at:"
        for author in "${AUTHORS[@]}"; do
            echo "  https://huggingface.co/contextlab/gpt2-$author"
        done
    fi
    exit 0
else
    print_error "No models were uploaded"
    exit 1
fi
