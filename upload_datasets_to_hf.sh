#!/bin/bash

# Upload Author Datasets to HuggingFace
# Upload cleaned text corpora to public HuggingFace dataset repositories

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
            echo "Upload author text datasets to HuggingFace"
            echo ""
            echo "Options:"
            echo "  --author NAME       Upload single author corpus"
            echo "  --all               Upload all 8 author corpora"
            echo "  --dry-run           Generate cards without uploading"
            echo "  -h, --help          Show this help"
            echo ""
            echo "Examples:"
            echo "  $0 --author baum --dry-run      # Test card generation"
            echo "  $0 --author baum                # Upload Baum corpus"
            echo "  $0 --all                        # Upload all 8 corpora"
            echo ""
            echo "Prerequisites:"
            echo "  - Credentials: .huggingface/credentials.json"
            echo "  - Format: {\"username\": \"contextlab\", \"token\": \"hf_...\"}"
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
echo "     HuggingFace Dataset Upload"
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

# Check if huggingface_hub is installed
if ! python -c "import huggingface_hub" 2>/dev/null; then
    print_warning "huggingface_hub not installed. Installing..."
    pip install huggingface_hub
fi

# Build author list
if [ "$UPLOAD_ALL" = true ]; then
    # Find all authors with data
    AUTHORS=()
    for data_dir in data/cleaned/*; do
        if [ -d "$data_dir" ]; then
            author=$(basename "$data_dir")
            # Skip variant directories
            if [[ ! "$author" =~ _only$ ]] && [[ "$author" != "contested" ]] && [[ "$author" != "non_oz_baum" ]] && [[ "$author" != "non_oz_thompson" ]]; then
                if ls "$data_dir"/*.txt &>/dev/null; then
                    AUTHORS+=("$author")
                fi
            fi
        fi
    done

    print_info "Found ${#AUTHORS[@]} author corpora: ${AUTHORS[*]}"
else
    # Verify single author exists
    DATA_DIR="data/cleaned/$UPLOAD_AUTHOR"
    if [ ! -d "$DATA_DIR" ]; then
        print_error "Data directory not found: $DATA_DIR"
        exit 1
    fi

    if ! ls "$DATA_DIR"/*.txt &>/dev/null; then
        print_error "No text files found in: $DATA_DIR"
        exit 1
    fi

    AUTHORS=("$UPLOAD_AUTHOR")
    print_info "Found corpus: $UPLOAD_AUTHOR"
fi

echo

# Upload each author
UPLOADED_COUNT=0
FAILED_COUNT=0

for author in "${AUTHORS[@]}"; do
    print_info "Processing $author..."

    DATA_DIR="data/cleaned/$author"

    # Generate dataset card
    print_info "  Generating dataset card..."
    if python code/generate_dataset_card.py --author "$author" --data-dir "$DATA_DIR"; then
        print_success "  Dataset card generated"
    else
        print_error "  Failed to generate dataset card"
        ((FAILED_COUNT++))
        continue
    fi

    if [ "$DRY_RUN" = true ]; then
        print_warning "  [DRY RUN] Skipping upload"
        print_info "  Dataset card preview: $DATA_DIR/README.md"

        # Show what would be uploaded
        FILE_COUNT=$(ls "$DATA_DIR"/*.txt 2>/dev/null | wc -l)
        TOTAL_SIZE=$(du -sh "$DATA_DIR" | cut -f1)
        print_info "  Would upload: $FILE_COUNT files, $TOTAL_SIZE total"

        # Clean up generated README
        rm -f "$DATA_DIR/README.md"

        ((UPLOADED_COUNT++))
        continue
    fi

    # Upload to HuggingFace
    print_info "  Uploading to HuggingFace..."

    python3 << ENDPYTHON
import json
from pathlib import Path
from huggingface_hub import HfApi, create_repo

# Load credentials
with open('.huggingface/credentials.json') as f:
    creds = json.load(f)

api = HfApi(token=creds['token'])

# Create or update repo
repo_id = f"contextlab/$author-corpus"
data_dir = "$DATA_DIR"

try:
    # Create dataset repo (public)
    create_repo(
        repo_id,
        repo_type="dataset",
        exist_ok=True,
        private=False,
        token=creds['token']
    )
    print(f"  Repository ready: {repo_id}")

    # Upload entire directory (includes README.md and all .txt files)
    api.upload_folder(
        folder_path=data_dir,
        repo_id=repo_id,
        repo_type="dataset",
        commit_message="Upload $author complete works corpus",
        ignore_patterns=["*.pyc", "__pycache__", ".DS_Store"]
    )

    print(f"  Upload complete: https://huggingface.co/datasets/{repo_id}")

except Exception as e:
    print(f"  ERROR: Upload failed: {e}")
    exit(1)
ENDPYTHON

    if [ $? -eq 0 ]; then
        print_success "  Uploaded: $author"

        # Clean up generated README from local directory
        rm -f "$DATA_DIR/README.md"

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
        print_success "Dry run complete! Dataset cards generated."
        echo "Review cards and run without --dry-run to upload"
    else
        print_success "Upload complete!"
        echo "View datasets at:"
        for author in "${AUTHORS[@]}"; do
            echo "  https://huggingface.co/datasets/contextlab/$author-corpus"
        done
    fi
    exit 0
else
    print_error "No datasets were uploaded"
    exit 1
fi
