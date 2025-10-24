#!/bin/bash

# Train HuggingFace Models for Public Release
# This script trains high-quality models (one per author) to low loss for HF deployment

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
TRAIN_AUTHOR=""
TRAIN_ALL=false
TARGET_LOSS=0.1
OUTPUT_DIR="models_hf"
MAX_EPOCHS=50000

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
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
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --max-epochs)
            MAX_EPOCHS="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Train high-quality models for HuggingFace deployment"
            echo ""
            echo "Options:"
            echo "  --author NAME       Train single author"
            echo "  --all               Train all 8 authors (sequentially)"
            echo "  --target-loss LOSS  Target training loss (default: 0.1)"
            echo "  --output-dir DIR    Output directory (default: models_hf)"
            echo "  --max-epochs N      Maximum epochs (default: 50000)"
            echo "  -h, --help          Show this help"
            echo ""
            echo "Examples:"
            echo "  $0 --author baum                # Train Baum model to loss 0.1"
            echo "  $0 --all --target-loss 0.05     # Train all to loss 0.05"
            echo "  $0 --author austen              # Train Austen model"
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Validate arguments
if [ "$TRAIN_ALL" = false ] && [ -z "$TRAIN_AUTHOR" ]; then
    print_error "Must specify --author or --all"
    echo "Use --help for usage information"
    exit 1
fi

echo "=================================================="
echo "   HuggingFace Model Training"
echo "=================================================="
echo
print_info "Target loss: $TARGET_LOSS"
print_info "Output directory: $OUTPUT_DIR"
print_info "Max epochs: $MAX_EPOCHS"
echo

# Activate conda environment
if ! command -v conda &> /dev/null; then
    print_error "conda not found. Please install Miniconda or Anaconda."
    exit 1
fi

print_info "Activating conda environment..."
eval "$(conda shell.bash hook)" 2>/dev/null || {
    print_error "Failed to initialize conda"
    exit 1
}

conda activate llm-stylometry 2>/dev/null || {
    print_error "Failed to activate llm-stylometry environment"
    print_info "Run: ./run_llm_stylometry.sh (to set up environment)"
    exit 1
}

# Build author list
if [ "$TRAIN_ALL" = true ]; then
    AUTHORS="austen baum dickens fitzgerald melville thompson twain wells"
    print_info "Training all 8 authors in parallel (one per GPU)"
else
    AUTHORS="$TRAIN_AUTHOR"
    print_info "Training single author: $TRAIN_AUTHOR"
fi

echo

# Train each author
TRAINED_COUNT=0
FAILED_COUNT=0

if [ "$TRAIN_ALL" = true ]; then
    # Parallel training: start all authors in background
    PIDS=()
    for author in $AUTHORS; do
        print_info "Starting $author in background..."

        python code/train_hf_model.py \
            --author "$author" \
            --target-loss "$TARGET_LOSS" \
            --output-dir "$OUTPUT_DIR" \
            --max-epochs "$MAX_EPOCHS" \
            > logs/hf_${author}.log 2>&1 &

        PIDS+=($!)
        sleep 2  # Small delay to stagger GPU assignment
    done

    # Wait for all to complete
    print_info "All 8 authors started. Waiting for completion..."
    for i in "${!PIDS[@]}"; do
        author=$(echo "$AUTHORS" | cut -d' ' -f$((i+1)))
        if wait ${PIDS[$i]}; then
            print_success "Completed: $author"
            ((TRAINED_COUNT++))
        else
            print_error "Failed: $author"
            ((FAILED_COUNT++))
        fi
    done
else
    # Sequential training for single author
    for author in $AUTHORS; do
        print_info "Training $author..."

        if python code/train_hf_model.py \
            --author "$author" \
            --target-loss "$TARGET_LOSS" \
            --output-dir "$OUTPUT_DIR" \
            --max-epochs "$MAX_EPOCHS"; then

            print_success "Completed: $author"
            ((TRAINED_COUNT++))
        else
            print_error "Failed: $author"
            ((FAILED_COUNT++))
        fi

        echo
    done
fi

# Summary
echo "=================================================="
echo "                  Summary"
echo "=================================================="
echo "✓ Models trained: $TRAINED_COUNT"
if [ "$FAILED_COUNT" -gt 0 ]; then
    echo "✗ Failed: $FAILED_COUNT"
fi
echo
echo "Models saved to: $OUTPUT_DIR"
echo

if [ "$TRAINED_COUNT" -gt 0 ]; then
    print_success "Training complete!"
    echo
    echo "Next steps:"
    echo "1. Verify model quality: python -c 'from transformers import GPT2LMHeadModel; ...'"
    echo "2. Generate text samples to check quality"
    echo "3. Upload to HuggingFace: ./upload_to_huggingface.sh"
    exit 0
else
    print_error "No models were trained successfully"
    exit 1
fi
