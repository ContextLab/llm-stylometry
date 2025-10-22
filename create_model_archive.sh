#!/bin/bash

# Create Model Archive Script for LLM Stylometry
# This script creates compressed archives of model weights for Dropbox distribution

set -e

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
print_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
print_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
print_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Default: nothing selected (will default to all if nothing specified)
CREATE_BASELINE=false
CREATE_CONTENT=false
CREATE_FUNCTION=false
CREATE_POS=false
FORCE_OVERWRITE=false
OUTPUT_DIR="."

# Parse command line arguments (stackable)
while [[ $# -gt 0 ]]; do
    case $1 in
        -b|--baseline)
            CREATE_BASELINE=true
            shift
            ;;
        -co|--content-only)
            CREATE_CONTENT=true
            shift
            ;;
        -fo|--function-only)
            CREATE_FUNCTION=true
            shift
            ;;
        -pos|--part-of-speech)
            CREATE_POS=true
            shift
            ;;
        -a|--all)
            CREATE_BASELINE=true
            CREATE_CONTENT=true
            CREATE_FUNCTION=true
            CREATE_POS=true
            shift
            ;;
        -f|--force)
            FORCE_OVERWRITE=true
            shift
            ;;
        -o|--output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Create compressed archives of model weights for distribution."
            echo ""
            echo "Options:"
            echo "  -b, --baseline          Create baseline model archive"
            echo "  -co, --content-only     Create content-only variant archive"
            echo "  -fo, --function-only    Create function-only variant archive"
            echo "  -pos, --part-of-speech  Create part-of-speech variant archive"
            echo "  -a, --all               Create all archives (default if none specified)"
            echo "  -f, --force             Overwrite existing archives without prompting"
            echo "  -o, --output-dir DIR    Output directory (default: current directory)"
            echo "  -h, --help              Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0                      # Create all archives"
            echo "  $0 -b                   # Create baseline archive only"
            echo "  $0 -co -fo              # Create content and function archives"
            echo "  $0 -a -o /tmp           # Create all archives in /tmp"
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            echo "Use -h or --help for usage information"
            exit 1
            ;;
    esac
done

# If nothing was selected, default to all
if [ "$CREATE_BASELINE" = false ] && [ "$CREATE_CONTENT" = false ] && [ "$CREATE_FUNCTION" = false ] && [ "$CREATE_POS" = false ]; then
    CREATE_BASELINE=true
    CREATE_CONTENT=true
    CREATE_FUNCTION=true
    CREATE_POS=true
fi

# Verify we're in the project root
if [ ! -d "models" ]; then
    print_error "models/ directory not found. Please run this script from the project root."
    exit 1
fi

# Create output directory if needed
mkdir -p "$OUTPUT_DIR"

echo "=================================================="
echo "       LLM Stylometry Model Archive Creator"
echo "=================================================="
echo
print_info "Output directory: $OUTPUT_DIR"
echo
echo "Archive configuration:"
[ "$CREATE_BASELINE" = true ] && echo "  ✓ Baseline models"
[ "$CREATE_CONTENT" = true ] && echo "  ✓ Content-only variant"
[ "$CREATE_FUNCTION" = true ] && echo "  ✓ Function-only variant"
[ "$CREATE_POS" = true ] && echo "  ✓ Part-of-speech variant"
echo

# Function to check disk space
check_disk_space() {
    local required_gb=$1
    local required_bytes=$((required_gb * 1024 * 1024 * 1024))

    # Get available space on output directory's filesystem
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        local available=$(df -k "$OUTPUT_DIR" | tail -1 | awk '{print $4}')
        available=$((available * 1024))  # Convert to bytes
    else
        # Linux
        local available=$(df --output=avail -B 1 "$OUTPUT_DIR" | tail -1)
    fi

    if [ "$available" -lt "$required_bytes" ]; then
        print_error "Insufficient disk space. Required: ${required_gb}GB, Available: $((available / 1024 / 1024 / 1024))GB"
        return 1
    fi

    return 0
}

# Function to find models matching a pattern
find_models() {
    local variant=$1

    if [ -z "$variant" ]; then
        # Baseline models (no variant in name)
        find models/ -maxdepth 1 -type d -name "*_tokenizer=gpt2_seed=*" ! -name "*variant=*" | sort
    else
        # Variant models
        find models/ -maxdepth 1 -type d -name "*variant=${variant}_tokenizer=gpt2_seed=*" | sort
    fi
}

# Function to verify model has required weight files
verify_model_complete() {
    local model_dir=$1

    if [ ! -f "$model_dir/model.safetensors" ]; then
        print_warning "Missing model.safetensors in $model_dir"
        return 1
    fi

    if [ ! -f "$model_dir/training_state.pt" ]; then
        print_warning "Missing training_state.pt in $model_dir"
        return 1
    fi

    return 0
}

# Function to create archive for a variant
create_archive() {
    local variant_name=$1  # e.g., "baseline", "content", "function", "pos"
    local variant_suffix=$2  # e.g., "", "content", "function", "pos"

    local archive_name="model_weights_${variant_name}.tar.gz"
    local archive_path="${OUTPUT_DIR}/${archive_name}"
    local checksum_path="${archive_path}.sha256"

    print_info "Creating ${variant_name} archive..."

    # Check if archive already exists
    if [ -f "$archive_path" ] && [ "$FORCE_OVERWRITE" = false ]; then
        echo
        print_warning "Archive already exists: $archive_path"
        read -p "Overwrite? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            print_info "Skipping ${variant_name} archive"
            return 0
        fi
    fi

    # Find all models for this variant
    local models
    if [ -z "$variant_suffix" ]; then
        models=$(find_models "")
    else
        models=$(find_models "$variant_suffix")
    fi

    local model_count=$(echo "$models" | wc -l | tr -d ' ')

    if [ -z "$models" ] || [ "$model_count" -eq 0 ]; then
        print_error "No models found for variant: $variant_name"
        return 1
    fi

    print_info "Found $model_count model directories"

    # Verify all models are complete
    local complete_count=0
    local incomplete_models=()

    while IFS= read -r model_dir; do
        if verify_model_complete "$model_dir"; then
            ((complete_count++))
        else
            incomplete_models+=("$model_dir")
        fi
    done <<< "$models"

    if [ "$complete_count" -eq 0 ]; then
        print_error "No complete models found (all missing weight files)"
        return 1
    fi

    if [ ${#incomplete_models[@]} -gt 0 ]; then
        print_warning "Found ${#incomplete_models[@]} incomplete models (will be skipped):"
        for model in "${incomplete_models[@]}"; do
            echo "  - $model"
        done
    fi

    print_info "Creating archive with $complete_count complete models..."

    # Check disk space (estimate 7GB per archive)
    if ! check_disk_space 8; then
        return 1
    fi

    # Create temporary file list
    local temp_filelist=$(mktemp)

    while IFS= read -r model_dir; do
        if verify_model_complete "$model_dir"; then
            # Add only the weight files (not config files which are in git)
            echo "$model_dir/model.safetensors" >> "$temp_filelist"
            echo "$model_dir/training_state.pt" >> "$temp_filelist"
        fi
    done <<< "$models"

    # Create archive with progress
    print_info "Compressing files (this may take several minutes)..."

    if tar -czf "$archive_path" -T "$temp_filelist" 2>&1; then
        print_success "Archive created: $archive_path"
    else
        print_error "Failed to create archive"
        rm -f "$temp_filelist"
        return 1
    fi

    # Clean up temp file
    rm -f "$temp_filelist"

    # Get archive size
    local archive_size
    if [[ "$OSTYPE" == "darwin"* ]]; then
        archive_size=$(ls -lh "$archive_path" | awk '{print $5}')
    else
        archive_size=$(du -h "$archive_path" | cut -f1)
    fi

    print_info "Archive size: $archive_size"

    # Verify archive integrity
    print_info "Verifying archive integrity..."
    if tar -tzf "$archive_path" > /dev/null 2>&1; then
        print_success "Archive integrity verified"
    else
        print_error "Archive is corrupted!"
        return 1
    fi

    # Count files in archive
    local file_count=$(tar -tzf "$archive_path" | wc -l | tr -d ' ')
    local expected_count=$((complete_count * 2))  # 2 files per model

    if [ "$file_count" -ne "$expected_count" ]; then
        print_warning "File count mismatch: expected $expected_count, got $file_count"
    else
        print_success "Archive contains $file_count files ($complete_count models)"
    fi

    # Generate SHA256 checksum
    print_info "Generating SHA256 checksum..."

    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        (cd "$OUTPUT_DIR" && shasum -a 256 "$archive_name" > "${archive_name}.sha256")
    else
        # Linux
        (cd "$OUTPUT_DIR" && sha256sum "$archive_name" > "${archive_name}.sha256")
    fi

    if [ -f "$checksum_path" ]; then
        print_success "Checksum saved: $checksum_path"
        cat "$checksum_path"
    else
        print_error "Failed to generate checksum"
        return 1
    fi

    echo
    return 0
}

# Create requested archives
CREATED_COUNT=0
FAILED_COUNT=0

if [ "$CREATE_BASELINE" = true ]; then
    if create_archive "baseline" ""; then
        ((CREATED_COUNT++))
    else
        ((FAILED_COUNT++))
    fi
fi

if [ "$CREATE_CONTENT" = true ]; then
    if create_archive "content" "content"; then
        ((CREATED_COUNT++))
    else
        ((FAILED_COUNT++))
    fi
fi

if [ "$CREATE_FUNCTION" = true ]; then
    if create_archive "function" "function"; then
        ((CREATED_COUNT++))
    else
        ((FAILED_COUNT++))
    fi
fi

if [ "$CREATE_POS" = true ]; then
    if create_archive "pos" "pos"; then
        ((CREATED_COUNT++))
    else
        ((FAILED_COUNT++))
    fi
fi

# Print summary
echo "=================================================="
echo "                   Summary"
echo "=================================================="
echo "✓ Archives created: $CREATED_COUNT"
if [ "$FAILED_COUNT" -gt 0 ]; then
    echo "✗ Failed: $FAILED_COUNT"
fi
echo
echo "Archives saved to: $OUTPUT_DIR"
echo

if [ "$CREATED_COUNT" -gt 0 ]; then
    print_success "Archive creation complete!"
    echo
    echo "Next steps:"
    echo "1. Upload archives and .sha256 files to Dropbox"
    echo "2. Generate shareable links with dl=1 parameter"
    echo "3. Update download_model_weights.sh with URLs"
    exit 0
else
    print_error "No archives were created"
    exit 1
fi
