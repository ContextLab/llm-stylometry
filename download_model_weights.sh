#!/bin/bash

# Model Download Script for LLM Stylometry
# This script downloads pre-trained model weights from Dropbox

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

# Dropbox URLs (updated 2025-10-22)
# NOTE: URLs modified with dl=1 parameter for direct download
BASELINE_URL="https://www.dropbox.com/scl/fi/ol2et7qyz6yps0frl6kzs/model_weights_baseline.tar.gz?rlkey=t58xh437hfej4zx0w8vuj4cu0&dl=1"
BASELINE_SHA_URL=""  # TODO: Add .sha256 checksum file URL
CONTENT_URL="https://www.dropbox.com/scl/fi/pe78lgs617p29caisexhk/model_weights_content.tar.gz?rlkey=j6casahg68kj8u1wq203km5yd&dl=1"
CONTENT_SHA_URL=""  # TODO: Add .sha256 checksum file URL
FUNCTION_URL="https://www.dropbox.com/scl/fi/klncxsb41jzxpn9xfvvow/model_weights_function.tar.gz?rlkey=l5chc9g5b93ynacaxedki7j6x&dl=1"
FUNCTION_SHA_URL=""  # TODO: Add .sha256 checksum file URL
POS_URL="https://www.dropbox.com/scl/fi/y3vr91jt24dwb5qw2mbr7/model_weights_pos.tar.gz?rlkey=xntev319gcakmc2zjw5fv48iu&dl=1"
POS_SHA_URL=""  # TODO: Add .sha256 checksum file URL

# Default: nothing selected (will default to all if nothing specified)
DOWNLOAD_BASELINE=false
DOWNLOAD_CONTENT=false
DOWNLOAD_FUNCTION=false
DOWNLOAD_POS=false
KEEP_ARCHIVES=false
SKIP_CHECKSUM=false

# Parse command line arguments (stackable)
while [[ $# -gt 0 ]]; do
    case $1 in
        -b|--baseline)
            DOWNLOAD_BASELINE=true
            shift
            ;;
        -co|--content-only)
            DOWNLOAD_CONTENT=true
            shift
            ;;
        -fo|--function-only)
            DOWNLOAD_FUNCTION=true
            shift
            ;;
        -pos|--part-of-speech)
            DOWNLOAD_POS=true
            shift
            ;;
        -a|--all)
            DOWNLOAD_BASELINE=true
            DOWNLOAD_CONTENT=true
            DOWNLOAD_FUNCTION=true
            DOWNLOAD_POS=true
            shift
            ;;
        --keep-archives)
            KEEP_ARCHIVES=true
            shift
            ;;
        --skip-checksum)
            SKIP_CHECKSUM=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Download pre-trained model weights from Dropbox."
            echo ""
            echo "Options:"
            echo "  -b, --baseline          Download baseline models (~6.7GB)"
            echo "  -co, --content-only     Download content-only variant (~6.7GB)"
            echo "  -fo, --function-only    Download function-only variant (~6.7GB)"
            echo "  -pos, --part-of-speech  Download part-of-speech variant (~6.7GB)"
            echo "  -a, --all               Download all variants (~27GB total)"
            echo "  --keep-archives         Keep downloaded tar.gz files after extraction"
            echo "  --skip-checksum         Skip SHA256 verification (not recommended)"
            echo "  -h, --help              Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0                      # Download all variants"
            echo "  $0 -b                   # Download baseline only"
            echo "  $0 -co -fo              # Download content and function variants"
            echo "  $0 -a --keep-archives   # Download all, keep archives"
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
if [ "$DOWNLOAD_BASELINE" = false ] && [ "$DOWNLOAD_CONTENT" = false ] && [ "$DOWNLOAD_FUNCTION" = false ] && [ "$DOWNLOAD_POS" = false ]; then
    DOWNLOAD_BASELINE=true
    DOWNLOAD_CONTENT=true
    DOWNLOAD_FUNCTION=true
    DOWNLOAD_POS=true
fi

# Verify we're in the project root
if [ ! -d "models" ]; then
    print_warning "models/ directory not found. Creating it..."
    mkdir -p models
fi

echo "=================================================="
echo "     LLM Stylometry Model Weight Downloader"
echo "=================================================="
echo
echo "Download configuration:"
[ "$DOWNLOAD_BASELINE" = true ] && echo "  ✓ Baseline models (~6.7GB)"
[ "$DOWNLOAD_CONTENT" = true ] && echo "  ✓ Content-only variant (~6.7GB)"
[ "$DOWNLOAD_FUNCTION" = true ] && echo "  ✓ Function-only variant (~6.7GB)"
[ "$DOWNLOAD_POS" = true ] && echo "  ✓ Part-of-speech variant (~6.7GB)"
echo

# Function to check disk space
check_disk_space() {
    local required_gb=$1

    # Get available space on current directory's filesystem
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        local available=$(df -k . | tail -1 | awk '{print $4}')
        local available_gb=$((available / 1024 / 1024))
    else
        # Linux
        local available=$(df --output=avail -B 1G . | tail -1)
        local available_gb=$available
    fi

    if [ "$available_gb" -lt "$required_gb" ]; then
        print_error "Insufficient disk space. Required: ${required_gb}GB, Available: ${available_gb}GB"
        return 1
    fi

    print_info "Disk space check: ${available_gb}GB available"
    return 0
}

# Function to count existing models for a variant
count_existing_models() {
    local variant=$1

    if [ -z "$variant" ]; then
        # Baseline models
        find models/ -maxdepth 1 -type d -name "*_tokenizer=gpt2_seed=*" ! -name "*variant=*" \
            -exec test -f "{}/model.safetensors" \; -print 2>/dev/null | wc -l | tr -d ' '
    else
        # Variant models
        find models/ -maxdepth 1 -type d -name "*variant=${variant}_tokenizer=gpt2_seed=*" \
            -exec test -f "{}/model.safetensors" \; -print 2>/dev/null | wc -l | tr -d ' '
    fi
}

# Function to download a file with progress and resume support
download_file() {
    local url=$1
    local output_path=$2
    local description=$3

    print_info "Downloading $description..."

    # Check if URL is set
    if [ -z "$url" ]; then
        print_error "Download URL not configured. Please check that Dropbox URLs have been added to this script."
        return 1
    fi

    # Download with curl (supports resume with -C -)
    if curl -L -C - --progress-bar "$url" -o "$output_path"; then
        print_success "Downloaded: $output_path"
        return 0
    else
        print_error "Download failed for: $description"
        return 1
    fi
}

# Function to verify checksum
verify_checksum() {
    local archive_path=$1
    local checksum_path=$2
    local variant_name=$3

    if [ "$SKIP_CHECKSUM" = true ]; then
        print_warning "Skipping checksum verification (--skip-checksum flag)"
        return 0
    fi

    if [ ! -f "$checksum_path" ]; then
        print_error "Checksum file not found: $checksum_path"
        return 1
    fi

    print_info "Verifying SHA256 checksum for $variant_name..."

    # Compute checksum of downloaded file
    local computed_checksum
    if [[ "$OSTYPE" == "darwin"* ]]; then
        computed_checksum=$(shasum -a 256 "$archive_path" | awk '{print $1}')
    else
        computed_checksum=$(sha256sum "$archive_path" | awk '{print $1}')
    fi

    # Read expected checksum from file
    local expected_checksum=$(awk '{print $1}' "$checksum_path")

    if [ "$computed_checksum" = "$expected_checksum" ]; then
        print_success "Checksum verified: $variant_name"
        return 0
    else
        print_error "Checksum mismatch for $variant_name!"
        echo "  Expected: $expected_checksum"
        echo "  Computed: $computed_checksum"
        echo "  The download may be corrupted. Please delete the archive and try again."
        return 1
    fi
}

# Function to extract archive
extract_archive() {
    local archive_path=$1
    local variant_name=$2

    print_info "Extracting $variant_name models..."

    # Verify archive is valid before extracting
    if ! tar -tzf "$archive_path" > /dev/null 2>&1; then
        print_error "Archive is corrupted: $archive_path"
        return 1
    fi

    # Extract (overwrites existing files)
    if tar -xzf "$archive_path" 2>&1; then
        print_success "Extraction complete: $variant_name"
        return 0
    else
        print_error "Extraction failed: $variant_name"
        return 1
    fi
}

# Function to verify extracted models
verify_extraction() {
    local variant=$1
    local variant_name=$2
    local expected_count=80

    local actual_count
    if [ -z "$variant" ]; then
        # Baseline models
        actual_count=$(find models/ -maxdepth 1 -type d -name "*_tokenizer=gpt2_seed=*" ! -name "*variant=*" \
            -exec test -f "{}/model.safetensors" -a -f "{}/training_state.pt" \; -print 2>/dev/null | wc -l | tr -d ' ')
    else
        # Variant models
        actual_count=$(find models/ -maxdepth 1 -type d -name "*variant=${variant}_tokenizer=gpt2_seed=*" \
            -exec test -f "{}/model.safetensors" -a -f "{}/training_state.pt" \; -print 2>/dev/null | wc -l | tr -d ' ')
    fi

    if [ "$actual_count" -eq "$expected_count" ]; then
        print_success "Verified $actual_count/$expected_count $variant_name models"
        return 0
    else
        print_warning "Found $actual_count/$expected_count $variant_name models (some may be incomplete)"
        return 1
    fi
}

# Function to download and install a variant
download_variant() {
    local variant_name=$1
    local variant_suffix=$2
    local archive_url=$3
    local checksum_url=$4

    echo
    print_info "Processing $variant_name variant..."

    # Check existing models
    local existing_count=$(count_existing_models "$variant_suffix")

    if [ "$existing_count" -eq 80 ]; then
        echo
        print_warning "All 80 $variant_name models already exist locally."
        read -p "Re-download and overwrite? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            print_info "Skipping $variant_name variant"
            return 0
        fi
    elif [ "$existing_count" -gt 0 ]; then
        print_info "Found $existing_count existing $variant_name models (will merge with downloaded models)"
    fi

    # Set up file paths
    local archive_name="model_weights_${variant_name}.tar.gz"
    local archive_path="/tmp/${archive_name}"
    local checksum_path="/tmp/${archive_name}.sha256"

    # Download archive
    if ! download_file "$archive_url" "$archive_path" "$variant_name archive"; then
        return 1
    fi

    # Download checksum
    if [ "$SKIP_CHECKSUM" = false ]; then
        if ! download_file "$checksum_url" "$checksum_path" "$variant_name checksum"; then
            return 1
        fi

        # Verify checksum
        if ! verify_checksum "$archive_path" "$checksum_path" "$variant_name"; then
            return 1
        fi
    fi

    # Extract archive
    if ! extract_archive "$archive_path" "$variant_name"; then
        return 1
    fi

    # Verify extraction
    verify_extraction "$variant_suffix" "$variant_name"

    # Clean up downloaded files unless --keep-archives flag is set
    if [ "$KEEP_ARCHIVES" = false ]; then
        print_info "Cleaning up downloaded archives..."
        rm -f "$archive_path" "$checksum_path"
        print_success "Cleanup complete"
    else
        print_info "Keeping archives in /tmp/ (--keep-archives flag)"
    fi

    echo
    return 0
}

# Estimate total download size
TOTAL_SIZE_GB=0
[ "$DOWNLOAD_BASELINE" = true ] && TOTAL_SIZE_GB=$((TOTAL_SIZE_GB + 7))
[ "$DOWNLOAD_CONTENT" = true ] && TOTAL_SIZE_GB=$((TOTAL_SIZE_GB + 7))
[ "$DOWNLOAD_FUNCTION" = true ] && TOTAL_SIZE_GB=$((TOTAL_SIZE_GB + 7))
[ "$DOWNLOAD_POS" = true ] && TOTAL_SIZE_GB=$((TOTAL_SIZE_GB + 7))

print_info "Estimated download size: ~${TOTAL_SIZE_GB}GB"

# Check disk space (need space for archives + extracted files)
if ! check_disk_space $((TOTAL_SIZE_GB * 2)); then
    exit 1
fi

echo
read -p "Continue with download? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    print_info "Download cancelled"
    exit 0
fi

# Download requested variants
DOWNLOADED_COUNT=0
FAILED_COUNT=0

if [ "$DOWNLOAD_BASELINE" = true ]; then
    if download_variant "baseline" "" "$BASELINE_URL" "$BASELINE_SHA_URL"; then
        ((DOWNLOADED_COUNT++))
    else
        ((FAILED_COUNT++))
    fi
fi

if [ "$DOWNLOAD_CONTENT" = true ]; then
    if download_variant "content" "content" "$CONTENT_URL" "$CONTENT_SHA_URL"; then
        ((DOWNLOADED_COUNT++))
    else
        ((FAILED_COUNT++))
    fi
fi

if [ "$DOWNLOAD_FUNCTION" = true ]; then
    if download_variant "function" "function" "$FUNCTION_URL" "$FUNCTION_SHA_URL"; then
        ((DOWNLOADED_COUNT++))
    else
        ((FAILED_COUNT++))
    fi
fi

if [ "$DOWNLOAD_POS" = true ]; then
    if download_variant "pos" "pos" "$POS_URL" "$POS_SHA_URL"; then
        ((DOWNLOADED_COUNT++))
    else
        ((FAILED_COUNT++))
    fi
fi

# Print summary
echo "=================================================="
echo "                   Summary"
echo "=================================================="
echo "✓ Variants downloaded: $DOWNLOADED_COUNT"
if [ "$FAILED_COUNT" -gt 0 ]; then
    echo "✗ Failed: $FAILED_COUNT"
fi
echo

if [ "$DOWNLOADED_COUNT" -gt 0 ]; then
    print_success "Model weights download complete!"
    echo
    echo "You can now:"
    echo "  - Generate all figures: ./run_llm_stylometry.sh"
    echo "  - Load models in Python: from transformers import GPT2LMHeadModel"
    echo "  - Resume training: ./run_llm_stylometry.sh --train --resume"
    exit 0
else
    print_error "No model weights were downloaded"
    exit 1
fi
