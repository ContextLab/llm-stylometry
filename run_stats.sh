#!/bin/bash

# Statistical Analysis Script for LLM Stylometry
# Computes key statistics to reproduce paper results

set -e

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

print_header() {
    echo -e "${CYAN}╔══════════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║           LLM Stylometry Statistical Analysis            ║${NC}"
    echo -e "${CYAN}╚══════════════════════════════════════════════════════════╝${NC}"
    echo
}

print_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
print_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
print_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
print_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Parse command line arguments
VARIANTS=()
DATA_PATH="data/model_results.pkl"

show_help() {
    echo "Usage: $0 [OPTIONS]"
    echo
    echo "Options:"
    echo "  -h, --help              Show this help message"
    echo "  -d, --data PATH         Path to model results file (default: data/model_results.pkl)"
    echo "  -co, --content-only     Compute statistics for content-word variant only"
    echo "  -fo, --function-only    Compute statistics for function-word variant only"
    echo "  -pos, --part-of-speech  Compute statistics for part-of-speech variant only"
    echo "  -a, --all               Compute statistics for baseline + all 3 variants"
    echo
    echo "If no variant flags are specified, computes baseline statistics only."
    echo
    exit 0
}

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            ;;
        -d|--data)
            DATA_PATH="$2"
            shift 2
            ;;
        -co|--content-only)
            VARIANTS+=("content")
            shift
            ;;
        -fo|--function-only)
            VARIANTS+=("function")
            shift
            ;;
        -pos|--part-of-speech)
            VARIANTS+=("pos")
            shift
            ;;
        -a|--all)
            VARIANTS=("baseline" "content" "function" "pos")
            shift
            ;;
        *)
            print_error "Unknown option: $1"
            echo "Use -h or --help for usage information"
            exit 1
            ;;
    esac
done

# If no variants specified, default to baseline only
if [ ${#VARIANTS[@]} -eq 0 ]; then
    VARIANTS=("baseline")
fi

# Check if script is being run from correct directory
if [ ! -f "run_llm_stylometry.sh" ]; then
    print_error "This script must be run from the root of the llm-stylometry repository"
    exit 1
fi

print_header

# Check if data exists
if [ ! -f "$DATA_PATH" ]; then
    print_error "Model results not found at $DATA_PATH"
    print_info "Please run './run_llm_stylometry.sh' first to generate results"
    exit 1
fi

# Set up conda environment
print_info "Setting up conda environment..."
if ! conda info --envs | grep -q "llm-stylometry"; then
    print_warning "Environment 'llm-stylometry' not found, creating..."
    conda create -n llm-stylometry python=3.11 -y
fi

print_info "Activating environment..."
eval "$(conda shell.bash hook)"
conda activate llm-stylometry

# Check if required packages are installed
print_info "Checking dependencies..."
python -c "import scipy, pandas, numpy" 2>/dev/null || {
    print_warning "Installing required packages..."
    pip install scipy pandas numpy
}

# Run the statistical analysis for each variant
for variant in "${VARIANTS[@]}"; do
    echo
    if [ "$variant" == "baseline" ]; then
        print_info "Computing baseline statistics..."
        python code/compute_stats.py --data "$DATA_PATH"
    else
        print_info "Computing statistics for $variant variant..."
        python code/compute_stats.py --data "$DATA_PATH" --variant "$variant"
    fi
    echo
done

print_success "Statistical analysis complete!"