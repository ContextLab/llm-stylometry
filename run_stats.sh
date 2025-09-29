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

# Check if script is being run from correct directory
if [ ! -f "run_llm_stylometry.sh" ]; then
    print_error "This script must be run from the root of the llm-stylometry repository"
    exit 1
fi

print_header

# Check if data exists
if [ ! -f "data/model_results.pkl" ]; then
    print_error "Model results not found at data/model_results.pkl"
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

# Run the statistical analysis
print_info "Computing statistics..."
echo

python code/compute_stats.py

print_success "Statistical analysis complete!"