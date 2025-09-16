#!/bin/bash

# LLM Stylometry CLI - Complete setup and execution script
# Usage: ./run_llm_stylometry.sh [options]

set -e  # Exit on error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
CONDA_ENV="llm-stylometry"
PYTHON_VERSION="3.10"
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Function to print colored output
print_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
print_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
print_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
print_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Function to display help
show_help() {
    cat << EOF
LLM Stylometry CLI - Train models and generate figures

Usage: $0 [OPTIONS]

OPTIONS:
    -h, --help              Show this help message
    -f, --figure FIGURE     Generate specific figure (1a, 1b, 2a, 2b, 3, 4, 5)
    -t, --train             Train models from scratch before generating figures
    -d, --data PATH         Path to model_results.pkl (default: data/model_results.pkl)
    -o, --output DIR        Output directory for figures (default: paper/figs/source)
    -l, --list              List available figures
    --setup-only            Only setup environment without generating figures
    --no-setup              Skip environment setup (assume already configured)
    --force-install         Force reinstall of all dependencies

EXAMPLES:
    $0                      # Setup environment and generate all figures
    $0 -f 1a                # Generate only Figure 1A
    $0 -f 4                 # Generate only Figure 4 (MDS plot)
    $0 -t                   # Train models from scratch, then generate figures
    $0 -l                   # List available figures
    $0 --setup-only         # Only setup the environment

FIGURES:
    1a - Figure 1A: Training curves (all_losses.pdf)
    1b - Figure 1B: Strip plot (stripplot.pdf)
    2a - Figure 2A: Individual t-tests (t_test.pdf)
    2b - Figure 2B: Average t-test (t_test_avg.pdf)
    3  - Figure 3: Confusion matrix heatmap (average_loss_heatmap.pdf)
    4  - Figure 4: 3D MDS plot (3d_MDS_plot.pdf)
    5  - Figure 5: Oz authorship analysis (oz_losses.pdf)

EOF
}

# Function to detect OS
detect_os() {
    case "$OSTYPE" in
        linux*)   echo "linux" ;;
        darwin*)  echo "macos" ;;
        msys*)    echo "windows" ;;
        cygwin*)  echo "windows" ;;
        *)        echo "unknown" ;;
    esac
}

# Function to check if conda is installed
check_conda() {
    if command -v conda &> /dev/null; then
        return 0
    else
        return 1
    fi
}

# Function to install conda
install_conda() {
    print_info "Conda not found. Installing Miniconda..."

    OS=$(detect_os)
    ARCH=$(uname -m)

    case "$OS" in
        linux)
            if [ "$ARCH" = "x86_64" ]; then
                INSTALLER="Miniconda3-latest-Linux-x86_64.sh"
            elif [ "$ARCH" = "aarch64" ]; then
                INSTALLER="Miniconda3-latest-Linux-aarch64.sh"
            else
                print_error "Unsupported architecture: $ARCH"
                exit 1
            fi
            ;;
        macos)
            if [ "$ARCH" = "x86_64" ]; then
                INSTALLER="Miniconda3-latest-MacOSX-x86_64.sh"
            elif [ "$ARCH" = "arm64" ]; then
                INSTALLER="Miniconda3-latest-MacOSX-arm64.sh"
            else
                print_error "Unsupported architecture: $ARCH"
                exit 1
            fi
            ;;
        windows)
            print_error "Please install Anaconda/Miniconda manually on Windows"
            print_info "Visit: https://docs.conda.io/en/latest/miniconda.html"
            exit 1
            ;;
        *)
            print_error "Unsupported OS: $OS"
            exit 1
            ;;
    esac

    # Download and install
    INSTALLER_URL="https://repo.anaconda.com/miniconda/$INSTALLER"
    print_info "Downloading from: $INSTALLER_URL"

    curl -LO "$INSTALLER_URL"
    bash "$INSTALLER" -b -p "$HOME/miniconda3"
    rm "$INSTALLER"

    # Initialize conda
    "$HOME/miniconda3/bin/conda" init bash

    print_success "Miniconda installed successfully"
    print_warning "Please restart your terminal and run this script again"
    exit 0
}

# Function to setup conda environment
setup_environment() {
    if [ "$SKIP_SETUP" = true ]; then
        print_info "Skipping environment setup (--no-setup flag)"
        return 0
    fi

    print_info "Setting up conda environment..."

    # Check if environment exists
    if conda env list | grep -q "^$CONDA_ENV "; then
        print_info "Environment '$CONDA_ENV' already exists"

        if [ "$FORCE_INSTALL" = true ]; then
            print_warning "Force reinstalling dependencies..."
        else
            print_info "Activating environment..."
            eval "$(conda shell.bash hook)"
            conda activate "$CONDA_ENV"
            return 0
        fi
    else
        print_info "Creating conda environment '$CONDA_ENV'..."
        conda create -n "$CONDA_ENV" python="$PYTHON_VERSION" -y
    fi

    # Activate environment
    eval "$(conda shell.bash hook)"
    conda activate "$CONDA_ENV"

    print_info "Installing dependencies..."

    # Install PyTorch with CUDA support
    conda install -c pytorch -c nvidia pytorch pytorch-cuda=12.1 -y 2>/dev/null || \
        conda install -c pytorch pytorch -y

    # Install other dependencies
    pip install --upgrade pip
    pip install "numpy<2" scipy transformers matplotlib seaborn pandas tqdm
    pip install cleantext plotly scikit-learn

    # Install the package
    pip install -e .

    print_success "Environment setup complete"
}

# Parse command line arguments
FIGURE=""
TRAIN=false
DATA_PATH="data/model_results.pkl"
OUTPUT_DIR="paper/figs/source"
LIST_FIGURES=false
SETUP_ONLY=false
SKIP_SETUP=false
FORCE_INSTALL=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        -f|--figure)
            FIGURE="$2"
            shift 2
            ;;
        -t|--train)
            TRAIN=true
            shift
            ;;
        -d|--data)
            DATA_PATH="$2"
            shift 2
            ;;
        -o|--output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -l|--list)
            LIST_FIGURES=true
            shift
            ;;
        --setup-only)
            SETUP_ONLY=true
            shift
            ;;
        --no-setup)
            SKIP_SETUP=true
            shift
            ;;
        --force-install)
            FORCE_INSTALL=true
            shift
            ;;
        *)
            print_error "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Main execution
echo "╔══════════════════════════════════════════════════════════╗"
echo "║                    LLM Stylometry CLI                    ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo

# Check and install conda if needed
if ! check_conda; then
    print_warning "Conda not found"
    read -p "Would you like to install Miniconda? [y/N]: " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        install_conda
    else
        print_error "Conda is required to run this script"
        exit 1
    fi
fi

# Setup environment
setup_environment

# Exit if setup-only
if [ "$SETUP_ONLY" = true ]; then
    print_success "Environment setup complete"
    exit 0
fi

# Build the Python command
PYTHON_CMD="python generate_figures.py"

if [ "$LIST_FIGURES" = true ]; then
    PYTHON_CMD="$PYTHON_CMD --list"
elif [ -n "$FIGURE" ]; then
    PYTHON_CMD="$PYTHON_CMD --figure $FIGURE"
fi

if [ "$TRAIN" = true ]; then
    PYTHON_CMD="$PYTHON_CMD --train"
fi

if [ "$DATA_PATH" != "data/model_results.pkl" ]; then
    PYTHON_CMD="$PYTHON_CMD --data $DATA_PATH"
fi

if [ "$OUTPUT_DIR" != "paper/figs/source" ]; then
    PYTHON_CMD="$PYTHON_CMD --output $OUTPUT_DIR"
fi

# Execute the Python script
print_info "Running: $PYTHON_CMD"
eval $PYTHON_CMD

print_success "Done!"