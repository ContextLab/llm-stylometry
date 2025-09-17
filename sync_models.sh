#!/bin/bash

# Model Sync Script for LLM Stylometry
# This script downloads trained models from a GPU server

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

echo "=================================================="
echo "       LLM Stylometry Model Sync"
echo "=================================================="
echo

# Get server details
read -p "Enter GPU server address (hostname or IP): " SERVER_ADDRESS
if [ -z "$SERVER_ADDRESS" ]; then
    print_error "Server address cannot be empty"
    exit 1
fi

read -p "Enter username for $SERVER_ADDRESS: " USERNAME
if [ -z "$USERNAME" ]; then
    print_error "Username cannot be empty"
    exit 1
fi

print_info "Checking model status on remote server..."

# Check if all models are trained
REMOTE_CHECK=$(ssh "$USERNAME@$SERVER_ADDRESS" 'bash -s' << 'ENDSSH'
#!/bin/bash

MODELS_DIR="$HOME/llm-stylometry/models"
EXPECTED_MODELS=80
AUTHORS=("austen" "baum" "dickens" "fitzgerald" "melville" "thompson" "twain" "wells")

if [ ! -d "$MODELS_DIR" ]; then
    echo "ERROR: Models directory not found"
    exit 1
fi

# Count model directories
MODEL_COUNT=0
MISSING_MODELS=""

for author in "${AUTHORS[@]}"; do
    for seed in {0..9}; do
        MODEL_DIR="$MODELS_DIR/${author}_tokenizer=gpt2_seed=${seed}"
        if [ -d "$MODEL_DIR" ]; then
            # Check for model weights (looking for .pth or .bin files)
            if ls "$MODEL_DIR"/*.pth &>/dev/null || ls "$MODEL_DIR"/*.bin &>/dev/null || ls "$MODEL_DIR"/model.safetensors &>/dev/null; then
                ((MODEL_COUNT++))
            else
                MISSING_MODELS="${MISSING_MODELS}${author}_seed=${seed} (no weights), "
            fi
        else
            MISSING_MODELS="${MISSING_MODELS}${author}_seed=${seed} (no dir), "
        fi
    done
done

echo "MODEL_COUNT=$MODEL_COUNT"
if [ -n "$MISSING_MODELS" ]; then
    echo "MISSING_MODELS=${MISSING_MODELS%, }"
fi

# If all models exist, create tarball
if [ $MODEL_COUNT -eq $EXPECTED_MODELS ]; then
    echo "STATUS=COMPLETE"

    # Create tarball
    cd "$HOME/llm-stylometry"
    TAR_FILE="$HOME/llm_stylometry_models_$(date +%Y%m%d_%H%M%S).tar.gz"
    echo "Creating tarball: $TAR_FILE"
    tar -czf "$TAR_FILE" models/

    # Get file size
    SIZE=$(ls -lh "$TAR_FILE" | awk '{print $5}')
    echo "TAR_FILE=$TAR_FILE"
    echo "TAR_SIZE=$SIZE"
else
    echo "STATUS=INCOMPLETE"
fi
ENDSSH
)

# Parse the remote check results
eval "$REMOTE_CHECK"

if [ "$STATUS" = "ERROR" ]; then
    print_error "Models directory not found on remote server"
    exit 1
elif [ "$STATUS" = "INCOMPLETE" ]; then
    print_warning "Training is not complete!"
    echo
    echo "Found: $MODEL_COUNT / 80 models"
    if [ -n "$MISSING_MODELS" ]; then
        echo "Missing models: $MISSING_MODELS"
    fi
    echo
    echo "Please wait for all models to finish training before syncing."
    exit 1
elif [ "$STATUS" = "COMPLETE" ]; then
    print_success "All 80 models found with weights!"
    echo "Tarball created: $TAR_FILE ($TAR_SIZE)"
    echo
else
    print_error "Unexpected status from remote server"
    exit 1
fi

# Ask for confirmation before downloading
echo "This will download and replace your local models directory."
read -p "Continue? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    print_info "Download cancelled"
    exit 0
fi

# Create local backup if models exist
LOCAL_MODELS_DIR="$HOME/llm-stylometry/models"
if [ -d "$LOCAL_MODELS_DIR" ] && [ "$(ls -A $LOCAL_MODELS_DIR)" ]; then
    print_info "Backing up existing local models..."
    BACKUP_DIR="$HOME/llm-stylometry/models_backup_$(date +%Y%m%d_%H%M%S)"
    mv "$LOCAL_MODELS_DIR" "$BACKUP_DIR"
    print_success "Local models backed up to: $BACKUP_DIR"
fi

# Download the tarball
print_info "Downloading models from remote server..."
LOCAL_TAR="$HOME/$(basename $TAR_FILE)"

# Use rsync for efficient transfer with progress
rsync -avz --progress "$USERNAME@$SERVER_ADDRESS:$TAR_FILE" "$LOCAL_TAR"

if [ ! -f "$LOCAL_TAR" ]; then
    print_error "Download failed"
    exit 1
fi

# Extract the tarball
print_info "Extracting models..."
cd "$HOME/llm-stylometry"
tar -xzf "$LOCAL_TAR"

# Verify extraction
if [ -d "models" ]; then
    MODEL_COUNT=$(find models -maxdepth 1 -type d -name "*_tokenizer=gpt2_seed=*" | wc -l)
    print_success "Successfully extracted $MODEL_COUNT model directories"
else
    print_error "Extraction failed"
    exit 1
fi

# Clean up tarball
rm "$LOCAL_TAR"
print_info "Removed temporary tarball"

# Clean up remote tarball
print_info "Cleaning up remote tarball..."
ssh "$USERNAME@$SERVER_ADDRESS" "rm $TAR_FILE"

# Also download model_results.pkl if it exists
print_info "Checking for consolidated results file..."
RESULTS_EXISTS=$(ssh "$USERNAME@$SERVER_ADDRESS" "[ -f '$HOME/llm-stylometry/data/model_results.pkl' ] && echo 'yes' || echo 'no'")

if [ "$RESULTS_EXISTS" = "yes" ]; then
    print_info "Downloading model_results.pkl..."
    rsync -avz "$USERNAME@$SERVER_ADDRESS:$HOME/llm-stylometry/data/model_results.pkl" \
        "$HOME/llm-stylometry/data/model_results.pkl"
    print_success "Downloaded model_results.pkl"
else
    print_warning "model_results.pkl not found on remote server"
    print_info "You may need to run consolidate_model_results.py locally"
fi

# Print summary
echo
echo "=================================================="
echo "                 Sync Complete!"
echo "=================================================="
echo "✓ Downloaded and extracted $MODEL_COUNT models"
if [ "$RESULTS_EXISTS" = "yes" ]; then
    echo "✓ Downloaded model_results.pkl"
fi
if [ -n "$BACKUP_DIR" ]; then
    echo "✓ Previous models backed up to: $BACKUP_DIR"
fi
echo
echo "Models are now available in: $LOCAL_MODELS_DIR"
echo
echo "You can generate figures with:"
echo "  cd $HOME/llm-stylometry"
echo "  ./run_llm_stylometry.sh"