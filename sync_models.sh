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

# Default: nothing selected (will default to baseline if nothing specified)
SYNC_BASELINE=false
SYNC_CONTENT=false
SYNC_FUNCTION=false
SYNC_POS=false

# Parse command line arguments (stackable)
while [[ $# -gt 0 ]]; do
    case $1 in
        -b|--baseline)
            SYNC_BASELINE=true
            shift
            ;;
        -co|--content-only)
            SYNC_CONTENT=true
            shift
            ;;
        -fo|--function-only)
            SYNC_FUNCTION=true
            shift
            ;;
        -pos|--part-of-speech)
            SYNC_POS=true
            shift
            ;;
        -a|--all)
            SYNC_BASELINE=true
            SYNC_CONTENT=true
            SYNC_FUNCTION=true
            SYNC_POS=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  -b, --baseline          Sync baseline models"
            echo "  -co, --content-only     Sync content-only variant models"
            echo "  -fo, --function-only    Sync function-only variant models"
            echo "  -pos, --part-of-speech  Sync part-of-speech variant models"
            echo "  -a, --all               Sync all models (baseline + all variants)"
            echo "  -h, --help              Show this help message"
            echo ""
            echo "Flags are stackable. Examples:"
            echo "  $0                      # Sync baseline only (default)"
            echo "  $0 -b -co               # Sync baseline and content-only"
            echo "  $0 -fo -pos             # Sync function-only and POS"
            echo "  $0 -a                   # Sync everything"
            echo ""
            echo "Default: Sync baseline models only"
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            echo "Use -h or --help for usage information"
            exit 1
            ;;
    esac
done

# If nothing was selected, default to baseline
if [ "$SYNC_BASELINE" = false ] && [ "$SYNC_CONTENT" = false ] && [ "$SYNC_FUNCTION" = false ] && [ "$SYNC_POS" = false ]; then
    SYNC_BASELINE=true
fi

echo "=================================================="
echo "       LLM Stylometry Model Sync"
echo "=================================================="
echo
echo "Sync configuration:"
[ "$SYNC_BASELINE" = true ] && echo "  ✓ Baseline models"
[ "$SYNC_CONTENT" = true ] && echo "  ✓ Content-only variant"
[ "$SYNC_FUNCTION" = true ] && echo "  ✓ Function-only variant"
[ "$SYNC_POS" = true ] && echo "  ✓ Part-of-speech variant"
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

# Create temporary file for remote check results
TEMP_FILE=$(mktemp)

# Pass sync flags to remote script via environment
REMOTE_SYNC_BASELINE=$SYNC_BASELINE
REMOTE_SYNC_CONTENT=$SYNC_CONTENT
REMOTE_SYNC_FUNCTION=$SYNC_FUNCTION
REMOTE_SYNC_POS=$SYNC_POS

# Check which models are available on remote server
ssh "$USERNAME@$SERVER_ADDRESS" \
    "SYNC_BASELINE='$REMOTE_SYNC_BASELINE' SYNC_CONTENT='$REMOTE_SYNC_CONTENT' SYNC_FUNCTION='$REMOTE_SYNC_FUNCTION' SYNC_POS='$REMOTE_SYNC_POS' bash -s" << 'ENDSSH' > "$TEMP_FILE"
#!/bin/bash

MODELS_DIR="$HOME/llm-stylometry/models"
EXPECTED_MODELS_PER_VARIANT=80  # 8 authors × 10 seeds

if [ ! -d "$MODELS_DIR" ]; then
    echo "STATUS=ERROR"
    exit 0
fi

# Function to check models for a specific variant
check_variant() {
    local variant_name=$1
    local variant_suffix=$2
    local count=0
    local missing=""

    for author in austen baum dickens fitzgerald melville thompson twain wells; do
        for seed in 0 1 2 3 4 5 6 7 8 9; do
            if [ -z "$variant_suffix" ]; then
                # Baseline model
                MODEL_DIR="$MODELS_DIR/${author}_tokenizer=gpt2_seed=${seed}"
            else
                # Variant model
                MODEL_DIR="$MODELS_DIR/${author}_variant=${variant_suffix}_tokenizer=gpt2_seed=${seed}"
            fi

            if [ -d "$MODEL_DIR" ]; then
                # Check for model weights
                if ls "$MODEL_DIR"/*.pth &>/dev/null || ls "$MODEL_DIR"/*.bin &>/dev/null || ls "$MODEL_DIR"/model.safetensors &>/dev/null; then
                    ((count++))
                else
                    missing="${missing}${author}_seed=${seed} (no weights), "
                fi
            else
                missing="${missing}${author}_seed=${seed} (no dir), "
            fi
        done
    done

    echo "${variant_name}_COUNT=$count"
    if [ -n "$missing" ]; then
        echo "${variant_name}_MISSING=${missing%, }"
    fi

    if [ $count -eq $EXPECTED_MODELS_PER_VARIANT ]; then
        echo "${variant_name}_STATUS=COMPLETE"
    else
        echo "${variant_name}_STATUS=INCOMPLETE"
    fi
}

# Check each requested variant
[ "$SYNC_BASELINE" = "true" ] && check_variant "BASELINE" ""
[ "$SYNC_CONTENT" = "true" ] && check_variant "CONTENT" "content"
[ "$SYNC_FUNCTION" = "true" ] && check_variant "FUNCTION" "function"
[ "$SYNC_POS" = "true" ] && check_variant "POS" "pos"

echo "STATUS=CHECKED"
ENDSSH

# Parse the remote check results
# Using simple variables for Bash 3.2 compatibility (macOS default)
BASELINE_COUNT=0
BASELINE_MISSING=""
BASELINE_STATUS="INCOMPLETE"
CONTENT_COUNT=0
CONTENT_MISSING=""
CONTENT_STATUS="INCOMPLETE"
FUNCTION_COUNT=0
FUNCTION_MISSING=""
FUNCTION_STATUS="INCOMPLETE"
POS_COUNT=0
POS_MISSING=""
POS_STATUS="INCOMPLETE"

while IFS= read -r line; do
    if [[ $line == BASELINE_COUNT=* ]]; then
        BASELINE_COUNT="${line#*=}"
    elif [[ $line == BASELINE_MISSING=* ]]; then
        BASELINE_MISSING="${line#*=}"
    elif [[ $line == BASELINE_STATUS=* ]]; then
        BASELINE_STATUS="${line#*=}"
    elif [[ $line == CONTENT_COUNT=* ]]; then
        CONTENT_COUNT="${line#*=}"
    elif [[ $line == CONTENT_MISSING=* ]]; then
        CONTENT_MISSING="${line#*=}"
    elif [[ $line == CONTENT_STATUS=* ]]; then
        CONTENT_STATUS="${line#*=}"
    elif [[ $line == FUNCTION_COUNT=* ]]; then
        FUNCTION_COUNT="${line#*=}"
    elif [[ $line == FUNCTION_MISSING=* ]]; then
        FUNCTION_MISSING="${line#*=}"
    elif [[ $line == FUNCTION_STATUS=* ]]; then
        FUNCTION_STATUS="${line#*=}"
    elif [[ $line == POS_COUNT=* ]]; then
        POS_COUNT="${line#*=}"
    elif [[ $line == POS_MISSING=* ]]; then
        POS_MISSING="${line#*=}"
    elif [[ $line == POS_STATUS=* ]]; then
        POS_STATUS="${line#*=}"
    elif [[ $line == STATUS=* ]]; then
        OVERALL_STATUS="${line#STATUS=}"
    fi
done < "$TEMP_FILE"

# Clean up temp file
rm -f "$TEMP_FILE"

if [ "$OVERALL_STATUS" = "ERROR" ]; then
    print_error "Models directory not found on remote server"
    exit 1
fi

# Report status for each variant
echo
echo "Remote model status:"
echo "===================="

VARIANTS_TO_SYNC=()

if [ "$SYNC_BASELINE" = true ]; then
    if [ "$BASELINE_STATUS" = "COMPLETE" ]; then
        print_success "Baseline: $BASELINE_COUNT/80 models complete"
        VARIANTS_TO_SYNC+=("baseline")
    else
        print_warning "Baseline: $BASELINE_COUNT/80 models complete - SKIPPING"
        [ -n "$BASELINE_MISSING" ] && echo "  Missing: $BASELINE_MISSING"
    fi
fi

if [ "$SYNC_CONTENT" = true ]; then
    if [ "$CONTENT_STATUS" = "COMPLETE" ]; then
        print_success "Content-only: $CONTENT_COUNT/80 models complete"
        VARIANTS_TO_SYNC+=("content")
    else
        print_warning "Content-only: $CONTENT_COUNT/80 models complete - SKIPPING"
        [ -n "$CONTENT_MISSING" ] && echo "  Missing: $CONTENT_MISSING"
    fi
fi

if [ "$SYNC_FUNCTION" = true ]; then
    if [ "$FUNCTION_STATUS" = "COMPLETE" ]; then
        print_success "Function-only: $FUNCTION_COUNT/80 models complete"
        VARIANTS_TO_SYNC+=("function")
    else
        print_warning "Function-only: $FUNCTION_COUNT/80 models complete - SKIPPING"
        [ -n "$FUNCTION_MISSING" ] && echo "  Missing: $FUNCTION_MISSING"
    fi
fi

if [ "$SYNC_POS" = true ]; then
    if [ "$POS_STATUS" = "COMPLETE" ]; then
        print_success "Part-of-speech: $POS_COUNT/80 models complete"
        VARIANTS_TO_SYNC+=("pos")
    else
        print_warning "Part-of-speech: $POS_COUNT/80 models complete - SKIPPING"
        [ -n "$POS_MISSING" ] && echo "  Missing: $POS_MISSING"
    fi
fi

# Check if we have anything to sync
if [ ${#VARIANTS_TO_SYNC[@]} -eq 0 ]; then
    echo
    print_error "No complete model sets found to sync"
    exit 1
fi

echo
print_info "Will sync: ${VARIANTS_TO_SYNC[*]}"

# Ask for confirmation before downloading
echo
echo "This will download and replace your local models for the selected variants."
read -p "Continue? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    print_info "Download cancelled"
    exit 0
fi

# Prepare local models directory
LOCAL_MODELS_DIR="$PWD/models"
BACKUP_DIR=""

# Only backup if doing a full replacement (all variants selected)
# Otherwise, merge new models into existing directory
if [ "$SYNC_BASELINE" = true ] && [ "$SYNC_CONTENT" = true ] && [ "$SYNC_FUNCTION" = true ] && [ "$SYNC_POS" = true ]; then
    # Full replacement - backup existing models
    if [ -d "$LOCAL_MODELS_DIR" ] && [ "$(ls -A $LOCAL_MODELS_DIR)" ]; then
        print_info "Full sync requested - backing up ALL existing local models..."
        BACKUP_DIR="${PWD}/models_backup_$(date +%Y%m%d_%H%M%S)"
        mv "$LOCAL_MODELS_DIR" "$BACKUP_DIR"
        print_success "Local models backed up to: $BACKUP_DIR"
    fi
    mkdir -p "$LOCAL_MODELS_DIR"
else
    # Partial sync - merge into existing directory
    mkdir -p "$LOCAL_MODELS_DIR"
    if [ -d "$LOCAL_MODELS_DIR" ] && [ "$(ls -A $LOCAL_MODELS_DIR)" ]; then
        print_info "Merging new models into existing directory (existing models from other variants preserved)"
    fi
fi

# Download models for each variant
TOTAL_SYNCED=0

for variant in "${VARIANTS_TO_SYNC[@]}"; do
    print_info "Syncing $variant models..."

    if [ "$variant" = "baseline" ]; then
        # Sync baseline models (no variant suffix)
        rsync -avz --progress --include="*_tokenizer=gpt2_seed=*/" \
            --exclude="*_variant=*" \
            --include="*/" --include="**" \
            "$USERNAME@$SERVER_ADDRESS:~/llm-stylometry/models/" "$LOCAL_MODELS_DIR/"
    else
        # Sync variant models
        rsync -avz --progress --include="*_variant=${variant}_tokenizer=gpt2_seed=*/" \
            --include="*/" --include="**" \
            "$USERNAME@$SERVER_ADDRESS:~/llm-stylometry/models/" "$LOCAL_MODELS_DIR/"
    fi

    if [ $? -eq 0 ]; then
        print_success "$variant models synced successfully"
        ((TOTAL_SYNCED++))
    else
        print_error "Failed to sync $variant models"
    fi
done

# Verify synced models
echo
print_info "Verifying synced models..."
SYNCED_COUNT=$(find "$LOCAL_MODELS_DIR" -maxdepth 1 -type d -name "*_tokenizer=gpt2_seed=*" -o -name "*_variant=*_tokenizer=gpt2_seed=*" | wc -l)
print_success "Successfully synced $SYNCED_COUNT model directories"

# Also download model_results.pkl if it exists (for any synced variant)
print_info "Checking for consolidated results file..."
RESULTS_EXISTS=$(ssh "$USERNAME@$SERVER_ADDRESS" '[ -f "$HOME/llm-stylometry/data/model_results.pkl" ] && echo "yes" || echo "no"')

if [ "$RESULTS_EXISTS" = "yes" ]; then
    print_info "Downloading model_results.pkl..."
    mkdir -p "$PWD/data"
    rsync -avz "$USERNAME@$SERVER_ADDRESS:~/llm-stylometry/data/model_results.pkl" \
        "$PWD/data/model_results.pkl"
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
echo "✓ Synced $TOTAL_SYNCED variant(s)"
echo "✓ Total model directories: $SYNCED_COUNT"
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
echo "  ./run_llm_stylometry.sh"