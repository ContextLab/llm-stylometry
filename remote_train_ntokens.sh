#!/bin/bash

# Remote Training Script for LLM Stylometry — Dataset-Size Sweep
# Trains models at multiple token levels (N_TRAIN_TOKENS) on a GPU cluster.
#
# Usage: ./remote_train_ntokens.sh --cluster tensor02 [options]

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
print_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
print_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
print_error() { echo -e "${RED}[ERROR]${NC} $1"; }

echo "=================================================="
echo "   LLM Stylometry — Dataset-Size Sweep Training"
echo "=================================================="
echo
echo "Usage: $0 [options]"
echo "Options:"
echo "  --kill, -k              Kill existing training sessions"
echo "  --resume, -r            Resume from existing checkpoints"
echo "  --cluster NAME          Select cluster (required)"
echo "  --tokens LIST           Comma-separated token levels (default: all 19 levels)"
echo "  -g, --max-gpus NUM      Maximum GPUs to use (default: all)"
echo

# Default token levels matching the paper's sweep
DEFAULT_TOKENS="2500,5000,10000,20000,40000,45000,50000,55000,60000,65000,70000,75000,80000,128608,160000,257216,385825,514433,643041"

# Parse arguments
KILL_MODE=false
RESUME_MODE=false
MAX_GPUS=""
CLUSTER=""
TOKEN_LEVELS="$DEFAULT_TOKENS"

while [[ $# -gt 0 ]]; do
    case $1 in
        --kill|-k)
            KILL_MODE=true
            shift
            ;;
        --resume|-r)
            RESUME_MODE=true
            shift
            ;;
        -g|--max-gpus)
            MAX_GPUS="$2"
            shift 2
            ;;
        --cluster)
            CLUSTER="$2"
            shift 2
            ;;
        --tokens)
            TOKEN_LEVELS="$2"
            shift 2
            ;;
        *)
            shift
            ;;
    esac
done

if [ -z "$CLUSTER" ]; then
    print_error "Cluster must be specified with --cluster flag"
    echo "Example: $0 --cluster tensor02"
    exit 1
fi

# Load credentials
CRED_FILE=".ssh/credentials_${CLUSTER}.json"
if [ -f "$CRED_FILE" ]; then
    print_info "Found credentials for $CLUSTER"
    SERVER_ADDRESS=$(python3 -c "import json; print(json.load(open('$CRED_FILE'))['server'])" 2>/dev/null)
    USERNAME=$(python3 -c "import json; print(json.load(open('$CRED_FILE'))['username'])" 2>/dev/null)
    PASSWORD=$(python3 -c "import json; print(json.load(open('$CRED_FILE'))['password'])" 2>/dev/null)

    if [ -z "$SERVER_ADDRESS" ] || [ -z "$USERNAME" ] || [ -z "$PASSWORD" ]; then
        print_error "Failed to read credentials from $CRED_FILE"
        exit 1
    fi
    USE_SSHPASS=true
else
    print_warning "No credentials file at $CRED_FILE"
    read -p "Enter server address: " SERVER_ADDRESS
    read -p "Enter username: " USERNAME
    USE_SSHPASS=false
fi

print_info "Connecting to $USERNAME@$SERVER_ADDRESS..."
print_info "Token levels: $TOKEN_LEVELS"

# Build SSH command
if [ "$USE_SSHPASS" = true ]; then
    if ! command -v sshpass &> /dev/null; then
        print_error "sshpass required but not installed: brew install hudochenkov/sshpass/sshpass"
        exit 1
    fi
    SSH_CMD="sshpass -p '$PASSWORD' ssh -o StrictHostKeyChecking=no -t"
else
    SSH_CMD="ssh -t"
fi

eval "$SSH_CMD \"$USERNAME@$SERVER_ADDRESS\" \"KILL_MODE='$KILL_MODE' RESUME_MODE='$RESUME_MODE' TOKEN_LEVELS='$TOKEN_LEVELS' MAX_GPUS='$MAX_GPUS' bash -s\"" << 'ENDSSH'
#!/bin/bash
set -e

echo "=================================================="
echo "Setting up ntokens sweep on remote server"
echo "=================================================="

# Kill mode
if [ "$KILL_MODE" = "true" ]; then
    echo "Killing existing ntokens training sessions..."
    screen -ls | grep -o '[0-9]*\.ntokens_training' | cut -d. -f1 | while read pid; do
        [ -n "$pid" ] && screen -X -S "$pid.ntokens_training" quit
    done
    pkill -f "N_TRAIN_TOKENS.*python.*main.py" 2>/dev/null || true
    echo "Sessions terminated."
fi

# Clone or update repo
if [ -d ~/llm-stylometry ]; then
    echo "Updating repository..."
    cd ~/llm-stylometry
    git stash -u 2>/dev/null || true
    git pull
    git stash pop 2>/dev/null || true
else
    echo "Cloning repository..."
    cd ~
    git clone https://github.com/ContextLab/llm-stylometry.git
    cd ~/llm-stylometry
fi

# Ensure screen is available
if ! command -v screen &> /dev/null; then
    if command -v apt-get &> /dev/null; then
        sudo apt-get update && sudo apt-get install -y screen
    elif command -v yum &> /dev/null; then
        sudo yum install -y screen
    fi
fi

# Setup conda environment if needed
if ! conda env list 2>/dev/null | grep -q llm-stylometry; then
    echo "Creating conda environment..."
    conda create -n llm-stylometry python=3.10 -y
fi

eval "$(conda shell.bash hook)"
conda activate llm-stylometry

# Install dependencies
pip install -e . 2>/dev/null || true
pip install torch --index-url https://download.pytorch.org/whl/cu121 2>/dev/null || true

# Create training script
mkdir -p ~/llm-stylometry/logs
LOG_FILE=~/llm-stylometry/logs/ntokens_training_$(date +%Y%m%d_%H%M%S).log

cat > /tmp/ntokens_train.sh << TRAINSCRIPT
#!/bin/bash
set -e
cd ~/llm-stylometry

eval "\$(conda shell.bash hook)"
conda activate llm-stylometry

LOG_FILE=$LOG_FILE
echo "Ntokens sweep started at \$(date)" | tee \$LOG_FILE

IFS=',' read -ra TOKENS <<< "$TOKEN_LEVELS"
TOTAL=\${#TOKENS[@]}
CURRENT=0

for N in "\${TOKENS[@]}"; do
    ((CURRENT++))
    echo "" | tee -a \$LOG_FILE
    echo "========================================" | tee -a \$LOG_FILE
    echo "[\$CURRENT/\$TOTAL] Training with N_TRAIN_TOKENS=\$N" | tee -a \$LOG_FILE
    echo "Started at \$(date)" | tee -a \$LOG_FILE
    echo "========================================" | tee -a \$LOG_FILE

    RESUME_FLAG=""
    if [ "$RESUME_MODE" = "true" ]; then
        RESUME_FLAG="RESUME_TRAINING=1"
    fi

    GPU_ENV=""
    if [ -n "$MAX_GPUS" ]; then
        GPU_ENV="MAX_GPUS=$MAX_GPUS"
    fi

    env N_TRAIN_TOKENS=\$N \$RESUME_FLAG \$GPU_ENV DISABLE_TQDM=1 \
        python code/main.py 2>&1 | tee -a \$LOG_FILE

    echo "Completed N_TRAIN_TOKENS=\$N at \$(date)" | tee -a \$LOG_FILE
done

echo "" | tee -a \$LOG_FILE
echo "All token levels complete at \$(date)" | tee -a \$LOG_FILE
TRAINSCRIPT

chmod +x /tmp/ntokens_train.sh

# Kill existing session if any
screen -X -S ntokens_training quit 2>/dev/null || true

echo ""
echo "=================================================="
echo "Starting ntokens sweep in screen session"
echo "=================================================="
echo "Screen session: ntokens_training"
echo "Log: $LOG_FILE"
echo ""
echo "  Detach: Ctrl+A, then D"
echo "  Reattach: screen -r ntokens_training"
echo "  View log: tail -f $LOG_FILE"
echo ""
sleep 3

screen -dmS ntokens_training /tmp/ntokens_train.sh
sleep 2

if screen -list | grep -q "ntokens_training"; then
    echo "✓ Ntokens sweep started!"
    sleep 2
    screen -r ntokens_training
else
    echo "ERROR: Failed to start screen session"
    exit 1
fi
ENDSSH

RESULT=$?
if [ $RESULT -eq 0 ]; then
    print_success "Remote ntokens sweep setup completed!"
    echo "Training running on $SERVER_ADDRESS"
    echo "Check progress: ./check_ntokens_status.sh --cluster $CLUSTER"
else
    print_error "Remote setup failed"
    exit 1
fi
