#!/bin/bash

# Remote Training Script for LLM Stylometry
# This script connects to a GPU server, clones/updates the repository, and starts training

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

# Verification functions for backup/restore (to be used on remote server)
verify_backup() {
    local BACKUP_DIR=$1
    local MODELS_DIR=$2

    MODEL_COUNT=$(find "$BACKUP_DIR" -name "model.pth" 2>/dev/null | wc -l)
    STATE_COUNT=$(find "$BACKUP_DIR" -name "training_state.pt" 2>/dev/null | wc -l)
    LOG_COUNT=$(find "$BACKUP_DIR" -name "loss_logs.csv" 2>/dev/null | wc -l)
    CONFIG_COUNT=$(find "$BACKUP_DIR" -name "config.json" 2>/dev/null | wc -l)

    echo "Backup verification:"
    echo "  - $MODEL_COUNT model.pth files"
    echo "  - $STATE_COUNT training_state.pt files"
    echo "  - $LOG_COUNT loss_logs.csv files"
    echo "  - $CONFIG_COUNT config.json files"

    # Check if we expected backups but got none
    if [ -d "$MODELS_DIR" ] && [ "$(ls -A "$MODELS_DIR" 2>/dev/null)" ]; then
        # models/ directory exists and is not empty
        if [ $MODEL_COUNT -eq 0 ] && [ $STATE_COUNT -eq 0 ]; then
            echo "ERROR: models/ directory exists but backup contains no checkpoint files!"
            return 1
        fi
    fi

    return 0
}

verify_restore() {
    local BACKUP_DIR=$1
    local MODELS_DIR=$2

    BACKUP_MODEL_COUNT=$(find "$BACKUP_DIR" -name "model.pth" 2>/dev/null | wc -l)
    BACKUP_STATE_COUNT=$(find "$BACKUP_DIR" -name "training_state.pt" 2>/dev/null | wc -l)
    RESTORED_MODEL_COUNT=$(find "$MODELS_DIR" -name "model.pth" 2>/dev/null | wc -l)
    RESTORED_STATE_COUNT=$(find "$MODELS_DIR" -name "training_state.pt" 2>/dev/null | wc -l)

    echo "Restore verification:"
    echo "  - Backed up: $BACKUP_MODEL_COUNT model.pth, $BACKUP_STATE_COUNT training_state.pt"
    echo "  - Restored: $RESTORED_MODEL_COUNT model.pth, $RESTORED_STATE_COUNT training_state.pt"

    if [ $BACKUP_MODEL_COUNT -ne $RESTORED_MODEL_COUNT ]; then
        echo "ERROR: Model count mismatch! Backup=$BACKUP_MODEL_COUNT, Restored=$RESTORED_MODEL_COUNT"
        return 1
    fi

    if [ $BACKUP_STATE_COUNT -ne $RESTORED_STATE_COUNT ]; then
        echo "ERROR: Training state count mismatch! Backup=$BACKUP_STATE_COUNT, Restored=$RESTORED_STATE_COUNT"
        return 1
    fi

    return 0
}

echo "=================================================="
echo "       LLM Stylometry Remote Training Setup"
echo "=================================================="
echo
echo "Usage: $0 [options]"
echo "Options:"
echo "  --kill, -k              Kill existing training sessions before starting new one"
echo "  --resume, -r            Resume training from existing checkpoints"
echo "  -co, --content-only     Train content-only variant"
echo "  -fo, --function-only    Train function-only variant"
echo "  -pos, --part-of-speech  Train part-of-speech variant"
echo "  -g, --max-gpus NUM      Maximum number of GPUs to use (default: 4)"
echo "  --cluster NAME          Select cluster: tensor01 or tensor02 (default: tensor02)"
echo

# Parse command line arguments
KILL_MODE=false
RESUME_MODE=false
VARIANT_ARG=""
MAX_GPUS=""
CLUSTER="tensor02"  # Default cluster

while [[ $# -gt 0 ]]; do
    case $1 in
        --kill|-k)
            echo "Kill mode: Will terminate existing training sessions"
            KILL_MODE=true
            shift
            ;;
        --resume|-r)
            echo "Resume mode: Will continue training from existing checkpoints"
            RESUME_MODE=true
            shift
            ;;
        -co|--content-only)
            VARIANT_ARG="-co"
            echo "Training content-only variant"
            shift
            ;;
        -fo|--function-only)
            VARIANT_ARG="-fo"
            echo "Training function-only variant"
            shift
            ;;
        -pos|--part-of-speech)
            VARIANT_ARG="-pos"
            echo "Training part-of-speech variant"
            shift
            ;;
        -g|--max-gpus)
            MAX_GPUS="$2"
            echo "Using maximum $MAX_GPUS GPUs"
            shift 2
            ;;
        --cluster)
            CLUSTER="$2"
            echo "Using cluster: $CLUSTER"
            shift 2
            ;;
        *)
            shift
            ;;
    esac
done

# Get server details - try to read from cluster-specific credentials file first
CRED_FILE=".ssh/credentials_${CLUSTER}.json"
if [ -f "$CRED_FILE" ]; then
    print_info "Found credentials file for $CLUSTER, using saved credentials..."
    SERVER_ADDRESS=$(python3 -c "import json; print(json.load(open('$CRED_FILE'))['server'])" 2>/dev/null)
    USERNAME=$(python3 -c "import json; print(json.load(open('$CRED_FILE'))['username'])" 2>/dev/null)
    PASSWORD=$(python3 -c "import json; print(json.load(open('$CRED_FILE'))['password'])" 2>/dev/null)

    if [ -z "$SERVER_ADDRESS" ] || [ -z "$USERNAME" ] || [ -z "$PASSWORD" ]; then
        print_error "Failed to read credentials from $CRED_FILE"
        exit 1
    fi

    USE_SSHPASS=true
else
    print_warning "No credentials file found at $CRED_FILE"
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

    USE_SSHPASS=false
fi

print_info "Connecting to $USERNAME@$SERVER_ADDRESS..."

# Build SSH command based on authentication method
if [ "$USE_SSHPASS" = true ]; then
    # Use sshpass for password authentication
    if ! command -v sshpass &> /dev/null; then
        print_error "sshpass is required but not installed. Please install it: brew install hudochenkov/sshpass/sshpass"
        exit 1
    fi
    SSH_CMD="sshpass -p '$PASSWORD' ssh -o StrictHostKeyChecking=no -t"
else
    # Test SSH connection first with interactive authentication
    if ! ssh -o ConnectTimeout=5 -o BatchMode=yes "$USERNAME@$SERVER_ADDRESS" "echo 'Connection test successful'" 2>/dev/null; then
        print_warning "Initial connection test failed. Trying with interactive authentication..."
    fi
    echo
    SSH_CMD="ssh -t"
fi

# Execute the remote script via SSH
eval "$SSH_CMD \"$USERNAME@$SERVER_ADDRESS\" \"KILL_MODE='$KILL_MODE' RESUME_MODE='$RESUME_MODE' VARIANT_ARG='$VARIANT_ARG' MAX_GPUS='$MAX_GPUS' bash -s\"" << ENDSSH
#!/bin/bash
set -e

# Define verification functions for use on remote server
verify_backup() {
    local BACKUP_DIR=\$1
    local MODELS_DIR=\$2

    MODEL_COUNT=\$(find "\$BACKUP_DIR" -name "model.pth" 2>/dev/null | wc -l)
    STATE_COUNT=\$(find "\$BACKUP_DIR" -name "training_state.pt" 2>/dev/null | wc -l)
    LOG_COUNT=\$(find "\$BACKUP_DIR" -name "loss_logs.csv" 2>/dev/null | wc -l)
    CONFIG_COUNT=\$(find "\$BACKUP_DIR" -name "config.json" 2>/dev/null | wc -l)

    echo "Backup verification:"
    echo "  - \$MODEL_COUNT model.pth files"
    echo "  - \$STATE_COUNT training_state.pt files"
    echo "  - \$LOG_COUNT loss_logs.csv files"
    echo "  - \$CONFIG_COUNT config.json files"

    # Check if we expected backups but got none
    if [ -d "\$MODELS_DIR" ] && [ "\$(ls -A "\$MODELS_DIR" 2>/dev/null)" ]; then
        # models/ directory exists and is not empty
        if [ \$MODEL_COUNT -eq 0 ] && [ \$STATE_COUNT -eq 0 ]; then
            echo "ERROR: models/ directory exists but backup contains no checkpoint files!"
            return 1
        fi
    fi

    return 0
}

verify_restore() {
    local BACKUP_DIR=\$1
    local MODELS_DIR=\$2

    BACKUP_MODEL_COUNT=\$(find "\$BACKUP_DIR" -name "model.pth" 2>/dev/null | wc -l)
    BACKUP_STATE_COUNT=\$(find "\$BACKUP_DIR" -name "training_state.pt" 2>/dev/null | wc -l)
    RESTORED_MODEL_COUNT=\$(find "\$MODELS_DIR" -name "model.pth" 2>/dev/null | wc -l)
    RESTORED_STATE_COUNT=\$(find "\$MODELS_DIR" -name "training_state.pt" 2>/dev/null | wc -l)

    echo "Restore verification:"
    echo "  - Backed up: \$BACKUP_MODEL_COUNT model.pth, \$BACKUP_STATE_COUNT training_state.pt"
    echo "  - Restored: \$RESTORED_MODEL_COUNT model.pth, \$RESTORED_STATE_COUNT training_state.pt"

    if [ \$BACKUP_MODEL_COUNT -ne \$RESTORED_MODEL_COUNT ]; then
        echo "ERROR: Model count mismatch! Backup=\$BACKUP_MODEL_COUNT, Restored=\$RESTORED_MODEL_COUNT"
        return 1
    fi

    if [ \$BACKUP_STATE_COUNT -ne \$RESTORED_STATE_COUNT ]; then
        echo "ERROR: Training state count mismatch! Backup=\$BACKUP_STATE_COUNT, Restored=\$RESTORED_STATE_COUNT"
        return 1
    fi

    return 0
}

echo "=================================================="
echo "Setting up LLM Stylometry on remote server"
echo "=================================================="
echo

# Check if we're in kill mode
if [ "\$KILL_MODE" = "true" ]; then
    echo "Kill mode activated - terminating existing training sessions..."

    # Kill any existing screen sessions
    screen -ls | grep -o '[0-9]*\.llm_training' | cut -d. -f1 | while read pid; do
        if [ ! -z "\$pid" ]; then
            echo "Killing screen session with PID: \$pid"
            screen -X -S "\$pid.llm_training" quit
        fi
    done

    # Also kill any remaining python training processes
    pkill -f "python.*generate_figures.py.*--train" 2>/dev/null || true

    echo "All training sessions terminated."
    echo ""

    # In non-interactive mode, always start new training after killing
    echo "Starting new training session..."
    echo ""
fi

# Check if repository exists
if [ -d ~/llm-stylometry ]; then
    echo "Repository exists. Updating..."
    cd ~/llm-stylometry

    # BACKUP PHASE: Create filesystem backup of models directory
    BACKUP_DIR=~/model_backups/backup_\$(date +%Y%m%d_%H%M%S)
    if [ -d ~/llm-stylometry/models ] && [ "\$(ls -A ~/llm-stylometry/models 2>/dev/null)" ]; then
        echo "Backing up models directory to \$BACKUP_DIR..."
        mkdir -p "\$BACKUP_DIR"
        rsync -a ~/llm-stylometry/models/ "\$BACKUP_DIR/"

        # Verify backup
        if ! verify_backup "\$BACKUP_DIR" "~/llm-stylometry/models"; then
            echo "ERROR: Backup verification failed!"
            exit 1
        fi
    else
        echo "No models directory to backup (this is normal for first run)"
        BACKUP_DIR=""
    fi

    # GIT OPERATIONS PHASE
    echo "Stashing tracked/untracked changes (config files, logs, etc.)..."
    git stash -u
    echo "Pulling latest changes..."
    git pull
    echo "Restoring stashed changes..."
    git stash pop || echo "No stashed changes to restore (this is normal)"

    # RESTORE PHASE: Restore models from backup
    if [ -n "\$BACKUP_DIR" ]; then
        echo "Restoring models from backup..."
        rsync -a "\$BACKUP_DIR/" ~/llm-stylometry/models/

        # Verify restoration
        if ! verify_restore "\$BACKUP_DIR" "~/llm-stylometry/models"; then
            echo "ERROR: Restore verification failed!"
            exit 1
        fi

        echo "Models restored successfully"
    fi

    # CLEANUP PHASE: Keep only last 3 backups
    if [ -d ~/model_backups ]; then
        echo "Cleaning up old backups (keeping last 3)..."
        cd ~/model_backups
        ls -t | tail -n +4 | xargs -r rm -rf
        echo "Cleanup complete. Current backups:"
        ls -t | head -3
    fi

    echo "Repository updated successfully"
else
    echo "Repository not found. Cloning..."
    cd ~
    git clone https://github.com/ContextLab/llm-stylometry.git
    cd ~/llm-stylometry
    echo "Repository cloned successfully"
fi

# Check for screen
if ! command -v screen &> /dev/null; then
    echo "Installing screen..."
    if command -v apt-get &> /dev/null; then
        sudo apt-get update && sudo apt-get install -y screen
    elif command -v yum &> /dev/null; then
        sudo yum install -y screen
    else
        echo "Warning: Could not install screen. Please install it manually."
    fi
fi

# Create log directory
mkdir -p ~/llm-stylometry/logs
LOG_FILE=~/llm-stylometry/logs/training_\$(date +%Y%m%d_%H%M%S).log

echo ""
echo "=================================================="
echo "Starting training in screen session"
echo "=================================================="
echo "Training will run in a screen session named: llm_training"
echo "Log file: \$LOG_FILE"
echo ""
echo "Useful commands:"
echo "  - Detach from screen: Ctrl+A, then D"
echo "  - Reattach later: screen -r llm_training"
echo "  - View log: tail -f ~/llm-stylometry/logs/training_*.log"
echo ""
echo "Starting training in 5 seconds..."
sleep 5

# Kill any existing screen session with the same name
screen -X -S llm_training quit 2>/dev/null || true

# Start training in screen (use --no-confirm flag for non-interactive mode)
# Create a script file first with RESUME_MODE, VARIANT_ARG, and MAX_GPUS variables
echo "RESUME_MODE='$RESUME_MODE'" > /tmp/llm_train.sh
echo "VARIANT_ARG='$VARIANT_ARG'" >> /tmp/llm_train.sh
echo "MAX_GPUS='$MAX_GPUS'" >> /tmp/llm_train.sh
echo "echo '[DEBUG] RESUME_MODE value is:' '\$RESUME_MODE'" >> /tmp/llm_train.sh
echo "echo '[DEBUG] VARIANT_ARG value is:' '\$VARIANT_ARG'" >> /tmp/llm_train.sh
echo "echo '[DEBUG] MAX_GPUS value is:' '\$MAX_GPUS'" >> /tmp/llm_train.sh
cat >> /tmp/llm_train.sh << 'TRAINSCRIPT'
#!/bin/bash
set -e  # Exit on error

# Change to the repository directory
cd ~/llm-stylometry

# Create log directory and file
mkdir -p logs
LOG_FILE=~/llm-stylometry/logs/training_\$(date +%Y%m%d_%H%M%S).log
echo "Training started at \$(date)" | tee \$LOG_FILE

# Check if the run script exists
if [ ! -f ./run_llm_stylometry.sh ]; then
    echo "ERROR: run_llm_stylometry.sh not found in \$(pwd)!" | tee -a \$LOG_FILE
    ls -la | tee -a \$LOG_FILE
    exit 1
fi

# Make sure it's executable
chmod +x ./run_llm_stylometry.sh

# Run the training script with non-interactive flag
echo "Starting training with run_llm_stylometry.sh..." | tee -a \$LOG_FILE

# Build GPU flag if MAX_GPUS is set
GPU_FLAG=""
if [ -n "\$MAX_GPUS" ]; then
    GPU_FLAG="--max-gpus \$MAX_GPUS"
    echo "Using maximum \$MAX_GPUS GPUs" | tee -a \$LOG_FILE
fi

# Check if we're in resume mode
if [ "\$RESUME_MODE" = "true" ]; then
    echo "Running in resume mode - continuing from existing checkpoints" | tee -a \$LOG_FILE
    ./run_llm_stylometry.sh --train --resume -y \$VARIANT_ARG \$GPU_FLAG 2>&1 | tee -a \$LOG_FILE
else
    echo "Running full training from scratch" | tee -a \$LOG_FILE
    ./run_llm_stylometry.sh --train -y \$VARIANT_ARG \$GPU_FLAG 2>&1 | tee -a \$LOG_FILE
fi

echo "Training completed at \$(date)" | tee -a \$LOG_FILE
TRAINSCRIPT

chmod +x /tmp/llm_train.sh

# Start screen session
screen -dmS llm_training /tmp/llm_train.sh

# Wait a moment for screen to start
sleep 2

# Check if screen session started
echo "Checking screen sessions:"
screen -list

if screen -list | grep -q "llm_training"; then
    echo ""
    echo "✓ Training started successfully in screen session!"
    echo ""
    echo "The training is now running in the background."
    echo "You can safely disconnect from SSH."
    echo ""
    echo "To monitor progress, reconnect and run:"
    echo "  screen -r llm_training"
    echo ""
    echo "Or view the log file:"
    echo "  tail -f \$LOG_FILE"

    # Attach to screen session
    echo ""
    echo "Attaching to screen session in 3 seconds..."
    echo "(Press Ctrl+A, then D to detach and leave training running)"
    sleep 3
    screen -r llm_training
else
    echo "Error: Failed to start screen session"
    exit 1
fi
ENDSSH

RESULT=$?
if [ $RESULT -eq 0 ]; then
    print_success "Remote training setup completed!"
    echo
    echo "Training is running on $SERVER_ADDRESS"
    echo "To reconnect and check progress:"
    echo "  ssh $USERNAME@$SERVER_ADDRESS"
    echo "  screen -r llm_training"
else
    print_error "Remote training setup failed"
    exit 1
fi