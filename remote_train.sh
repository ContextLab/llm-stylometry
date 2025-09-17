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

echo "=================================================="
echo "       LLM Stylometry Remote Training Setup"
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

# Create the remote training script
REMOTE_SCRIPT='
#!/bin/bash
set -e

echo "=================================================="
echo "Setting up LLM Stylometry on remote server"
echo "=================================================="
echo

# Check if repo exists
if [ -d "$HOME/llm-stylometry" ]; then
    echo "Repository exists. Updating to latest version..."
    cd "$HOME/llm-stylometry"

    # Stash any local changes
    if ! git diff --quiet || ! git diff --cached --quiet; then
        echo "Stashing local changes..."
        git stash
    fi

    # Update repository
    git fetch origin
    git checkout main
    git pull origin main
    echo "Repository updated successfully"
else
    echo "Cloning repository..."
    cd "$HOME"
    git clone https://github.com/ContextLab/llm-stylometry.git
    cd "$HOME/llm-stylometry"
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
mkdir -p "$HOME/llm-stylometry/logs"
LOG_FILE="$HOME/llm-stylometry/logs/training_$(date +%Y%m%d_%H%M%S).log"

echo ""
echo "=================================================="
echo "Starting training in screen session"
echo "=================================================="
echo "Training will run in a screen session named: llm_training"
echo "Log file: $LOG_FILE"
echo ""
echo "Useful commands:"
echo "  - Detach from screen: Ctrl+A, then D"
echo "  - Reattach later: screen -r llm_training"
echo "  - View log: tail -f $LOG_FILE"
echo ""
echo "Starting training in 5 seconds..."
sleep 5

# Kill any existing screen session with the same name
screen -X -S llm_training quit 2>/dev/null || true

# Start training in screen
screen -dmS llm_training bash -c "
    cd $HOME/llm-stylometry
    echo 'Training started at $(date)' | tee -a $LOG_FILE
    ./run_llm_stylometry.sh --train 2>&1 | tee -a $LOG_FILE
    echo 'Training completed at $(date)' | tee -a $LOG_FILE
"

# Wait a moment for screen to start
sleep 2

# Check if screen session started
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
    echo "  tail -f $LOG_FILE"

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
'

# Execute the remote script via SSH
print_info "Connecting to $USERNAME@$SERVER_ADDRESS..."
print_info "You may be prompted for your password and/or GitHub credentials."
echo

ssh -t "$USERNAME@$SERVER_ADDRESS" "$REMOTE_SCRIPT"

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