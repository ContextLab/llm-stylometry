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
echo "Usage: $0 [options]"
echo "Options:"
echo "  --kill, -k   Kill existing training sessions before starting new one"
echo

# Check for --kill flag
if [ "$1" = "--kill" ] || [ "$1" = "-k" ]; then
    echo "Kill mode: Will terminate existing training sessions"
    KILL_MODE=true
else
    KILL_MODE=false
fi

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

print_info "Connecting to $USERNAME@$SERVER_ADDRESS..."

# Test SSH connection first
if ! ssh -o ConnectTimeout=5 -o BatchMode=yes "$USERNAME@$SERVER_ADDRESS" "echo 'Connection test successful'" 2>/dev/null; then
    print_warning "Initial connection test failed. Trying with interactive authentication..."
fi

echo

# Execute the remote script via SSH
ssh -t "$USERNAME@$SERVER_ADDRESS" "KILL_MODE='$KILL_MODE' bash -s" << 'ENDSSH'
#!/bin/bash
set -e

echo "=================================================="
echo "Setting up LLM Stylometry on remote server"
echo "=================================================="
echo

# Check if we're in kill mode
if [ "$KILL_MODE" = "true" ]; then
    echo "Kill mode activated - terminating existing training sessions..."

    # Kill any existing screen sessions
    screen -ls | grep -o '[0-9]*\.llm_training' | cut -d. -f1 | while read pid; do
        if [ ! -z "$pid" ]; then
            echo "Killing screen session with PID: $pid"
            screen -X -S "$pid.llm_training" quit
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
    git pull
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
LOG_FILE=~/llm-stylometry/logs/training_$(date +%Y%m%d_%H%M%S).log

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
echo "  - View log: tail -f ~/llm-stylometry/logs/training_*.log"
echo ""
echo "Starting training in 5 seconds..."
sleep 5

# Kill any existing screen session with the same name
screen -X -S llm_training quit 2>/dev/null || true

# Start training in screen (use --no-confirm flag for non-interactive mode)
# Create a script file first
cat > /tmp/llm_train.sh << 'TRAINSCRIPT'
#!/bin/bash
set -e  # Exit on error

# Change to the repository directory
cd ~/llm-stylometry

# Create log directory and file
mkdir -p logs
LOG_FILE=~/llm-stylometry/logs/training_$(date +%Y%m%d_%H%M%S).log
echo "Training started at $(date)" | tee $LOG_FILE

# Check if the run script exists
if [ ! -f ./run_llm_stylometry.sh ]; then
    echo "ERROR: run_llm_stylometry.sh not found in $(pwd)!" | tee -a $LOG_FILE
    ls -la | tee -a $LOG_FILE
    exit 1
fi

# Make sure it's executable
chmod +x ./run_llm_stylometry.sh

# Run the training script with non-interactive flag
echo "Starting training with run_llm_stylometry.sh..." | tee -a $LOG_FILE
./run_llm_stylometry.sh --train -y 2>&1 | tee -a $LOG_FILE

echo "Training completed at $(date)" | tee -a $LOG_FILE
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