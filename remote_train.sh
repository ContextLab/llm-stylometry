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

# Ask about GitHub authentication method first
print_info "GitHub authentication options:"
echo "1. SSH key (recommended if already set up on server)"
echo "2. Personal Access Token (will prompt when needed)"
echo "3. Skip repository update (use existing code on server)"
read -p "Choose option [1-3]: " AUTH_OPTION

if [ "$AUTH_OPTION" = "2" ]; then
    echo
    print_warning "GitHub now requires Personal Access Tokens instead of passwords."
    print_info "Create one at: https://github.com/settings/tokens"
    print_info "Grant 'repo' scope for private repository access."
    echo
    read -p "Enter your GitHub username: " GH_USER
    read -s -p "Enter your GitHub Personal Access Token: " GH_TOKEN
    echo
    export GH_USER GH_TOKEN
fi

# Now get server details
echo
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
# Pass variables as arguments to the remote bash
ssh -t "$USERNAME@$SERVER_ADDRESS" "AUTH_OPTION='$AUTH_OPTION' GH_USER='$GH_USER' GH_TOKEN='$GH_TOKEN' KILL_MODE='$KILL_MODE' bash -s" << 'ENDSSH'
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

    # Ask if user wants to start new training
    read -t 10 -p "Start new training session? (y/n, default: n in 10s): " START_NEW || START_NEW="n"
    if [ "$START_NEW" != "y" ] && [ "$START_NEW" != "Y" ]; then
        echo "Exiting without starting new training."
        exit 0
    fi
    echo "Continuing to start new training session..."
    echo ""
fi

# Handle different authentication options
if [ "$AUTH_OPTION" = "3" ]; then
    echo "Skipping all repository operations as requested..."
    if [ ! -d ~/llm-stylometry ]; then
        echo "Error: Repository not found at ~/llm-stylometry"
        echo "Please choose option 1 or 2 to clone the repository first."
        exit 1
    fi
    cd ~/llm-stylometry
    echo "Using existing repository without updates"
else
    # Configure Git to use credential caching to avoid repeated auth prompts
    git config --global credential.helper cache
    git config --global credential.helper 'cache --timeout=3600'

    # Set up GitHub token authentication if provided (only for option 2)
    if [ "$AUTH_OPTION" = "2" ] && [ -n "$GH_TOKEN" ] && [ -n "$GH_USER" ]; then
        echo "Setting up GitHub token authentication..."
        git config --global url."https://${GH_USER}:${GH_TOKEN}@github.com/".insteadOf "https://github.com/"
    fi

    # Check if repo exists
    if [ -d ~/llm-stylometry ]; then
    echo "Repository exists. Updating to latest version..."
    cd ~/llm-stylometry

    # Stash any local changes
    if ! git diff --quiet || ! git diff --cached --quiet; then
        echo "Stashing local changes..."
        git stash
    fi

    # Determine authentication method
    if [ "$AUTH_OPTION" = "1" ]; then
        # Try SSH first for option 1
        if ssh -o BatchMode=yes -o ConnectTimeout=5 git@github.com 2>&1 | grep -q "successfully authenticated"; then
            echo "Using SSH authentication..."
            git remote set-url origin git@github.com:ContextLab/llm-stylometry.git
        else
            echo "SSH key not found or not authorized."
            echo "Please ensure your SSH key is added to GitHub and try again."
            exit 1
        fi
    elif [ "$AUTH_OPTION" = "2" ]; then
        # Use HTTPS with token for option 2
        echo "Using HTTPS with Personal Access Token..."
        git remote set-url origin https://github.com/ContextLab/llm-stylometry.git

        if [ -z "$GH_TOKEN" ] || [ -z "$GH_USER" ]; then
            echo "GitHub credentials not provided. You'll be prompted during git operations."
            echo "Note: Use your Personal Access Token as the password."
        fi
    fi

    # Update repository
    git fetch origin
    git checkout main
    git pull origin main
    echo "Repository updated successfully"
else
    echo "Repository not found. Cloning..."

    if [ "$AUTH_OPTION" = "1" ]; then
        # Check if SSH key is available
        if ssh -o BatchMode=yes -o ConnectTimeout=5 git@github.com 2>&1 | grep -q "successfully authenticated"; then
            echo "Using SSH to clone repository..."
            cd ~
            git clone git@github.com:ContextLab/llm-stylometry.git
        else
            echo "SSH key not found or not authorized."
            echo "Please ensure your SSH key is added to GitHub and try again."
            exit 1
        fi
    elif [ "$AUTH_OPTION" = "2" ]; then
        echo "Using HTTPS to clone repository..."
        if [ -z "$GH_TOKEN" ] || [ -z "$GH_USER" ]; then
            echo "GitHub credentials not configured. You'll be prompted."
            echo "Username: Your GitHub username"
            echo "Password: Your Personal Access Token (NOT your GitHub password)"
            echo "Create a token at: https://github.com/settings/tokens"
            echo ""
        fi
        cd ~
        git clone https://github.com/ContextLab/llm-stylometry.git
    fi

    cd ~/llm-stylometry
    echo "Repository cloned successfully"
    fi
fi  # End of AUTH_OPTION check

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
cd ~/llm-stylometry
LOG_FILE=~/llm-stylometry/logs/training_$(date +%Y%m%d_%H%M%S).log
echo "Training started at \$(date)" | tee -a \$LOG_FILE

# Try to activate conda environment if it exists
if command -v conda &> /dev/null; then
    source \$(conda info --base)/etc/profile.d/conda.sh
    if conda env list | grep -q llm-stylometry; then
        echo "Activating conda environment: llm-stylometry" | tee -a \$LOG_FILE
        conda activate llm-stylometry
    fi
fi

# Check if Python is available
if ! command -v python &> /dev/null; then
    echo "Error: Python not found" | tee -a \$LOG_FILE
    exit 1
fi

# Run the training script
python code/generate_figures.py --train --no-confirm 2>&1 | tee -a \$LOG_FILE
TRAINSCRIPT

chmod +x /tmp/llm_train.sh
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