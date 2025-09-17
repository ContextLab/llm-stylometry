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

# Ask about GitHub authentication method
echo
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

print_info "Connecting to $USERNAME@$SERVER_ADDRESS..."
echo

# Execute the remote script via SSH, passing environment variables
ssh -t "$USERNAME@$SERVER_ADDRESS" \
    AUTH_OPTION="$AUTH_OPTION" \
    GH_USER="$GH_USER" \
    GH_TOKEN="$GH_TOKEN" \
    'bash -s' << 'ENDSSH'
#!/bin/bash
set -e

echo "=================================================="
echo "Setting up LLM Stylometry on remote server"
echo "=================================================="
echo

# Handle different authentication options
if [ "$AUTH_OPTION" = "3" ]; then
    echo "Skipping repository update as requested..."
    if [ ! -d "$HOME/llm-stylometry" ]; then
        echo "Error: Repository not found at $HOME/llm-stylometry"
        echo "Please choose option 1 or 2 to clone the repository first."
        exit 1
    fi
    cd "$HOME/llm-stylometry"
else
    # Configure Git to use credential caching to avoid repeated auth prompts
    git config --global credential.helper cache
    git config --global credential.helper 'cache --timeout=3600'

    # Set up GitHub token authentication if provided
    if [ -n "$GH_TOKEN" ] && [ -n "$GH_USER" ]; then
        echo "Setting up GitHub token authentication..."
        git config --global url."https://${GH_USER}:${GH_TOKEN}@github.com/".insteadOf "https://github.com/"
    fi

    # Check if repo exists
    if [ -d "$HOME/llm-stylometry" ]; then
    echo "Repository exists. Updating to latest version..."
    cd "$HOME/llm-stylometry"

    # Stash any local changes
    if ! git diff --quiet || ! git diff --cached --quiet; then
        echo "Stashing local changes..."
        git stash
    fi

    # Check if we can use SSH (if SSH key is set up)
    if ssh -o BatchMode=yes -o ConnectTimeout=5 git@github.com 2>&1 | grep -q "successfully authenticated"; then
        echo "Using SSH authentication..."
        git remote set-url origin git@github.com:ContextLab/llm-stylometry.git
    else
        echo "Using HTTPS (you may be prompted for GitHub username/password or token)..."
        echo "Note: GitHub now requires personal access tokens instead of passwords."
        echo "Create one at: https://github.com/settings/tokens"
        git remote set-url origin https://github.com/ContextLab/llm-stylometry.git
    fi

    # Update repository
    git fetch origin
    git checkout main
    git pull origin main
    echo "Repository updated successfully"
else
    echo "Checking authentication method..."

    # Check if SSH key is available
    if ssh -o BatchMode=yes -o ConnectTimeout=5 git@github.com 2>&1 | grep -q "successfully authenticated"; then
        echo "Using SSH to clone repository..."
        cd "$HOME"
        git clone git@github.com:ContextLab/llm-stylometry.git
    else
        echo "Using HTTPS to clone repository..."
        echo "You will be prompted for GitHub credentials."
        echo "Note: GitHub requires a personal access token (not password)."
        echo "Create one at: https://github.com/settings/tokens"
        echo ""
        cd "$HOME"
        git clone https://github.com/ContextLab/llm-stylometry.git
    fi

    cd "$HOME/llm-stylometry"
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

# Start training in screen (use --no-confirm flag for non-interactive mode)
screen -dmS llm_training bash -c "
    cd $HOME/llm-stylometry
    echo \"Training started at \$(date)\" | tee -a $LOG_FILE
    python code/generate_figures.py --train --no-confirm 2>&1 | tee -a $LOG_FILE
    echo \"Training completed at \$(date)\" | tee -a $LOG_FILE
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