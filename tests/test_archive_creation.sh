#!/bin/bash

# Test script for create_model_archive.sh
# NO MOCKS - tests use real model files and real tar operations

set -e

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

print_test() { echo -e "\n${YELLOW}[TEST]${NC} $1"; }
print_pass() { echo -e "${GREEN}[PASS]${NC} $1"; }
print_fail() { echo -e "${RED}[FAIL]${NC} $1"; exit 1; }

# Test counters
TESTS_RUN=0
TESTS_PASSED=0

run_test() {
    ((TESTS_RUN++))
    "$@" && ((TESTS_PASSED++))
}

# Setup: Create test directory
TEST_DIR="/tmp/test_archive_$$"
mkdir -p "$TEST_DIR"
cd "$(dirname "$0")/.."  # Go to project root
PROJECT_ROOT="$(pwd)"

print_test "Test 1: Script exists and is executable"
if [ -x "./create_model_archive.sh" ]; then
    print_pass "Script is executable"
else
    print_fail "Script not found or not executable"
fi

print_test "Test 2: Help option works"
if ./create_model_archive.sh --help | grep -q "Usage:"; then
    print_pass "Help option works"
else
    print_fail "Help option failed"
fi

print_test "Test 3: Create baseline archive with 2 models (real data)"
# Find first 2 baseline models
MODELS=$(find models/ -maxdepth 1 -type d -name "*_tokenizer=gpt2_seed=*" ! -name "*variant=*" | head -2)
MODEL_COUNT=$(echo "$MODELS" | wc -l | tr -d ' ')

if [ "$MODEL_COUNT" -lt 2 ]; then
    print_fail "Need at least 2 baseline models for testing"
fi

# Create temporary test models directory with just 2 models
TEST_MODELS="$TEST_DIR/models"
mkdir -p "$TEST_MODELS"

for model in $MODELS; do
    model_name=$(basename "$model")
    mkdir -p "$TEST_MODELS/$model_name"

    # Copy only weight files
    cp "$model/model.safetensors" "$TEST_MODELS/$model_name/" 2>/dev/null || print_fail "Missing model.safetensors in $model"
    cp "$model/training_state.pt" "$TEST_MODELS/$model_name/" 2>/dev/null || print_fail "Missing training_state.pt in $model"
done

# Create archive from test models
cd "$TEST_DIR"
"$PROJECT_ROOT/create_model_archive.sh" -b -o . --force > /dev/null 2>&1

if [ -f "model_weights_baseline.tar.gz" ]; then
    print_pass "Archive created"
else
    print_fail "Archive not created"
fi

print_test "Test 4: Archive has correct structure"
# List archive contents
ARCHIVE_CONTENTS=$(tar -tzf model_weights_baseline.tar.gz)

# Check for models/ prefix
if echo "$ARCHIVE_CONTENTS" | grep -q "^models/"; then
    print_pass "Archive has correct directory structure"
else
    print_fail "Archive missing models/ prefix"
fi

# Count files (should be 4: 2 models × 2 files each)
FILE_COUNT=$(echo "$ARCHIVE_CONTENTS" | wc -l | tr -d ' ')
if [ "$FILE_COUNT" -eq 4 ]; then
    print_pass "Archive contains correct number of files ($FILE_COUNT)"
else
    print_fail "Expected 4 files, found $FILE_COUNT"
fi

print_test "Test 5: Archive only contains weight files"
# Check no .json or .csv files
if echo "$ARCHIVE_CONTENTS" | grep -q "\.json\|\.csv"; then
    print_fail "Archive contains config/log files (should only have weights)"
else
    print_pass "Archive only contains weight files"
fi

# Check has .safetensors and .pt files
if echo "$ARCHIVE_CONTENTS" | grep -q "\.safetensors"; then
    print_pass "Archive contains .safetensors files"
else
    print_fail "Archive missing .safetensors files"
fi

if echo "$ARCHIVE_CONTENTS" | grep -q "training_state\.pt"; then
    print_pass "Archive contains training_state.pt files"
else
    print_fail "Archive missing training_state.pt files"
fi

print_test "Test 6: Checksum file generated"
if [ -f "model_weights_baseline.tar.gz.sha256" ]; then
    print_pass "Checksum file created"
else
    print_fail "Checksum file not created"
fi

print_test "Test 7: Checksum is valid"
# Verify checksum format
CHECKSUM=$(cat model_weights_baseline.tar.gz.sha256)
if echo "$CHECKSUM" | grep -qE "^[a-f0-9]{64}"; then
    print_pass "Checksum has valid format"
else
    print_fail "Invalid checksum format: $CHECKSUM"
fi

print_test "Test 8: Archive integrity check"
if tar -tzf model_weights_baseline.tar.gz > /dev/null 2>&1; then
    print_pass "Archive integrity verified"
else
    print_fail "Archive is corrupted"
fi

print_test "Test 9: Extract and verify contents"
EXTRACT_DIR="$TEST_DIR/extracted"
mkdir -p "$EXTRACT_DIR"
cd "$EXTRACT_DIR"
tar -xzf ../model_weights_baseline.tar.gz

# Verify extracted structure
for model in $MODELS; do
    model_name=$(basename "$model")

    if [ -f "models/$model_name/model.safetensors" ]; then
        print_pass "Extracted model.safetensors for $model_name"
    else
        print_fail "Missing model.safetensors for $model_name after extraction"
    fi

    if [ -f "models/$model_name/training_state.pt" ]; then
        print_pass "Extracted training_state.pt for $model_name"
    else
        print_fail "Missing training_state.pt for $model_name after extraction"
    fi
done

print_test "Test 10: Verify checksum matches"
cd "$TEST_DIR"
if [[ "$OSTYPE" == "darwin"* ]]; then
    COMPUTED=$(shasum -a 256 model_weights_baseline.tar.gz | awk '{print $1}')
else
    COMPUTED=$(sha256sum model_weights_baseline.tar.gz | awk '{print $1}')
fi

EXPECTED=$(awk '{print $1}' model_weights_baseline.tar.gz.sha256)

if [ "$COMPUTED" = "$EXPECTED" ]; then
    print_pass "Checksum verification passed"
else
    print_fail "Checksum mismatch: expected $EXPECTED, got $COMPUTED"
fi

# Cleanup
cd /
rm -rf "$TEST_DIR"

# Summary
echo
echo "======================================"
echo "Archive Creation Tests Complete"
echo "======================================"
echo "Tests run: $TESTS_RUN"
echo "Tests passed: $TESTS_PASSED"

if [ "$TESTS_RUN" -eq "$TESTS_PASSED" ]; then
    echo -e "${GREEN}All tests passed!${NC}"
    exit 0
else
    echo -e "${RED}Some tests failed${NC}"
    exit 1
fi
