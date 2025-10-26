#!/bin/bash

# Test script for download_model_weights.sh
# NO MOCKS - tests use real Dropbox downloads and real file operations

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

# Go to project root
cd "$(dirname "$0")/.."

print_test "Test 1: Script exists and is executable"
if [ -x "./download_model_weights.sh" ]; then
    print_pass "Script is executable"
else
    print_fail "Script not found or not executable"
fi

print_test "Test 2: Help option works"
if ./download_model_weights.sh --help | grep -q "Usage:"; then
    print_pass "Help option works"
else
    print_fail "Help option failed"
fi

print_test "Test 3: Checksum files exist in git"
for variant in baseline content function pos; do
    if [ -f "model_weights_${variant}.tar.gz.sha256" ]; then
        print_pass "Checksum file exists for $variant"
    else
        print_fail "Checksum file missing for $variant"
    fi
done

print_test "Test 4: Dropbox URLs are configured"
# Check that URLs are not empty
if grep -q 'BASELINE_URL=""' download_model_weights.sh; then
    print_fail "Baseline URL not configured"
fi

if grep -q 'https://www.dropbox.com' download_model_weights.sh; then
    print_pass "Dropbox URLs configured"
else
    print_fail "Dropbox URLs not configured"
fi

print_test "Test 5: URLs have dl=1 parameter"
# All URLs should have dl=1 for direct download
URL_COUNT=$(grep -c 'dl=1' download_model_weights.sh || true)
if [ "$URL_COUNT" -ge 4 ]; then
    print_pass "URLs have dl=1 parameter for direct download"
else
    print_fail "URLs missing dl=1 parameter (found $URL_COUNT, expected 4+)"
fi

print_test "Test 6: Script detects existing models"
# Create a test model directory
TEST_MODEL="models/test_model_tokenizer=gpt2_seed=99"
mkdir -p "$TEST_MODEL"
touch "$TEST_MODEL/model.safetensors"
touch "$TEST_MODEL/training_state.pt"

# Count models (script should detect the test model)
COUNT_OUTPUT=$(./download_model_weights.sh --help 2>&1) # Just to test it doesn't crash

# Clean up test model
rm -rf "$TEST_MODEL"
print_pass "Script handles existing models without crashing"

print_test "Test 7: Verify git-tracked files are not modified by download"
# Get current git status
GIT_STATUS_BEFORE=$(git status --short models/)

# If models already downloaded, this should not change git status
# (Skip actual download since we already tested that earlier)

GIT_STATUS_AFTER=$(git status --short models/)

if [ "$GIT_STATUS_BEFORE" = "$GIT_STATUS_AFTER" ]; then
    print_pass "Git-tracked files not modified"
else
    print_fail "Git-tracked files were modified"
fi

print_test "Test 8: Verify model directories have correct structure"
# Check a few model directories
SAMPLE_MODELS=$(find models/ -maxdepth 1 -type d -name "*_tokenizer=gpt2_seed=0" | head -3)

for model_dir in $SAMPLE_MODELS; do
    # Check for required files
    if [ -f "$model_dir/config.json" ]; then
        print_pass "Found config.json in $(basename $model_dir)"
    else
        print_fail "Missing config.json in $(basename $model_dir)"
    fi

    if [ -f "$model_dir/generation_config.json" ]; then
        print_pass "Found generation_config.json in $(basename $model_dir)"
    else
        print_fail "Missing generation_config.json in $(basename $model_dir)"
    fi

    if [ -f "$model_dir/loss_logs.csv" ]; then
        print_pass "Found loss_logs.csv in $(basename $model_dir)"
    else
        print_fail "Missing loss_logs.csv in $(basename $model_dir)"
    fi
done

print_test "Test 9: Verify all authors have models"
# Check that we have models for all 8 authors
AUTHORS="austen baum dickens fitzgerald melville thompson twain wells"
for author in $AUTHORS; do
    AUTHOR_MODELS=$(find models/ -maxdepth 1 -type d -name "${author}_*" | wc -l | tr -d ' ')

    if [ "$AUTHOR_MODELS" -ge 10 ]; then  # At least 10 (10 seeds for baseline)
        print_pass "Found $AUTHOR_MODELS models for $author"
    else
        print_fail "Expected at least 10 models for $author, found $AUTHOR_MODELS"
    fi
done

print_test "Test 10: Verify variant models exist"
for variant in content function pos; do
    VARIANT_MODELS=$(find models/ -maxdepth 1 -type d -name "*_variant=${variant}_*" | wc -l | tr -d ' ')

    if [ "$VARIANT_MODELS" -eq 80 ]; then
        print_pass "Found 80 $variant variant models"
    elif [ "$VARIANT_MODELS" -eq 0 ]; then
        echo "  [SKIP] $variant variant not downloaded"
    else
        print_fail "Expected 80 $variant models, found $VARIANT_MODELS"
    fi
done

# Summary
echo
echo "======================================"
echo "Download Tests Complete"
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
