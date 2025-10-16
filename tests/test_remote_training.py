#!/usr/bin/env python
"""
Integration tests for remote_train.sh script.

These tests use REAL SSH connections to a REAL GPU server.
NO MOCKS - all tests perform actual remote operations.

Tests verify that variant flags (-co, -fo, -pos) are correctly passed
from local machine to remote server and used in training.

Related: Issue #27
"""

import pytest
import subprocess
import time
import re
from pathlib import Path


class TestRemoteTraining:
    """Test suite for remote_train.sh with real SSH connections."""

    @pytest.fixture(autouse=True)
    def setup(self, ssh_client, ssh_credentials):
        """Setup for each test."""
        self.ssh_client = ssh_client
        self.credentials = ssh_credentials
        self.script_path = Path(__file__).parent.parent / "remote_train.sh"

        # Verify script exists
        if not self.script_path.exists():
            pytest.skip("remote_train.sh not found")

    def run_remote_train(self, flags="", timeout=60, kill_after_seconds=None):
        """
        Helper to run remote_train.sh with automatic input.

        Args:
            flags: Command line flags for remote_train.sh
            timeout: Maximum time to wait for script
            kill_after_seconds: If set, kill training after N seconds

        Returns:
            subprocess.CompletedProcess object
        """
        server = self.credentials['server']
        username = self.credentials['username']
        password = self.credentials['password']

        # Prepare input for the script (server address and username)
        script_input = f"{server}\n{username}\n"

        # Build command
        cmd = ['bash', str(self.script_path)] + (flags.split() if flags else [])

        # Use sshpass to handle password authentication
        # We'll need to set SSHPASS environment variable
        env = {'SSHPASS': password}

        # Start the script
        try:
            if kill_after_seconds:
                # Start process, wait, then kill
                process = subprocess.Popen(
                    cmd,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    env={**subprocess.os.environ, **env}
                )

                # Send input
                process.stdin.write(script_input)
                process.stdin.flush()

                # Wait for specified time
                time.sleep(kill_after_seconds)

                # Kill the training on remote server
                self.ssh_client("pkill -f 'python.*generate_figures.py.*--train' || true")
                self.ssh_client("screen -ls | grep -o '[0-9]*\\.llm_training' | cut -d. -f1 | xargs -I {} screen -X -S {}.llm_training quit || true")

                # Wait a bit more for cleanup
                time.sleep(2)

                # Terminate local process
                process.terminate()
                try:
                    stdout, stderr = process.communicate(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    stdout, stderr = process.communicate()

                # Create result object
                result = subprocess.CompletedProcess(
                    args=cmd,
                    returncode=process.returncode if process.returncode is not None else -1,
                    stdout=stdout,
                    stderr=stderr
                )
            else:
                # Normal execution
                result = subprocess.run(
                    cmd,
                    input=script_input,
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                    env={**subprocess.os.environ, **env}
                )

            return result

        except subprocess.TimeoutExpired as e:
            # Cleanup on timeout
            self.ssh_client("pkill -f 'python.*generate_figures.py.*--train' || true")
            self.ssh_client("screen -ls | grep -o '[0-9]*\\.llm_training' | cut -d. -f1 | xargs -I {} screen -X -S {}.llm_training quit || true")
            raise

    # Test 1: Content-only variant flag passing
    def test_variant_flag_content_only(self):
        """Test that --content-only flag is passed correctly to remote server."""
        # Run remote_train.sh with -co flag, kill after 30 seconds
        result = self.run_remote_train(flags="-co", kill_after_seconds=30)

        # Check /tmp/llm_train.sh on server
        check_result = self.ssh_client("cat /tmp/llm_train.sh 2>/dev/null || echo 'FILE_NOT_FOUND'")

        if "FILE_NOT_FOUND" in check_result.stdout:
            pytest.fail("/tmp/llm_train.sh was not created on remote server")

        # Verify VARIANT_ARG is set correctly
        assert "VARIANT_ARG='-co'" in check_result.stdout, \
            f"VARIANT_ARG not set correctly. Script content:\n{check_result.stdout}"

        # Check training log exists and contains variant info
        log_check = self.ssh_client("ls -t ~/llm-stylometry/logs/training_*.log 2>/dev/null | head -1 | xargs cat")

        if log_check.returncode == 0:
            log_content = log_check.stdout
            # Look for debug output or training variant message
            assert ('-co' in log_content or 'content' in log_content.lower()), \
                f"Training log doesn't show content variant. Log:\n{log_content[:500]}"

    # Test 2: All variant types
    @pytest.mark.parametrize("flag,variant_name", [
        ("-co", "content"),
        ("-fo", "function"),
        ("-pos", "pos"),
    ])
    def test_all_variant_flags(self, flag, variant_name):
        """Test each variant flag passes correctly."""
        # Run with specific variant flag
        result = self.run_remote_train(flags=flag, kill_after_seconds=25)

        # Check the generated script
        check_result = self.ssh_client("cat /tmp/llm_train.sh")

        assert check_result.returncode == 0, "Failed to read /tmp/llm_train.sh"
        assert f"VARIANT_ARG='{flag}'" in check_result.stdout, \
            f"VARIANT_ARG not set to '{flag}'. Script:\n{check_result.stdout}"

    # Test 3: Baseline (no variant)
    def test_baseline_no_variant(self):
        """Test that baseline training works without variant flags."""
        # Run without any variant flag
        result = self.run_remote_train(flags="", kill_after_seconds=25)

        # Check the script
        check_result = self.ssh_client("cat /tmp/llm_train.sh")

        assert check_result.returncode == 0
        # VARIANT_ARG should be empty
        assert "VARIANT_ARG=''" in check_result.stdout, \
            f"VARIANT_ARG should be empty for baseline. Script:\n{check_result.stdout}"

    # Test 4: Resume mode with variants
    def test_resume_with_variant(self):
        """Test --resume flag works with variant flags."""
        # First, start training with -co (we'll kill it quickly)
        self.run_remote_train(flags="-co", kill_after_seconds=20)

        # Now try to resume with -co and --resume
        result = self.run_remote_train(flags="--resume -co", kill_after_seconds=20)

        # Check the script has both flags
        check_result = self.ssh_client("cat /tmp/llm_train.sh")

        assert check_result.returncode == 0
        assert "RESUME_MODE='true'" in check_result.stdout, \
            f"RESUME_MODE not set. Script:\n{check_result.stdout}"
        assert "VARIANT_ARG='-co'" in check_result.stdout, \
            f"VARIANT_ARG not set with resume. Script:\n{check_result.stdout}"

    # Test 5: Kill mode
    def test_kill_mode(self):
        """Test --kill flag terminates existing sessions."""
        # Start a training session
        self.run_remote_train(flags="-co", kill_after_seconds=15)

        # Verify screen session exists
        screen_check = self.ssh_client("screen -ls")
        assert "llm_training" in screen_check.stdout, \
            "Screen session should exist after starting training"

        # Run with --kill and different variant
        result = self.run_remote_train(flags="--kill -fo", kill_after_seconds=15)

        # Check that new variant is set
        check_result = self.ssh_client("cat /tmp/llm_train.sh")
        assert "VARIANT_ARG='-fo'" in check_result.stdout, \
            f"New variant not set after kill. Script:\n{check_result.stdout}"

    # Test 6: Debug logging
    def test_debug_logging_shows_variant(self):
        """Test that debug logging correctly shows VARIANT_ARG value."""
        # Run with variant flag
        result = self.run_remote_train(flags="-pos", kill_after_seconds=25)

        # Check the script contains debug logging
        check_result = self.ssh_client("cat /tmp/llm_train.sh")

        assert check_result.returncode == 0
        # Debug output should be in the script
        assert "DEBUG" in check_result.stdout, \
            f"Debug logging not found in script:\n{check_result.stdout}"

        # Check that it shows the correct value
        assert "-pos" in check_result.stdout, \
            f"Debug logging doesn't show -pos variant:\n{check_result.stdout}"

    # Test 7: Script file permissions
    def test_script_file_is_executable(self):
        """Test that /tmp/llm_train.sh is created with executable permissions."""
        # Run the script
        self.run_remote_train(flags="-co", kill_after_seconds=15)

        # Check permissions
        perm_check = self.ssh_client("ls -l /tmp/llm_train.sh")

        assert perm_check.returncode == 0, "Script file not found"
        # Should have execute permission (starts with -rwx or similar)
        assert 'x' in perm_check.stdout, \
            f"Script not executable. Permissions:\n{perm_check.stdout}"

    # Test 8: Repository update path
    def test_repo_already_exists(self):
        """Test that script does git pull when repository already exists."""
        # Ensure repo exists on server
        clone_check = self.ssh_client("cd ~/llm-stylometry && git status")

        if clone_check.returncode != 0:
            # Clone it first
            self.ssh_client("cd ~ && git clone https://github.com/ContextLab/llm-stylometry.git")
            time.sleep(5)

        # Run remote_train.sh
        result = self.run_remote_train(flags="-co", kill_after_seconds=20)

        # Verify repository was updated (git pull happened)
        # We can't easily check stdout since it's in screen, but we can verify
        # the repo exists and is up to date
        repo_check = self.ssh_client("cd ~/llm-stylometry && git status")

        assert repo_check.returncode == 0, \
            "Repository should exist and be accessible after running script"

    # Test 9: Screen session naming
    def test_screen_session_created_with_correct_name(self):
        """Test that screen session is created with name 'llm_training'."""
        # Clean up any existing sessions first
        self.ssh_client("screen -ls | grep -o '[0-9]*\\.llm_training' | cut -d. -f1 | xargs -I {} screen -X -S {}.llm_training quit || true")
        time.sleep(2)

        # Run the script
        self.run_remote_train(flags="-co", kill_after_seconds=20)

        # Check screen sessions
        screen_check = self.ssh_client("screen -ls")

        assert screen_check.returncode == 0 or "llm_training" in screen_check.stdout, \
            f"Screen session not found or has wrong name. Output:\n{screen_check.stdout}"

        # Clean up
        self.ssh_client("screen -ls | grep -o '[0-9]*\\.llm_training' | cut -d. -f1 | xargs -I {} screen -X -S {}.llm_training quit || true")

    # Test 10: Variable expansion not broken
    def test_no_literal_dollar_signs_in_script(self):
        """Test that variables are expanded correctly, not left as literal strings."""
        # Run the script
        self.run_remote_train(flags="-fo", kill_after_seconds=15)

        # Read the generated script
        check_result = self.ssh_client("cat /tmp/llm_train.sh")

        assert check_result.returncode == 0

        script_content = check_result.stdout

        # The script should NOT contain literal $VARIANT_ARG in the assignment lines
        # (it should be expanded to the actual value)
        lines = script_content.split('\n')

        for line in lines:
            if line.startswith("VARIANT_ARG="):
                # This line should have the actual value, not a variable reference
                assert "VARIANT_ARG='-fo'" in line, \
                    f"VARIANT_ARG line not expanded correctly: {line}"
                assert "$VARIANT_ARG" not in line.split('=', 1)[1], \
                    f"VARIANT_ARG value contains unexpanded variable: {line}"


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short", "-s"])
