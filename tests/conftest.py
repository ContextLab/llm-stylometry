#!/usr/bin/env python
"""
Pytest configuration and shared fixtures.

This file contains fixtures for remote SSH testing with real server connections.
NO MOCKS - all SSH tests use actual connections to test servers.
"""

import json
import os
import pytest
import subprocess
import time
from pathlib import Path


@pytest.fixture(scope="session")
def ssh_credentials():
    """
    Load SSH credentials from .ssh/credentials.json.

    Returns dict with keys: server, username, password
    Skips tests if credentials file not found.
    """
    cred_path = Path(__file__).parent.parent / ".ssh" / "credentials.json"

    if not cred_path.exists():
        pytest.skip("SSH credentials file not found at .ssh/credentials.json")

    with open(cred_path, 'r') as f:
        credentials = json.load(f)

    # Validate required fields
    required_fields = ['server', 'username', 'password']
    for field in required_fields:
        if field not in credentials:
            pytest.skip(f"Missing required field '{field}' in credentials file")

    return credentials


@pytest.fixture(scope="session")
def ssh_connection_test(ssh_credentials):
    """
    Verify SSH connection works before running tests.

    Uses sshpass for password authentication.
    Skips tests if connection fails.
    """
    server = ssh_credentials['server']
    username = ssh_credentials['username']
    password = ssh_credentials['password']

    # Check if sshpass is installed
    try:
        subprocess.run(['which', 'sshpass'], check=True, capture_output=True)
    except subprocess.CalledProcessError:
        pytest.skip("sshpass not installed. Install with: brew install sshpass (macOS) or apt-get install sshpass (Linux)")

    # Test connection
    try:
        cmd = [
            'sshpass', '-p', password,
            'ssh', '-o', 'StrictHostKeyChecking=no',
            '-o', 'ConnectTimeout=10',
            f'{username}@{server}',
            'echo "Connection successful"'
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)

        if result.returncode != 0:
            pytest.skip(f"SSH connection failed: {result.stderr}")

        if "Connection successful" not in result.stdout:
            pytest.skip(f"SSH connection test did not produce expected output: {result.stdout}")

    except subprocess.TimeoutExpired:
        pytest.skip("SSH connection timed out")
    except Exception as e:
        pytest.skip(f"SSH connection error: {e}")

    return True


@pytest.fixture
def ssh_client(ssh_credentials, ssh_connection_test):
    """
    Provide SSH connection helper for tests.

    Returns a function that executes commands via SSH.
    Uses real SSH connection, no mocks.
    """
    server = ssh_credentials['server']
    username = ssh_credentials['username']
    password = ssh_credentials['password']

    def run_ssh_command(command, timeout=30):
        """
        Execute a command via SSH and return the result.

        Args:
            command: Command to execute on remote server
            timeout: Timeout in seconds (default 30)

        Returns:
            subprocess.CompletedProcess object
        """
        cmd = [
            'sshpass', '-p', password,
            'ssh', '-o', 'StrictHostKeyChecking=no',
            '-o', 'ConnectTimeout=10',
            f'{username}@{server}',
            command
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout
        )

        return result

    return run_ssh_command


@pytest.fixture
def test_workspace(ssh_client):
    """
    Create and cleanup unique test workspace on remote server.

    Yields the workspace path on remote server.
    Cleans up after test completes.
    """
    # Create unique workspace
    timestamp = int(time.time())
    workspace = f"~/llm-stylometry-test-{timestamp}"

    # Create workspace directory
    result = ssh_client(f"mkdir -p {workspace}")
    if result.returncode != 0:
        pytest.fail(f"Failed to create test workspace: {result.stderr}")

    yield workspace

    # Cleanup: Remove workspace
    ssh_client(f"rm -rf {workspace}")

    # Cleanup: Kill any screen sessions from this test
    ssh_client("screen -ls | grep -o '[0-9]*\\.llm_training' | cut -d. -f1 | xargs -I {} screen -X -S {}.llm_training quit || true")

    # Cleanup: Kill any remaining training processes
    ssh_client("pkill -f 'python.*generate_figures.py.*--train' || true")
