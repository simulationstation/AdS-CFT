"""Test holoop scaffold."""

import subprocess
import sys


def test_holoop_runs_successfully():
    """Test that python -m holoop exits successfully and prints the expected message."""
    result = subprocess.run(
        [sys.executable, "-m", "holoop"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "holoop scaffold OK" in result.stdout
