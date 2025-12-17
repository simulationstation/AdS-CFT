"""Fast end-to-end suite test."""

import subprocess
import sys
from pathlib import Path


def test_suite_fast_creates_reports(tmp_path):
    outdir = tmp_path / "outputs"
    cmd = [sys.executable, "-m", "holoop", "--suite", "--fast", "--outdir", str(outdir)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr

    assert (outdir / "results.json").exists()
    assert (outdir / "report.md").exists()
    plots = list((outdir / "distinguish").glob("distinguish_accuracy_vs_T_*.png"))
    assert plots, "No distinguish plots produced"
