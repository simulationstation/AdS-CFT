"""Markdown report generation for holoop demos."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List


def _format_distinguish_table(entries: Dict[str, Dict[str, object]]) -> str:
    lines = ["| Observer | Late-time accuracy | Distinguishable? |", "|---|---|---|"]
    for name, res in sorted(entries.items()):
        acc = res.get("late_time_accuracy", 0.0)
        flag = "yes" if res.get("late_time_distinguishable") else "no"
        lines.append(f"| {name} | {acc:.3f} | {flag} |")
    return "\n".join(lines)


def _format_ops_table(results: Dict[str, object]) -> str:
    lines = ["| Demo | Status |", "|---|---|"]
    for key in ["constant", "powerlaw", "pulsetrain", "combined_status"]:
        status = results.get(key, "not_run") if isinstance(results, dict) else "not_run"
        lines.append(f"| {key} | {status} |")
    return "\n".join(lines)


def _format_qec_table(results: Dict[str, object]) -> str:
    lines = ["| Scenario | Success prob |", "|---|---|"]
    if isinstance(results, dict):
        lines.append(f"| n=9, p=0.2 | {results.get('n9_p02', 'n/a')} |")
        lines.append(f"| n=1, p=0.2 | {results.get('n1_p02', 'n/a')} |")
    else:
        lines.append("| n=9, p=0.2 | n/a |")
        lines.append("| n=1, p=0.2 | n/a |")
    return "\n".join(lines)


def write_report(outdir: Path, merged_results: Dict[str, object]) -> None:
    """Write a consolidated markdown report."""

    outdir.mkdir(parents=True, exist_ok=True)
    lines: List[str] = []
    lines.append("# holoop consolidated report")
    lines.append("")
    lines.append("## What this project tests")
    lines.append(
        "A collection of small operational diagnostics: synthetic AdS3/CFT2 data generation and fitting, toy operations demos, quantum error correction sketches, and a bounded-observer distinguishability game contrasting continuous vs pulsed activity.",
    )
    lines.append("")

    lines.append("## Operations demos")
    ops_results = merged_results.get("ops_demo", {}) if isinstance(merged_results, dict) else {}
    lines.append(_format_ops_table(ops_results))
    lines.append("")

    lines.append("## QEC demo")
    qec_results = merged_results.get("qec_demo", {}) if isinstance(merged_results, dict) else {}
    lines.append(_format_qec_table(qec_results))
    lines.append("")

    lines.append("## Distinguishability demo")
    dist_results = merged_results.get("distinguish", {}) if isinstance(merged_results, dict) else {}
    lines.append(_format_distinguish_table(dist_results if isinstance(dist_results, dict) else {}))
    lines.append("")

    lines.append("## Caveats")
    lines.append("- These are toy models and diagnostics rather than predictive simulations.")
    lines.append("- Identifiability depends on the availability of inference routines and data quality.")
    lines.append("- Distinguishability depends on observer budgets, sampling strategy, and observation noise.")
    lines.append("")

    report_path = outdir / "report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")

    # Also store merged JSON for convenience.
    results_path = outdir / "results.json"
    with results_path.open("w", encoding="utf-8") as fp:
        json.dump(merged_results, fp, indent=2)
