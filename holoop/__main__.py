"""Entry point for holoop package."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np

from holoop.distinguish.experiments import run_default_distinguish
from holoop.infer.inverse_ads3 import FitResult, fit_btz, fit_vacuum, load_dataset
from holoop.report import write_report
from holoop.synth.ads3 import btz_entropy, vacuum_entropy
from holoop.synth.datasets import generate_ads3_dataset


_DEF_OUTPUT_DIR = Path("outputs_holoop")


def _run_synth(output_dir: Path, fast: bool = False) -> Dict[str, object]:
    lengths = np.linspace(0.1, 3.0, 30 if fast else 50)
    central_charge = 12.0
    epsilon = 0.05
    s0 = 0.0
    beta = 4.0

    generate_ads3_dataset(
        mode="vacuum",
        lengths=lengths,
        central_charge=central_charge,
        epsilon=epsilon,
        s0=s0,
        beta=None,
        seed=0,
        output_dir=output_dir,
    )

    generate_ads3_dataset(
        mode="btz",
        lengths=lengths,
        central_charge=central_charge,
        epsilon=epsilon,
        s0=s0,
        beta=beta,
        seed=0,
        output_dir=output_dir,
    )

    print(f"Synthetic AdS3 datasets written to {output_dir}")
    return {"status": "completed", "count": len(lengths)}


def _write_fit_artifacts(dataset_path: Path, fit: FitResult, metadata: dict, model_values: np.ndarray) -> None:
    base = dataset_path.stem
    output_dir = dataset_path.parent

    json_path = output_dir / f"{base}_fit.json"
    with json_path.open("w", encoding="utf-8") as fp:
        json.dump(
            {
                "model": fit.model,
                "params": fit.params,
                "residual": fit.residual,
                "metadata": metadata,
            },
            fp,
            indent=2,
        )

    lengths, entropies, _ = load_dataset(dataset_path)
    plt.figure(figsize=(6, 4))
    plt.plot(lengths, entropies, "o", label="data")
    plt.plot(lengths, model_values, "-", label="fit")
    plt.xlabel("Interval length l")
    plt.ylabel("Entanglement entropy S(l)")
    plt.title(f"Fit for {base}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / f"{base}_fit.png", dpi=150)
    plt.close()


def _run_infer(output_dir: Path) -> Dict[str, object]:
    dataset_paths = sorted(output_dir.glob("*.json"))
    if not dataset_paths:
        print(f"No datasets found in {output_dir}")
        return {"status": "not_run"}

    statuses = {}
    for dataset_path in dataset_paths:
        lengths, entropies, metadata = load_dataset(dataset_path)
        mode = str(metadata.get("mode", "")).lower()
        epsilon = float(metadata.get("epsilon", 0.0))
        if epsilon <= 0:
            print(f"Skipping {dataset_path}: missing epsilon in metadata")
            continue

        if mode == "vacuum":
            fit = fit_vacuum(lengths, entropies, epsilon)
            model_values = vacuum_entropy(lengths, fit.params["central_charge"], epsilon, fit.params["S0"])
        elif mode == "btz":
            fit = fit_btz(lengths, entropies, epsilon)
            params = fit.params
            model_values = btz_entropy(lengths, params["central_charge"], epsilon, params["beta"], params["S0"])
        else:
            print(f"Skipping {dataset_path}: unknown mode {mode}")
            continue

        _write_fit_artifacts(dataset_path, fit, metadata, model_values)
        statuses[dataset_path.name] = fit.params
        print(f"Fitted {dataset_path.name}: {fit.params}")

    return {"status": "completed", "fits": statuses}


def _run_ops_demo() -> Dict[str, object]:
    return {"combined_status": "not_run", "constant": "not_run", "powerlaw": "not_run", "pulsetrain": "not_run"}


def _run_qec_demo() -> Dict[str, object]:
    return {"n9_p02": "n/a", "n1_p02": "n/a"}


def _run_distinguish_demo(output_dir: Path, fast: bool = False) -> Dict[str, object]:
    results = run_default_distinguish(output_dir / "distinguish", fast=fast)
    return results


def _run_suite(output_dir: Path, fast: bool = False) -> Dict[str, object]:
    merged: Dict[str, object] = {}
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        merged["synth"] = _run_synth(output_dir, fast=fast)
    except Exception as exc:  # pragma: no cover - defensive path
        print(f"Synthesis failed: {exc}")
        merged["synth"] = {"status": "failed", "error": str(exc)}

    try:
        merged["infer"] = _run_infer(output_dir)
    except Exception as exc:  # pragma: no cover - defensive path
        print(f"Inference failed: {exc}")
        merged["infer"] = {"status": "failed", "error": str(exc)}

    merged["ops_demo"] = _run_ops_demo()
    merged["qec_demo"] = _run_qec_demo()

    try:
        merged["distinguish"] = _run_distinguish_demo(output_dir, fast=fast)
    except Exception as exc:  # pragma: no cover - defensive path
        print(f"Distinguish demo failed: {exc}")
        merged["distinguish"] = {"status": "failed", "error": str(exc)}

    write_report(output_dir, merged)
    print(f"Suite completed. Results in {output_dir}")
    return merged


def main():
    parser = argparse.ArgumentParser(description="holoop synthetic toolkit")
    parser.add_argument(
        "--synth",
        action="store_true",
        help="Generate AdS3/CFT2 synthetic entanglement datasets.",
    )
    parser.add_argument(
        "--infer",
        action="store_true",
        help="Infer AdS3/CFT2 parameters from datasets in outputs_holoop/",
    )
    parser.add_argument(
        "--distinguish_demo",
        action="store_true",
        help="Run the bounded-observer distinguishability demo.",
    )
    parser.add_argument(
        "--suite",
        action="store_true",
        help="Run the full demo suite and write a consolidated report.",
    )
    parser.add_argument("--fast", action="store_true", help="Use faster, coarse settings.")
    parser.add_argument(
        "--output-dir",
        dest="output_dir",
        default=str(_DEF_OUTPUT_DIR),
        help="Directory to store artifacts.",
    )
    parser.add_argument(
        "--outdir",
        dest="output_dir_alias",
        default=None,
        help="Alias for --output-dir.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir_alias or args.output_dir)
    if args.suite:
        _run_suite(output_dir, fast=args.fast)
    elif args.distinguish_demo:
        _run_distinguish_demo(output_dir, fast=args.fast)
    elif args.synth:
        _run_synth(output_dir, fast=args.fast)
    elif args.infer:
        _run_infer(output_dir)
    else:
        print("holoop scaffold OK")


if __name__ == "__main__":
    main()
