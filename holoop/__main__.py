"""Entry point for holoop package."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from holoop.infer.inverse_ads3 import FitResult, fit_btz, fit_vacuum, load_dataset
from holoop.synth.ads3 import btz_entropy, vacuum_entropy
from holoop.synth.datasets import generate_ads3_dataset


_DEF_OUTPUT_DIR = Path("outputs_holoop")


def _run_synth(output_dir: Path) -> None:
    lengths = np.linspace(0.1, 3.0, 50)
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


def _run_infer(output_dir: Path) -> None:
    dataset_paths = sorted(output_dir.glob("*.json"))
    if not dataset_paths:
        print(f"No datasets found in {output_dir}")
        return

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
        print(f"Fitted {dataset_path.name}: {fit.params}")


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
        "--output-dir",
        default=str(_DEF_OUTPUT_DIR),
        help="Directory to store synthetic datasets and plots.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    if args.synth:
        _run_synth(output_dir)
    elif args.infer:
        _run_infer(output_dir)
    else:
        print("holoop scaffold OK")


if __name__ == "__main__":
    main()
