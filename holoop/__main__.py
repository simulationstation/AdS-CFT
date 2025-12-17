"""Entry point for holoop package."""

import argparse
from pathlib import Path

import numpy as np

from holoop.synth.datasets import generate_ads3_dataset


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


def main():
    parser = argparse.ArgumentParser(description="holoop synthetic toolkit")
    parser.add_argument(
        "--synth",
        action="store_true",
        help="Generate AdS3/CFT2 synthetic entanglement datasets.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs_holoop",
        help="Directory to store synthetic datasets and plots.",
    )
    args = parser.parse_args()

    if args.synth:
        _run_synth(Path(args.output_dir))
    else:
        print("holoop scaffold OK")


if __name__ == "__main__":
    main()
