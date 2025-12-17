"""Synthetic dataset generation utilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, Tuple

import matplotlib.pyplot as plt
import numpy as np

from .ads3 import btz_entropy, vacuum_entropy


def _ensure_output_dir(output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _serialize_dataset(lengths: np.ndarray, entropies: np.ndarray, metadata: Dict) -> Dict:
    return {
        "lengths": lengths.tolist(),
        "entanglement_entropy": entropies.tolist(),
        "metadata": metadata,
    }


def generate_ads3_dataset(
    mode: str,
    lengths: Iterable[float],
    central_charge: float,
    epsilon: float,
    s0: float = 0.0,
    beta: float | None = None,
    seed: int = 123,
    output_dir: str | Path = "outputs_holoop",
    save_json: bool = True,
    save_plot: bool = True,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """Generate a synthetic AdS3 entanglement entropy dataset.

    Parameters
    ----------
    mode:
        Either ``"vacuum"`` or ``"btz"``.
    lengths:
        Interval lengths to evaluate.
    central_charge, epsilon, s0, beta:
        Parameters for the entanglement formulas. ``beta`` is required for BTZ.
    seed:
        RNG seed to ensure deterministic generation.
    output_dir:
        Directory for artifacts when ``save_json`` or ``save_plot`` are enabled.
    """

    rng = np.random.default_rng(seed)
    lengths = np.asarray(lengths, dtype=float)
    if np.any(lengths <= 0):
        raise ValueError("lengths must be positive")

    mode_normalized = mode.strip().lower()
    metadata: Dict[str, float | int | str | None] = {
        "mode": mode_normalized,
        "central_charge": central_charge,
        "epsilon": epsilon,
        "s0": s0,
        "beta": beta,
        "seed": seed,
    }

    if mode_normalized == "vacuum":
        entropies = vacuum_entropy(lengths, central_charge, epsilon, s0)
    elif mode_normalized == "btz":
        if beta is None:
            raise ValueError("beta is required for BTZ datasets")
        entropies = btz_entropy(lengths, central_charge, epsilon, beta, s0)
    else:
        raise ValueError("mode must be 'vacuum' or 'btz'")

    # Optional tiny deterministic jitter to reflect synthetic nature while remaining reproducible.
    jitter = rng.normal(scale=0.0, size=lengths.shape)
    entropies = entropies + jitter

    output_path = _ensure_output_dir(Path(output_dir))
    dataset_name = f"ads3_{mode_normalized}"

    artifact = _serialize_dataset(lengths, entropies, metadata)
    if save_json:
        json_path = output_path / f"{dataset_name}.json"
        with json_path.open("w", encoding="utf-8") as fp:
            json.dump(artifact, fp, indent=2)

    if save_plot:
        plt.figure(figsize=(6, 4))
        plt.plot(lengths, entropies, marker="o", linestyle="-", label=mode_normalized.upper())
        plt.xlabel("Interval length l")
        plt.ylabel("Entanglement entropy S(l)")
        plt.title(f"AdS3/CFT2 {mode_normalized} entanglement")
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_path / f"{dataset_name}.png", dpi=150)
        plt.close()

    return lengths, entropies, metadata
