"""Experiment wrappers for the distinguishability game."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt

from .game import ObserverConfig, run_distinguish


def _geomspace(start: float, stop: float, num: int) -> List[float]:
    if num <= 1:
        return [start]
    log_start = math.log10(start)
    log_stop = math.log10(stop)
    step = (log_stop - log_start) / (num - 1)
    return [10 ** (log_start + i * step) for i in range(num)]


_DEF_T_VALUES = _geomspace(8.0, 4096.0, num=20)
_DEF_TRIALS = 200


def _plot_accuracy(result: Dict[str, object], outpath: Path) -> None:
    t_values: List[float] = result["T_values"]
    acc: List[float] = result["accuracy"]
    plt.figure(figsize=(6, 4))
    plt.plot(t_values, acc, label=result["observer"]["name"])
    plt.plot(t_values, [0.5 for _ in t_values])
    plt.xlabel("Window start T")
    plt.ylabel("Accuracy")
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()


def run_default_distinguish(output_dir: Path, fast: bool = False, seed: int = 0) -> Dict[str, object]:
    """Run a small suite of distinguishability experiments."""

    output_dir.mkdir(parents=True, exist_ok=True)
    T_values = _geomspace(8.0, 1024.0, num=10 if fast else 20)
    n_trials = 50 if fast else _DEF_TRIALS

    observers = [
        ObserverConfig(
            name="stream_max",
            sample_budget=128 if not fast else 64,
            memory_budget=8,
            threshold=0.01,
            statistic="max",
            observer_type="streaming",
        ),
        ObserverConfig(
            name="buffer_median",
            sample_budget=128 if not fast else 64,
            memory_budget=32,
            threshold=0.005,
            statistic="median",
            observer_type="buffer",
        ),
    ]

    results = {}
    for idx, obs in enumerate(observers):
        res = run_distinguish(
            T_values=T_values,
            n_trials=n_trials,
            observer=obs,
            sigma=0.02,
            seed=seed + idx,
        )
        plot_path = output_dir / f"distinguish_accuracy_vs_T_{obs.name}.png"
        _plot_accuracy(res, plot_path)
        results[obs.name] = res

    out_json = output_dir / "distinguish_results.json"
    out_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results


def sweep_observer_budgets(output_dir: Path, fast: bool = False) -> Dict[str, object]:
    """Sweep sample and memory budgets."""

    output_dir.mkdir(parents=True, exist_ok=True)
    budgets_S = [32, 64] if fast else [32, 128]
    budgets_M = [8, 16] if fast else [8, 32]
    T_values = _geomspace(16.0, 1024.0, num=8 if fast else 16)
    n_trials = 40 if fast else 120

    grid_results = {}
    seed = 0
    for s in budgets_S:
        for m in budgets_M:
            obs = ObserverConfig(
                name=f"stream_max_S{s}_M{m}",
                sample_budget=s,
                memory_budget=m,
                threshold=0.015,
                statistic="max",
                observer_type="streaming",
            )
            res = run_distinguish(T_values, n_trials, obs, sigma=0.02, seed=seed)
            grid_results[obs.name] = res
            seed += 1

    out_json = output_dir / "budget_sweep_results.json"
    out_json.write_text(json.dumps(grid_results, indent=2), encoding="utf-8")
    return grid_results


def sweep_noise(output_dir: Path, fast: bool = False) -> Dict[str, object]:
    """Sweep over noise strengths."""

    output_dir.mkdir(parents=True, exist_ok=True)
    sigmas = [0.0, 0.05, 0.1] if not fast else [0.0, 0.1]
    T_values = _geomspace(16.0, 2048.0, num=8 if fast else 16)
    n_trials = 40 if fast else 120

    results = {}
    obs = ObserverConfig(
        name="stream_noise",
        sample_budget=96 if not fast else 64,
        memory_budget=12,
        threshold=0.012,
        statistic="max",
        observer_type="streaming",
    )
    for idx, sigma in enumerate(sigmas):
        res = run_distinguish(T_values, n_trials, obs, sigma=sigma, seed=idx)
        results[f"sigma_{sigma}"] = res

    out_json = output_dir / "noise_sweep_results.json"
    out_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results
