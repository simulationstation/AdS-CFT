"""Robustness helpers for inference."""
from __future__ import annotations

import random
import statistics
from dataclasses import dataclass
from typing import Callable, Iterable, List, Sequence


@dataclass
class NoiseResult:
    noise_level: float
    errors: dict


ArrayLike = Iterable[float]


def add_noise(values: ArrayLike, scale: float, *, rng: random.Random | None = None) -> list[float]:
    """Add Gaussian noise with standard deviation ``scale``."""

    rng = rng or random.Random()
    return [float(v) + rng.gauss(0.0, scale) for v in values]


def drop_points(
    ell: ArrayLike,
    S: ArrayLike,
    drop_fraction: float,
    *,
    rng: random.Random | None = None,
) -> tuple[list[float], list[float]]:
    """Randomly drop a fraction of points from paired ``ell``/``S`` arrays."""

    rng = rng or random.Random()
    ell_arr = [float(x) for x in ell]
    S_arr = [float(x) for x in S]
    n = len(ell_arr)
    keep = max(1, int(round(n * (1.0 - drop_fraction))))
    indices = rng.sample(range(n), k=keep)
    return [ell_arr[i] for i in indices], [S_arr[i] for i in indices]


def _param_error(true_params: dict, estimated: dict) -> dict:
    errors = {}
    for key, true_val in true_params.items():
        if key in estimated:
            errors[key] = abs(estimated[key] - true_val)
    return errors


def evaluate_noise_response(
    ell: Sequence[float],
    S: Sequence[float],
    eps: float,
    fit_fn: Callable[[Sequence[float], Sequence[float], float], "FitResult"],
    *,
    true_params: dict,
    noise_levels: Sequence[float],
    drop_fraction: float = 0.0,
    trials: int = 10,
    rng: random.Random | None = None,
) -> List[NoiseResult]:
    """Evaluate parameter errors as noise is increased."""

    from .inverse_ads3 import FitResult  # local import to avoid cycles

    rng = rng or random.Random()
    results: List[NoiseResult] = []
    for noise in noise_levels:
        aggregated: dict[str, list[float]] = {k: [] for k in true_params}
        for _ in range(trials):
            noisy_S = add_noise(S, noise, rng=rng)
            ell_sub, S_sub = drop_points(ell, noisy_S, drop_fraction, rng=rng)
            fit: FitResult = fit_fn(ell_sub, S_sub, eps)
            errors = _param_error(true_params, fit.params)
            for key, value in errors.items():
                aggregated[key].append(value)
        mean_errors = {k: float(statistics.fmean(v)) if len(v) > 0 else float("nan") for k, v in aggregated.items()}
        results.append(NoiseResult(noise_level=float(noise), errors=mean_errors))
    return results
