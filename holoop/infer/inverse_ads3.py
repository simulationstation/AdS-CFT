"""Inverse modelling utilities for AdS3/CFT2 datasets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Tuple

import json
import math

import numpy as np


@dataclass
class FitResult:
    """Container for inferred parameters."""

    model: str
    params: Dict[str, float]
    residual: float


_DEF_BETA_GRID_SIZE = 80


def _to_float_list(values: Iterable[float]) -> list[float]:
    return [float(v) for v in values]


def load_dataset(path: Path) -> Tuple[list[float], list[float], Dict]:
    """Load a JSON dataset produced by ``generate_ads3_dataset``."""

    with Path(path).open("r", encoding="utf-8") as fp:
        payload = json.load(fp)
    lengths = _to_float_list(payload["lengths"])
    entropies = _to_float_list(payload["entanglement_entropy"])
    metadata = payload.get("metadata", {})
    return lengths, entropies, metadata


def _linear_regression(x: list[float], y: list[float]) -> Tuple[float, float, float]:
    """Solve y = a * x + b using closed-form least squares."""

    n = len(x)
    sx = sum(x)
    sy = sum(y)
    sxx = sum(v * v for v in x)
    sxy = sum(a * b for a, b in zip(x, y))
    denom = n * sxx - sx * sx
    if denom == 0:
        slope = 0.0
    else:
        slope = (n * sxy - sx * sy) / denom
    intercept = (sy - slope * sx) / n
    mse = sum((slope * xv + intercept - yv) ** 2 for xv, yv in zip(x, y)) / n
    return slope, intercept, mse


def fit_vacuum(lengths: Iterable[float], entropies: Iterable[float], epsilon: float) -> FitResult:
    """Infer central charge ``c`` and ``S0`` for the vacuum model."""

    lengths_list = _to_float_list(lengths)
    entropies_list = _to_float_list(entropies)
    x = [math.log(l / epsilon) for l in lengths_list]
    slope, intercept, mse = _linear_regression(x, entropies_list)
    params = {"central_charge": 3.0 * slope, "S0": intercept, "epsilon": epsilon}
    return FitResult(model="vacuum", params=params, residual=mse)


def _log_btz_term(lengths: list[float], epsilon: float, beta: float) -> list[float]:
    return [math.log((beta / (math.pi * epsilon)) * math.sinh(math.pi * l / beta)) for l in lengths]


def _fit_c_s0_for_beta(lengths: list[float], entropies: list[float], epsilon: float, beta: float) -> Tuple[float, float, float]:
    x = _log_btz_term(lengths, epsilon, beta)
    slope, intercept, mse = _linear_regression(x, entropies)
    central_charge = 3.0 * slope
    return central_charge, intercept, mse


def _grid_search_beta(
    lengths: list[float],
    entropies: list[float],
    epsilon: float,
    grid: list[float],
) -> Tuple[float, float, float, float]:
    best_beta = float(grid[0])
    best_c, best_s0, best_mse = _fit_c_s0_for_beta(lengths, entropies, epsilon, best_beta)
    for candidate in grid[1:]:
        c, s0, mse = _fit_c_s0_for_beta(lengths, entropies, epsilon, float(candidate))
        if mse < best_mse:
            best_beta, best_c, best_s0, best_mse = float(candidate), c, s0, mse
    return best_beta, best_c, best_s0, best_mse


def fit_btz(
    lengths: Iterable[float],
    entropies: Iterable[float],
    epsilon: float,
    beta_grid: np.ndarray | None = None,
) -> FitResult:
    """Infer ``beta``, ``c`` and ``S0`` for the BTZ model."""

    lengths_list = _to_float_list(lengths)
    entropies_list = _to_float_list(entropies)

    default_grid = beta_grid.tolist() if beta_grid is not None else None
    if default_grid is None:
        lo = max(1e-3, 0.5 * min(lengths_list))
        hi = 4.0 * max(lengths_list)
        step = (hi - lo) / (_DEF_BETA_GRID_SIZE - 1)
        default_grid = [lo + i * step for i in range(_DEF_BETA_GRID_SIZE)]

    try:
        from scipy import optimize  # type: ignore
        import numpy as _np

        use_scipy = hasattr(optimize, "least_squares") and hasattr(_np, "array")
    except Exception:
        use_scipy = False

    if not use_scipy:
        beta, c, s0, mse = _grid_search_beta(lengths_list, entropies_list, epsilon, default_grid)
        params = {"beta": beta, "central_charge": c, "S0": s0, "epsilon": epsilon}
        return FitResult(model="btz", params=params, residual=mse)

    beta_seed, c_seed, s0_seed, _ = _grid_search_beta(lengths_list, entropies_list, epsilon, default_grid)

    def residuals(p):
        beta, c, s0 = p
        if beta <= 0:
            return [1e6 for _ in entropies_list]
        model = [(c / 3.0) * val + s0 for val in _log_btz_term(lengths_list, epsilon, beta)]
        return [m - e for m, e in zip(model, entropies_list)]

    result = optimize.least_squares(
        residuals,
        x0=np.asarray([beta_seed, c_seed, s0_seed], dtype=float),
        bounds=([1e-6, 0.0, -math.inf], [math.inf, math.inf, math.inf]),
    )
    beta_fit, c_fit, s0_fit = result.x
    mse = float(sum(v * v for v in residuals(result.x)) / len(lengths_list))
    params = {"beta": float(beta_fit), "central_charge": float(c_fit), "S0": float(s0_fit), "epsilon": epsilon}
    return FitResult(model="btz", params=params, residual=mse)

