"""Inverse problems for AdS3 models."""
from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Optional, Tuple


def _as_float_list(values: Iterable[float]) -> list[float]:
    return [float(v) for v in values]


def _mean(values: Iterable[float]) -> float:
    seq = list(values)
    return statistics.fmean(seq) if seq else 0.0


def _linspace(start: float, stop: float, num: int) -> list[float]:
    if num == 1:
        return [start]
    step = (stop - start) / (num - 1)
    return [start + i * step for i in range(num)]


def vacuum_entropy(ell: Iterable[float], c: float, S0: float, eps: float) -> list[float]:
    """Vacuum entanglement entropy for interval ``ell``.

    The model assumes ``S = (c/3) * log(ell/eps) + S0`` with a fixed cutoff ``eps``.
    """

    return [(c / 3.0) * math.log(float(x) / eps) + S0 for x in ell]


@dataclass
class FitResult:
    """Container describing a fitted model."""

    model: str
    params: dict
    residual: float
    method: str

    def summary(self) -> str:
        ordered = ", ".join(f"{k}={v:.4g}" for k, v in self.params.items())
        return f"{self.model} via {self.method}: {ordered} (residual {self.residual:.4g})"


def fit_vacuum(ell: Iterable[float], S: Iterable[float], eps: float) -> FitResult:
    """Fit the vacuum model for fixed ``eps``.

    Parameters
    ----------
    ell, S : array-like
        Interval sizes and corresponding entropy measurements.
    eps : float
        Cutoff used in the model.
    """

    ell_arr = _as_float_list(ell)
    S_arr = _as_float_list(S)
    x = [math.log(l / eps) for l in ell_arr]
    mean_x = _mean(x)
    mean_y = _mean(S_arr)
    denom = sum((xi - mean_x) ** 2 for xi in x) or 1e-12
    slope = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, S_arr)) / denom
    intercept = mean_y - slope * mean_x
    residual = _mean(((slope * xi + intercept) - yi) ** 2 for xi, yi in zip(x, S_arr))
    return FitResult(
        model="vacuum",
        params={"c": 3.0 * slope, "S0": float(intercept)},
        residual=residual,
        method="linear_least_squares",
    )


def btz_entropy(ell: Iterable[float], beta: float, c: float, S0: float, eps: float) -> list[float]:
    """BTZ thermal entropy model.

    ``S = (c/3) * log((beta/(pi*eps)) * sinh(pi * ell / beta)) + S0``
    """

    return [
        (c / 3.0) * math.log((beta / (math.pi * eps)) * math.sinh(math.pi * float(l) / beta)) + S0
        for l in ell
    ]


def _linear_fit_for_beta(ell: Iterable[float], S: Iterable[float], beta: float, eps: float) -> Tuple[float, float, float]:
    x = [math.log((beta / (math.pi * eps)) * math.sinh(math.pi * float(l) / beta)) for l in ell]
    S_list = _as_float_list(S)
    mean_x = _mean(x)
    mean_y = _mean(S_list)
    denom = sum((xi - mean_x) ** 2 for xi in x) or 1e-12
    slope = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, S_list)) / denom
    intercept = mean_y - slope * mean_x
    residual = _mean(((slope * xi + intercept) - yi) ** 2 for xi, yi in zip(x, S_list))
    c = 3.0 * slope
    return c, float(intercept), residual


def _scipy_fit(
    ell: list[float], S: list[float], eps: float, beta_guess: Optional[float]
) -> Optional[FitResult]:
    try:  # pragma: no cover - SciPy requires numpy and may be unavailable
        import numpy as np
        from scipy.optimize import least_squares  # type: ignore
    except Exception:
        return None

    ell_arr = _as_float_list(ell)
    S_arr = _as_float_list(S)

    def residuals(params: Any) -> Any:
        beta, c, S0 = params
        return (c / 3.0) * np.log((beta / (math.pi * eps)) * np.sinh(math.pi * ell_arr / beta)) + S0 - S_arr

    beta0 = beta_guess if beta_guess is not None else max(float(np.median(ell)), eps * 10)
    c0 = 1.0
    S00 = 0.0
    result = least_squares(
        residuals,
        x0=np.array([beta0, c0, S00], dtype=float),
        bounds=([eps * 1e-3, 0.0, -np.inf], [np.inf, np.inf, np.inf]),
    )
    if not result.success:
        return None

    beta, c, S0 = result.x
    residual = float(np.mean(residuals(result.x) ** 2))
    return FitResult(
        model="BTZ",
        params={"beta": float(beta), "c": float(c), "S0": float(S0)},
        residual=residual,
        method="scipy_least_squares",
    )


def fit_btz(
    ell: Iterable[float],
    S: Iterable[float],
    eps: float,
    *,
    beta_guess: Optional[float] = None,
    beta_grid: Tuple[float, float, int] = (0.2, 5.0, 60),
) -> FitResult:
    """Fit the BTZ model.

    Attempts a SciPy-based non-linear least squares fit; if SciPy is missing
    or fails, falls back to a grid search over ``beta`` with linear fits for
    ``c`` and ``S0``.
    """

    ell_arr = _as_float_list(ell)
    S_arr = _as_float_list(S)

    scipy_fit = _scipy_fit(ell_arr, S_arr, eps, beta_guess)
    if scipy_fit is not None:
        return scipy_fit

    beta_min, beta_max, num = beta_grid
    betas = _linspace(beta_min, beta_max, num)
    best: Optional[FitResult] = None
    for beta in betas:
        c, S0, residual = _linear_fit_for_beta(ell_arr, S_arr, beta, eps)
        if best is None or residual < best.residual:
            best = FitResult(
                model="BTZ",
                params={"beta": float(beta), "c": float(c), "S0": float(S0)},
                residual=residual,
                method="grid_search",
            )
    assert best is not None
    return best


def load_dataset(path: str) -> Tuple[list[float], list[float], float]:
    """Load a dataset from ``.npz`` or ``.npy``.

    The file must contain ``ell`` and ``S`` arrays and may include ``eps``.
    If ``eps`` is absent, a default of ``1e-3`` is used.
    """

    try:
        import numpy as np
    except Exception as exc:  # pragma: no cover - depends on optional numpy
        raise ImportError("numpy is required to load datasets") from exc

    data = np.load(path)
    ell = [float(x) for x in data["ell"]]
    S = [float(x) for x in data["S"]]
    eps = float(data["eps"]) if "eps" in data else 1e-3
    return ell, S, eps


def save_fit(path: str, fit: FitResult) -> None:
    """Persist fit parameters to a JSON file."""

    import json

    with open(path, "w", encoding="utf-8") as handle:
        json.dump({"model": fit.model, "params": fit.params, "residual": fit.residual, "method": fit.method}, handle, indent=2)


PlotFunc = Callable[[Iterable[float], Iterable[float], FitResult, float], None]


def save_plot(path: str, ell: Iterable[float], S: Iterable[float], fit: FitResult, eps: float) -> None:
    """Save a plot comparing data to the fitted curve.

    Uses matplotlib when available; otherwise no-op.
    """

    try:  # pragma: no cover - plotting is optional
        import matplotlib.pyplot as plt
    except Exception:
        return

    fig, ax = plt.subplots()
    ax.scatter(ell, S, label="data", color="tab:blue")

    if fit.model == "vacuum":
        S_fit = vacuum_entropy(ell, fit.params["c"], fit.params["S0"], eps)
    else:
        S_fit = btz_entropy(ell, fit.params["beta"], fit.params["c"], fit.params["S0"], eps)
    ax.plot(ell, S_fit, label=f"fit ({fit.method})", color="tab:orange")
    ax.set_xlabel(r"$\ell$")
    ax.set_ylabel("S")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
