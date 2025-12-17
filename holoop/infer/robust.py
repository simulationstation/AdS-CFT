"""Robustness helpers for inverse AdS3 inference."""

from __future__ import annotations

import math
from typing import Iterable, Sequence

import numpy as np

from .inverse_ads3 import fit_btz


def _resolve_rng(rng):
    if hasattr(rng, "normal"):
        return rng
    return np.random.default_rng(rng)


def add_noise(entropies: Iterable[float], sigma: float, rng=None):
    """Add Gaussian noise with scale ``sigma`` to entropies."""

    generator = _resolve_rng(rng)
    values = [float(v) for v in entropies]
    noise = generator.normal(scale=sigma, size=len(values))
    return [v + n for v, n in zip(values, noise)]


def drop_points(lengths: Iterable[float], entropies: Iterable[float], keep_fraction: float, rng=None):
    """Randomly drop points to keep a fraction of the dataset."""

    if not (0 < keep_fraction <= 1.0):
        raise ValueError("keep_fraction must be in (0, 1]")
    generator = _resolve_rng(rng)
    lengths_list = [float(v) for v in lengths]
    entropies_list = [float(v) for v in entropies]
    count = len(lengths_list)
    keep = max(1, int(math.floor(keep_fraction * count)))
    indices = list(generator.choice(count, size=keep, replace=False))
    return [lengths_list[i] for i in indices], [entropies_list[i] for i in indices]


def parameter_error_vs_noise(
    lengths: Iterable[float],
    entropies: Iterable[float],
    epsilon: float,
    beta_true: float,
    noise_levels: Sequence[float],
    trials: int = 10,
    rng=None,
):
    """Estimate mean beta error as a function of noise level."""

    errors = []
    generator = _resolve_rng(rng)
    for sigma in noise_levels:
        sigma_errors = []
        for _ in range(trials):
            noisy_entropies = add_noise(entropies, sigma, generator)
            beta_hat = fit_btz(lengths, noisy_entropies, epsilon).params["beta"]
            sigma_errors.append(abs(beta_hat - beta_true))
        errors.append(float(sum(sigma_errors) / len(sigma_errors)))
    return errors

