"""AdS3/CFT2 entanglement entropy synthetic profiles."""

from __future__ import annotations

import numpy as np


def vacuum_entropy(lengths: np.ndarray, central_charge: float, epsilon: float, s0: float = 0.0) -> np.ndarray:
    """Compute vacuum entanglement entropy for interval lengths.

    Parameters
    ----------
    lengths:
        Positive interval lengths.
    central_charge:
        CFT central charge ``c``.
    epsilon:
        UV cutoff ``\varepsilon``.
    s0:
        Additive constant ``S0``.
    """

    lengths = np.asarray(lengths, dtype=float)
    if np.any(lengths <= 0):
        raise ValueError("lengths must be positive")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    prefactor = central_charge / 3.0
    return prefactor * np.log(lengths / epsilon) + s0


def btz_entropy(
    lengths: np.ndarray,
    central_charge: float,
    epsilon: float,
    beta: float,
    s0: float = 0.0,
) -> np.ndarray:
    """Compute BTZ (finite-temperature) entanglement entropy for interval lengths."""

    lengths = np.asarray(lengths, dtype=float)
    if np.any(lengths <= 0):
        raise ValueError("lengths must be positive")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    if beta <= 0:
        raise ValueError("beta must be positive")
    prefactor = central_charge / 3.0
    scaled = (beta / (np.pi * epsilon)) * np.sinh(np.pi * lengths / beta)
    return prefactor * np.log(scaled) + s0
