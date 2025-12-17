"""Tests for inference utilities."""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from holoop.infer import btz_entropy, evaluate_noise_response, fit_btz


EPS = 1e-3


def _linspace(start: float, stop: float, num: int):
    if num == 1:
        return [start]
    step = (stop - start) / (num - 1)
    return [start + i * step for i in range(num)]


def test_noiseless_btz_recovers_beta():
    ell = _linspace(0.1, 1.2, 80)
    true_beta = 1.25
    true_c = 2.0
    true_S0 = 0.4
    S = btz_entropy(ell, true_beta, true_c, true_S0, EPS)

    fit = fit_btz(ell, S, EPS, beta_grid=(0.6, 2.0, 80))

    assert abs(fit.params["beta"] - true_beta) < 0.05
    assert abs(fit.params["c"] - true_c) < 0.2
    assert abs(fit.params["S0"] - true_S0) < 0.1


def test_noise_response_degrades_smoothly():
    import random

    rng = random.Random(4)
    ell = _linspace(0.1, 1.0, 40)
    true_params = {"beta": 1.1, "c": 1.7, "S0": 0.2}
    S_clean = btz_entropy(ell, true_params["beta"], true_params["c"], true_params["S0"], EPS)

    noise_levels = [0.0, 0.2, 0.5]
    results = evaluate_noise_response(
        ell,
        S_clean,
        EPS,
        lambda e, S, eps: fit_btz(e, S, eps, beta_grid=(0.5, 2.0, 60)),
        true_params=true_params,
        noise_levels=noise_levels,
        drop_fraction=0.1,
        trials=25,
        rng=rng,
    )

    beta_errors = [r.errors["beta"] for r in results]
    assert beta_errors[0] < 0.05
    assert beta_errors[-1] > beta_errors[0]
