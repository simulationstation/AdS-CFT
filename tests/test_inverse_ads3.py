import numpy as np

from holoop.infer.inverse_ads3 import fit_btz
from holoop.infer.robust import parameter_error_vs_noise
from holoop.synth.ads3 import btz_entropy


def test_btz_inference_noiseless():
    lengths = np.linspace(0.1, 2.5, 60)
    epsilon = 0.05
    beta_true = 4.0
    c_true = 12.0
    s0_true = 0.0
    entropies = btz_entropy(lengths, c_true, epsilon, beta_true, s0_true)

    fit = fit_btz(lengths, entropies, epsilon)

    assert abs(fit.params["beta"] - beta_true) < 0.15
    assert abs(fit.params["central_charge"] - c_true) < 0.5
    assert abs(fit.params["S0"] - s0_true) < 0.2


def test_noise_error_monotonic():
    lengths = np.linspace(0.1, 2.0, 40)
    epsilon = 0.05
    beta_true = 3.5
    c_true = 8.0
    s0_true = 0.0
    entropies = btz_entropy(lengths, c_true, epsilon, beta_true, s0_true)

    noise_levels = [0.0, 0.05, 0.1, 0.2]
    errors = parameter_error_vs_noise(
        lengths, entropies, epsilon, beta_true, noise_levels, trials=5, rng=123
    )

    # Errors should not decrease dramatically as noise increases.
    for lower, higher in zip(errors, errors[1:]):
        assert higher + 1e-6 >= lower

