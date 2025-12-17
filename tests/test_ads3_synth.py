"""Tests for synthetic AdS3 entanglement profiles."""

import numpy as np

from holoop.synth.ads3 import btz_entropy, vacuum_entropy


def test_vacuum_entropy_monotone_increasing():
    lengths = np.linspace(0.2, 2.0, 10)
    entropies = vacuum_entropy(lengths, central_charge=6.0, epsilon=0.1)
    diffs = np.diff(entropies)
    assert np.all(diffs > 0)


def test_btz_matches_vacuum_for_small_intervals():
    lengths = np.linspace(0.01, 0.1, 8)
    central_charge = 6.0
    epsilon = 0.001
    beta = 10.0

    vacuum_values = vacuum_entropy(lengths, central_charge, epsilon)
    btz_values = btz_entropy(lengths, central_charge, epsilon, beta)

    assert np.allclose(btz_values, vacuum_values, rtol=1e-4, atol=1e-6)
