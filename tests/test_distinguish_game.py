"""Tests for bounded distinguishability game."""

from holoop.distinguish.game import ObserverConfig, run_distinguish


def test_distinguish_pulsed_detectable_without_noise():
    config = ObserverConfig(
        name="stream_test",
        sample_budget=128,
        memory_budget=8,
        threshold=0.01,
        statistic="max",
        observer_type="streaming",
    )
    res = run_distinguish(
        T_values=[256.0],
        n_trials=200,
        observer=config,
        sigma=0.0,
        seed=42,
    )
    assert res["accuracy"][0] > 0.7


def test_distinguish_high_noise_near_chance():
    config = ObserverConfig(
        name="stream_noisy",
        sample_budget=8,
        memory_budget=4,
        threshold=0.05,
        statistic="max",
        observer_type="streaming",
    )
    res = run_distinguish(
        T_values=[128.0],
        n_trials=120,
        observer=config,
        sigma=0.1,
        seed=7,
    )
    assert abs(res["accuracy"][0] - 0.5) < 0.1
