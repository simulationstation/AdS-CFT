"""Bounded distinguishability game between continuous and pulsed activity."""

from __future__ import annotations

import math
import random
import statistics
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Tuple


WorldFunc = Callable[[List[float], Dict[str, float]], List[float]]


@dataclass
class ObserverConfig:
    """Configuration for the bounded observer."""

    name: str
    sample_budget: int
    memory_budget: int
    threshold: float
    statistic: str = "max"  # "max" or "median"
    observer_type: str = "streaming"  # "streaming" or "buffer"


def world_continuous(t: List[float], params: Dict[str, float]) -> List[float]:
    """Continuous 1/t activity with small regulator."""

    a0 = float(params.get("A0", 1.0))
    t0 = float(params.get("t0", 1.0))
    return [a0 / (x + t0) for x in t]


def world_pulsed(t: List[float], params: Dict[str, float]) -> List[float]:
    """Pulsed world delivering area ~A0 ln 2 in each [T, 2T] window."""

    a0 = float(params.get("A0", 1.0))
    t0 = float(params.get("t0", 1.0))
    frac = float(params.get("pulse_frac", 0.1))

    # Determine pulse scale for each t independently.
    values: List[float] = []
    for x in t:
        scale = max(x, t0)
        k = math.floor(math.log(scale / t0, 2))
        t_k = t0 * (2**k)
        width = frac * t_k
        height = (a0 * math.log(2.0)) / max(width, 1e-12)
        in_pulse = t_k <= x <= t_k + width
        values.append(height if in_pulse else 0.0)
    return values


WORLD_FUNCS: Dict[str, WorldFunc] = {
    "A": world_continuous,
    "B": world_pulsed,
}


def simulate_window(
    world: str,
    T: float,
    sample_budget: int,
    sigma: float,
    rng: random.Random,
    params: Dict[str, float] | None = None,
) -> Tuple[List[float], List[float]]:
    """Simulate samples from a window [T, 2T] for a given world."""

    params = params or {}
    times = [rng.uniform(T, 2 * T) for _ in range(sample_budget)]
    values = WORLD_FUNCS[world](times, params)
    if sigma > 0:
        values = [v + rng.gauss(0.0, sigma) for v in values]
    return times, values


def _observe_streaming(samples: Iterable[float]) -> Dict[str, float]:
    max_val = float("-inf")
    mean_val = 0.0
    count = 0
    count_pos = 0
    for val in samples:
        count += 1
        mean_val += (val - mean_val) / count
        count_pos += 1 if val > 0 else 0
        if val > max_val:
            max_val = val
    return {"max": max_val if count else 0.0, "mean": mean_val, "frac_pos": count_pos / max(1, count)}


def _observe_buffer(samples: List[float], memory_budget: int) -> Dict[str, float]:
    if memory_budget <= 0:
        return {"max": 0.0, "median": 0.0}
    buf: List[float] = []
    rng = random.Random(0)
    for idx, val in enumerate(samples):
        if idx < memory_budget:
            buf.append(val)
        else:
            j = rng.randint(0, idx)
            if j < memory_budget:
                buf[j] = val
    if not buf:
        return {"max": 0.0, "median": 0.0}
    buf_sorted = sorted(buf)
    median_val = statistics.median(buf_sorted)
    return {"max": max(buf_sorted), "median": median_val}


def observer_decide(samples: List[float], config: ObserverConfig) -> str:
    """Return guess "A" or "B" based on bounded statistics."""

    if config.observer_type == "streaming":
        stats = _observe_streaming(samples)
    elif config.observer_type == "buffer":
        stats = _observe_buffer(samples, config.memory_budget)
    else:
        raise ValueError(f"Unknown observer_type {config.observer_type}")

    stat_key = "median" if config.statistic == "median" else "max"
    stat_val = stats.get(stat_key, 0.0)
    guess_b = stat_val > config.threshold
    return "B" if guess_b else "A"


def run_trial(
    T: float,
    observer: ObserverConfig,
    sigma: float,
    rng: random.Random,
    worldA_params: Dict[str, float],
    worldB_params: Dict[str, float],
) -> Tuple[bool, float]:
    world_label = rng.choice(["A", "B"])
    params = worldA_params if world_label == "A" else worldB_params
    _, samples = simulate_window(world_label, T, observer.sample_budget, sigma, rng, params)
    guess = observer_decide(samples, observer)
    correct = guess == world_label
    return correct, max(samples) if samples else 0.0


def run_distinguish(
    T_values: List[float],
    n_trials: int,
    observer: ObserverConfig,
    sigma: float = 0.02,
    seed: int = 0,
    worldA_params: Dict[str, float] | None = None,
    worldB_params: Dict[str, float] | None = None,
) -> Dict[str, object]:
    """Run distinguishability trials over a schedule of windows."""

    worldA_params = worldA_params or {"A0": 1.0, "t0": 1.0}
    worldB_params = worldB_params or {"A0": 1.0, "t0": 1.0, "pulse_frac": 0.1}
    rng = random.Random(seed)

    T_values = list(T_values)
    accuracies: List[float] = []
    stderrs: List[float] = []
    max_samples: List[float] = []

    for T in T_values:
        correct = 0
        max_vals: List[float] = []
        for _ in range(n_trials):
            ok, max_val = run_trial(T, observer, sigma, rng, worldA_params, worldB_params)
            correct += int(ok)
            max_vals.append(max_val)
        acc = correct / n_trials
        accuracies.append(acc)
        stderrs.append(math.sqrt(acc * (1 - acc) / n_trials))
        max_samples.append(sum(max_vals) / len(max_vals))

    late_accuracy = accuracies[-1] if accuracies else 0.0
    if len(T_values) >= 2:
        logT = [math.log(max(t, 1.0)) for t in T_values]
        y = [a - 0.5 for a in accuracies]
        n = len(logT)
        mean_x = sum(logT) / n
        mean_y = sum(y) / n
        denom = sum((x - mean_x) ** 2 for x in logT)
        slope = sum((x - mean_x) * (yy - mean_y) for x, yy in zip(logT, y)) / denom if denom else 0.0
    else:
        slope = 0.0
    margin = 0.05
    distinguishable = (late_accuracy > 0.5 + margin) and (slope > -0.05)

    return {
        "observer": observer.__dict__,
        "T_values": T_values,
        "accuracy": accuracies,
        "stderr": stderrs,
        "avg_max": max_samples,
        "late_time_accuracy": late_accuracy,
        "trend_slope": slope,
        "late_time_distinguishable": distinguishable,
    }
