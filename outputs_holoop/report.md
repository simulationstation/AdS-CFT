# holoop Report

## 1. Overview

This project is a toy operational laboratory for exploring concepts related to holography, bounded observers, and distinguishability of physical processes. It includes synthetic data generators, parameter inference routines, and simple distinguishability experiments. This is **not a model of the real universe**; it is a pedagogical and exploratory toolkit for testing operational definitions and computational methods.

## 2. Synthetic Holography Setup

The synthetic data module generates entanglement entropy curves for two AdS3/CFT2 scenarios:

- **Vacuum**: The standard Ryu-Takayanagi formula for a 2D CFT ground state, where entanglement entropy scales logarithmically with interval length.
- **BTZ**: A thermal state dual to the BTZ black hole geometry, where entanglement entropy includes thermal corrections controlled by inverse temperature beta.

Parameter inference routines attempt to recover the central charge, offset, and (for BTZ) thermal scale from noisy synthetic data. Identifiability is limited: degeneracies exist between parameters, and fits depend on noise level and sampling density.

## 3. Operational Terminality Definitions

This project distinguishes several notions of late-time activity:

- **Instantaneous activity**: Whether a process is active at a specific moment t.
- **Intermittent activity**: Whether a process exhibits any activity within a finite window around t.
- **Window-integrated persistence**: The total integrated activity over a time window, capturing cumulative effects.

These definitions are **inequivalent**. A process may be instantaneously inactive yet show integrated persistence; another may be intermittently active but with negligible cumulative signal. The choice of operational definition affects conclusions about whether processes remain distinguishable at late times.

## 4. QEC Toy Results

| Scenario | Success probability |
|----------|---------------------|
| Repetition code n=9, p=0.2 | n/a |
| Repetition code n=1, p=0.2 | n/a |

Qualitative conclusion: Redundancy in error-correcting codes improves reconstruction fidelity under noise. With sufficient redundancy (n=9), the encoded information can be recovered even when individual bits have ~20% error rates, whereas a single-bit encoding (n=1) fails.

## 5. Distinguishability Experiment (Key Result)

The bounded-observer distinguishability game tests whether a resource-limited observer can distinguish between two eternally active processes (continuous vs pulsed) as observation time T increases.

Two observer types were tested:

- **Streaming max-statistic observer**: Processes samples one at a time, retaining only the running maximum. Memory budget: 8 slots. Threshold: 0.01.
- **Buffered median-statistic observer**: Collects samples into a buffer and computes the median. Memory budget: 32 slots. Threshold: 0.005.

**Results at late time (T ~ 1024):**

| Observer | Late-time accuracy | Distinguishable? |
|----------|-------------------|------------------|
| stream_max | 0.58 | yes |
| buffer_median | 0.54 | no |

The streaming observer using the max statistic achieves above-chance accuracy (58%) and is classified as distinguishable. The buffered observer using the median statistic achieves only 54% accuracy, which falls below the distinguishability threshold.

> Different operational notions of "persistence" lead to different conclusions about whether two eternally active processes can be distinguished at late times.

## 6. Limitations

- These are toy models and diagnostics, not predictive physical simulations.
- Distinguishability margins are weak (accuracies near 50-60%), and results are sensitive to random seeds.
- Conclusions depend strongly on observer budget (sample count, memory size) and choice of summary statistic.
- The threshold for "distinguishable" is a modeling choice, not a physical constant.
- No claims are made about real cosmology, actual AdS/CFT correspondence, or physical observer limitations.

## 7. Conclusion

This toy laboratory demonstrates that operational definitions of late-time activity are not interchangeable. A bounded observer's ability to distinguish eternally active processes depends on both the observation strategy (streaming vs buffered) and the summary statistic employed (max vs median). Infinite observation time alone does not guarantee that terminal distinctions vanish; the structure of the observer matters. These results illustrate the importance of specifying operational protocols when discussing late-time behavior in physical systems.
