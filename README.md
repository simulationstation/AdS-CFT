# holoop

A toy operational laboratory for exploring holography, bounded observers, and distinguishability of physical processes.

**This is not a model of the real universe.** It is a pedagogical toolkit for testing operational definitions and computational methods.

## Features

- **Synthetic holography**: Generate entanglement entropy curves for AdS3/CFT2 vacuum and BTZ black hole scenarios
- **Parameter inference**: Fit central charge, thermal scale, and offsets from noisy data
- **Distinguishability experiments**: Test whether bounded observers can distinguish eternally active processes

## Usage

```bash
# Run the full demo suite
python -m holoop --suite --outdir outputs_holoop

# Fast mode (fewer samples)
python -m holoop --suite --fast --outdir outputs_holoop

# Individual components
python -m holoop --synth      # Generate synthetic datasets
python -m holoop --infer      # Run parameter inference
```

## Requirements

- Python 3.10+
- pytest (for tests)

## Running Tests

```bash
pytest -q
```

## Project Structure

```
holoop/
├── __init__.py
├── __main__.py          # CLI entry point
├── report.py            # Report generation
├── synth/               # Synthetic data generation
│   ├── ads3.py          # AdS3/CFT2 entropy formulas
│   └── datasets.py      # Dataset generation
├── infer/               # Parameter inference
│   ├── inverse_ads3.py  # Fitting routines
│   └── robust.py        # Robust estimation
└── distinguish/         # Distinguishability experiments
    ├── game.py          # Observer game logic
    └── experiments.py   # Experiment runners
```

## Key Result

The bounded-observer distinguishability experiment shows that different operational notions of "persistence" lead to different conclusions:

| Observer | Late-time accuracy | Distinguishable? |
|----------|-------------------|------------------|
| stream_max | 0.58 | yes |
| buffer_median | 0.54 | no |

Infinite observation time alone does not guarantee that terminal distinctions vanish; the structure of the observer matters.

## Limitations

- Toy models only, not predictive simulations
- Weak distinguishability margins
- Results depend on observer budget and statistic choice
- No claims about real cosmology
