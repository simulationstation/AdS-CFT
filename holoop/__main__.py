"""Entry point for holoop package."""
from __future__ import annotations

import argparse
import pathlib
from typing import List

from .infer.inverse_ads3 import FitResult, fit_btz, fit_vacuum, load_dataset, save_fit, save_plot


def _infer_from_outputs(base: pathlib.Path) -> List[FitResult]:
    results: List[FitResult] = []
    if not base.exists():
        print(f"No datasets found: {base} is missing")
        return results

    for path in sorted(base.glob("*.npz")):
        ell, S, eps = load_dataset(str(path))
        vacuum_fit = fit_vacuum(ell, S, eps)
        btz_fit = fit_btz(ell, S, eps)
        best = vacuum_fit if vacuum_fit.residual <= btz_fit.residual else btz_fit
        results.append(best)

        json_path = path.with_suffix("")
        save_fit(json_path.with_suffix(".fit.json"), best)
        save_plot(json_path.with_suffix(".png"), ell, S, best, eps)
        print(f"Fitted {path.name}: {best.summary()}")
    if not results:
        print(f"No .npz datasets found in {base}")
    return results


def main():
    parser = argparse.ArgumentParser(description="holoop scaffold")
    parser.add_argument("--infer", action="store_true", help="run inference from outputs_holoop/")
    args = parser.parse_args()

    if args.infer:
        output_dir = pathlib.Path("outputs_holoop")
        _infer_from_outputs(output_dir)
    else:
        print("holoop scaffold OK")


if __name__ == "__main__":
    main()
