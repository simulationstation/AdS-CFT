"""Very small stub of matplotlib.pyplot to satisfy plotting calls."""

from __future__ import annotations

from pathlib import Path
from typing import Any, List, Tuple

_current_plot: dict[str, Any] = {}


def figure(*args, **kwargs):  # noqa: ANN002, ANN003 - mimic matplotlib signature
    _current_plot.clear()
    _current_plot["data"] = []
    return _current_plot


def plot(x, y, fmt=None, **kwargs):  # noqa: ANN002, ANN003
    paired = list(zip(x, y))
    _current_plot.setdefault("data", []).append(paired)


def xlabel(label: str):
    _current_plot["xlabel"] = label


def ylabel(label: str):
    _current_plot["ylabel"] = label


def title(text: str):
    _current_plot["title"] = text


def legend():
    _current_plot["legend"] = True


def tight_layout():  # pragma: no cover - no-op for stub
    return None


def savefig(path: str | Path, dpi: int | None = None):  # noqa: ARG001
    output_path = Path(path)
    lines: List[str] = []
    lines.append(f"title: {_current_plot.get('title', '')}")
    lines.append(f"xlabel: {_current_plot.get('xlabel', '')}")
    lines.append(f"ylabel: {_current_plot.get('ylabel', '')}")
    for idx, series in enumerate(_current_plot.get("data", []), start=1):
        lines.append(f"series {idx}: x,y")
        for point in series:
            lines.append(f"{point[0]},{point[1]}")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def close(*args, **kwargs):  # noqa: ANN002, ANN003
    _current_plot.clear()


__all__ = [
    "close",
    "figure",
    "legend",
    "plot",
    "savefig",
    "tight_layout",
    "title",
    "xlabel",
    "ylabel",
]
