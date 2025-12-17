"""Lightweight numpy stub for constrained environments."""

from __future__ import annotations

import math
from typing import Iterable, List, Sequence

pi = math.pi


class _Array:
    def __init__(self, data: Iterable[float]):
        self._data = [float(x) for x in data]

    def __iter__(self):
        return iter(self._data)

    def __len__(self):
        return len(self._data)

    def __getitem__(self, item):
        return self._data[item]

    def __add__(self, other):
        return _Array(_binary_op(self._data, other, lambda a, b: a + b))

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        return _Array(_binary_op(self._data, other, lambda a, b: a - b))

    def __mul__(self, other):
        return _Array(_binary_op(self._data, other, lambda a, b: a * b))

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        return _Array(_binary_op(self._data, other, lambda a, b: a / b))

    def __rtruediv__(self, other):
        return _Array(_binary_op(self._data, other, lambda a, b: b / a))

    def __le__(self, other):
        return _Array(_binary_op(self._data, other, lambda a, b: a <= b))

    def __lt__(self, other):
        return _Array(_binary_op(self._data, other, lambda a, b: a < b))

    def __ge__(self, other):
        return _Array(_binary_op(self._data, other, lambda a, b: a >= b))

    def __gt__(self, other):
        return _Array(_binary_op(self._data, other, lambda a, b: a > b))

    def tolist(self) -> List[float]:
        return list(self._data)

    @property
    def shape(self):
        return (len(self._data),)


def _binary_op(left: Sequence[float], right, op):
    if isinstance(right, _Array):
        right_vals = right._data
    elif isinstance(right, (list, tuple)):
        right_vals = right
    else:
        right_vals = [right for _ in range(len(left))]
    return [op(a, b) for a, b in zip(left, right_vals)]


def asarray(values: Iterable[float], dtype=float) -> _Array:  # noqa: ARG001
    return _Array(values)


def any(iterable) -> bool:  # type: ignore
    return builtin_any(iterable)


builtin_any = __builtins__["any"]


def log(values):
    if isinstance(values, _Array):
        return _Array(math.log(v) for v in values)
    return math.log(values)


def sinh(values):
    if isinstance(values, _Array):
        return _Array(math.sinh(v) for v in values)
    return math.sinh(values)


def diff(values: Iterable[float]):
    data = list(values)
    return _Array(data[i + 1] - data[i] for i in range(len(data) - 1))


def all(values) -> bool:  # type: ignore
    return builtin_any(not v for v in values) is False


def allclose(a: Iterable[float], b: Iterable[float], rtol=1e-5, atol=1e-8) -> bool:
    a_list = list(a)
    b_list = list(b)
    if len(a_list) != len(b_list):
        return False
    for x, y in zip(a_list, b_list):
        if abs(x - y) > (atol + rtol * abs(y)):
            return False
    return True


def linspace(start: float, stop: float, num: int):
    if num == 1:
        return _Array([start])
    step = (stop - start) / (num - 1)
    return _Array(start + i * step for i in range(num))


class _RNG:
    def __init__(self, seed=None):
        self.seed = seed

    def normal(self, scale=1.0, size=None):
        if size is None:
            size = 1
        if isinstance(size, int):
            count = size
        else:
            count = 1
            for dim in size:
                count *= dim
        zeros = [0.0 for _ in range(count)]
        return _Array(zeros)


class random:
    @staticmethod
    def default_rng(seed=None):
        return _RNG(seed)


ndarray = _Array
__all__ = [
    "all",
    "allclose",
    "any",
    "asarray",
    "diff",
    "linspace",
    "log",
    "ndarray",
    "pi",
    "random",
    "sinh",
]
