from __future__ import annotations

import math
import os
from functools import lru_cache

import numpy as np


def _draw_uint64(count: int) -> np.ndarray:
    if count <= 0:
        return np.array([], dtype=np.uint64)
    return np.frombuffer(os.urandom(8 * count), dtype=np.uint64)


def _uniform_mod_u64(modulus: int, count: int) -> np.ndarray:
    if modulus <= 0:
        raise ValueError("modulus must be positive")
    if count <= 0:
        return np.array([], dtype=np.int64)
    limit = ((1 << 64) // int(modulus)) * int(modulus)
    acceptance = limit / float(1 << 64)
    out: list[np.ndarray] = []
    need = count
    while need:
        draw_count = max(int(math.ceil(need / max(acceptance, 1e-12) * 1.05)), need + 8, 16)
        draw = _draw_uint64(draw_count)
        keep = draw[draw < limit]
        if keep.size:
            vals = (keep[:need] % np.uint64(modulus)).astype(np.int64)
            out.append(vals)
            need -= vals.size
    return np.concatenate(out)[:count]


def uniform_mod_coeffs(count: int, modulus: int) -> np.ndarray:
    return _uniform_mod_u64(modulus, count)


def uniform_bounded_coeffs(count: int, bound: int) -> np.ndarray:
    if bound < 0:
        raise ValueError("bound must be non-negative")
    span = 2 * int(bound) + 1
    return _uniform_mod_u64(span, count) - int(bound)


@lru_cache(maxsize=128)
def _discrete_gaussian_cdf(sigma: float, tail: float) -> tuple[np.ndarray, np.ndarray]:
    if sigma <= 0:
        return np.array([0], dtype=np.int64), np.array([1.0], dtype=np.float64)
    radius = max(1, int(math.ceil(float(tail) * float(sigma))))
    xs = np.arange(-radius, radius + 1, dtype=np.int64)
    weights = np.exp(-math.pi * (xs.astype(np.float64) ** 2) / (float(sigma) ** 2))
    cdf = np.cumsum(weights / np.sum(weights))
    cdf[-1] = 1.0
    return xs, cdf


def discrete_gaussian_coeffs(count: int, sigma: float, tail: float = 12.0) -> np.ndarray:
    xs, cdf = _discrete_gaussian_cdf(float(sigma), float(tail))
    if xs.size == 1:
        return np.zeros(count, dtype=np.int64)
    draws = _draw_uint64(count)
    uniforms = (draws >> np.uint64(11)).astype(np.float64) * (1.0 / (1 << 53))
    idx = np.searchsorted(cdf, uniforms, side="left")
    return xs[idx].astype(np.int64, copy=False)


def uniform_unit_float64(count: int) -> np.ndarray:
    if count <= 0:
        return np.array([], dtype=np.float64)
    draws = _draw_uint64(count)
    return ((draws >> np.uint64(11)).astype(np.float64) + 0.5) * (1.0 / (1 << 53))


def standard_normal_float64(count: int) -> np.ndarray:
    if count <= 0:
        return np.array([], dtype=np.float64)
    pairs = (count + 1) // 2
    u1 = uniform_unit_float64(pairs)
    u2 = uniform_unit_float64(pairs)
    radius = np.sqrt(-2.0 * np.log(u1))
    theta = 2.0 * math.pi * u2
    out = np.empty(2 * pairs, dtype=np.float64)
    out[0::2] = radius * np.cos(theta)
    out[1::2] = radius * np.sin(theta)
    return out[:count]


def randomized_round(values: np.ndarray) -> np.ndarray:
    floors = np.floor(values)
    frac = values - floors
    draws = uniform_unit_float64(values.size).reshape(values.shape)
    return (floors + (draws < frac)).astype(np.int64)
