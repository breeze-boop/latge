from __future__ import annotations

from typing import List, Sequence

import numpy as np

from .randomness import uniform_bounded_coeffs, uniform_mod_coeffs
from .ring import Poly, Ring


def const_poly(ring: Ring, value: int) -> Poly:
    coeffs = [0] * ring.n
    coeffs[0] = int(value)
    return Poly.from_ints(ring, coeffs)


def zero_vec(ring: Ring, length: int) -> List[Poly]:
    return [ring.zero() for _ in range(length)]


def zero_matrix(ring: Ring, rows: int, cols: int) -> List[List[Poly]]:
    return [[ring.zero() for _ in range(cols)] for _ in range(rows)]


def random_poly(ring: Ring) -> Poly:
    return Poly(ring, uniform_mod_coeffs(ring.n, ring.q))


def random_vec(ring: Ring, length: int) -> List[Poly]:
    coeffs = uniform_mod_coeffs(int(length) * ring.n, ring.q).reshape(int(length), ring.n)
    return [Poly(ring, coeffs[i]) for i in range(int(length))]


def random_matrix(ring: Ring, rows: int, cols: int) -> List[List[Poly]]:
    coeffs = uniform_mod_coeffs(int(rows) * int(cols) * ring.n, ring.q).reshape(int(rows), int(cols), ring.n)
    return [[Poly(ring, coeffs[i, j]) for j in range(int(cols))] for i in range(int(rows))]


def small_poly(ring: Ring, bound: int) -> Poly:
    return Poly(ring, uniform_bounded_coeffs(ring.n, bound))


def small_vec(ring: Ring, length: int, bound: int) -> List[Poly]:
    coeffs = uniform_bounded_coeffs(int(length) * ring.n, bound).reshape(int(length), ring.n)
    return [Poly(ring, coeffs[i]) for i in range(int(length))]


def small_matrix(ring: Ring, rows: int, cols: int, bound: int) -> List[List[Poly]]:
    coeffs = uniform_bounded_coeffs(int(rows) * int(cols) * ring.n, bound).reshape(int(rows), int(cols), ring.n)
    return [[Poly(ring, coeffs[i, j]) for j in range(int(cols))] for i in range(int(rows))]


def ternary_vec(ring: Ring, length: int) -> List[Poly]:
    return small_vec(ring, length, 1)


def mat_add(A: Sequence[Sequence[Poly]], B: Sequence[Sequence[Poly]]) -> List[List[Poly]]:
    return [[A[i][j] + B[i][j] for j in range(len(A[0]))] for i in range(len(A))]


def mat_sub(A: Sequence[Sequence[Poly]], B: Sequence[Sequence[Poly]]) -> List[List[Poly]]:
    return [[A[i][j] - B[i][j] for j in range(len(A[0]))] for i in range(len(A))]


def vec_add(a: Sequence[Poly], b: Sequence[Poly]) -> List[Poly]:
    return [a[i] + b[i] for i in range(len(a))]


def vec_sub(a: Sequence[Poly], b: Sequence[Poly]) -> List[Poly]:
    return [a[i] - b[i] for i in range(len(a))]


def mat_mul(A: Sequence[Sequence[Poly]], B: Sequence[Sequence[Poly]]) -> List[List[Poly]]:
    rows = len(A)
    mid = len(A[0])
    cols = len(B[0])
    ring = A[0][0].ring
    out = zero_matrix(ring, rows, cols)
    for i in range(rows):
        for j in range(cols):
            acc = ring.zero()
            for k in range(mid):
                acc = acc + A[i][k] * B[k][j]
            out[i][j] = acc
    return out


def _convolution_arrays(a: np.ndarray, b: np.ndarray, ring: Ring) -> np.ndarray:
    n = ring.n
    q = ring.q
    conv = np.convolve(a, b)
    out = np.array(conv[:n], dtype=np.int64, copy=True)
    if n > 1:
        out[: n - 1] -= conv[n:]
    out %= q
    return out


def _centered_arrays_matrix(A: Sequence[Sequence[Poly]]) -> List[List[np.ndarray]]:
    return [[poly.centered().astype(np.int64, copy=False) for poly in row] for row in A]


def _centered_arrays_vec(x: Sequence[Poly]) -> List[np.ndarray]:
    return [poly.centered().astype(np.int64, copy=False) for poly in x]


def _convolution_mul_array(a: Poly, b: Poly) -> np.ndarray:
    return _convolution_arrays(a.centered().astype(np.int64, copy=False), b.centered().astype(np.int64, copy=False), a.ring)


def mat_mul_convolution(A: Sequence[Sequence[Poly]], B: Sequence[Sequence[Poly]]) -> List[List[Poly]]:
    rows = len(A)
    mid = len(A[0])
    cols = len(B[0])
    ring = A[0][0].ring
    A_cent = _centered_arrays_matrix(A)
    B_cent = _centered_arrays_matrix(B)
    out = zero_matrix(ring, rows, cols)
    for i in range(rows):
        for j in range(cols):
            acc = np.zeros(ring.n, dtype=np.int64)
            for k in range(mid):
                acc += _convolution_arrays(A_cent[i][k], B_cent[k][j], ring)
            out[i][j] = Poly(ring, acc % ring.q)
    return out


def centered_matrix(A: Sequence[Sequence[Poly]]) -> List[List[np.ndarray]]:
    return _centered_arrays_matrix(A)


def mat_mul_convolution_centered_left(A_cent: Sequence[Sequence[np.ndarray]], B: Sequence[Sequence[Poly]], ring: Ring) -> List[List[Poly]]:
    rows = len(A_cent)
    mid = len(A_cent[0])
    cols = len(B[0])
    B_cent = _centered_arrays_matrix(B)
    max_a = max((int(np.max(np.abs(A_cent[i][j]))) for i in range(rows) for j in range(mid)), default=0)
    max_b = max((int(np.max(np.abs(B_cent[i][j]))) for i in range(mid) for j in range(cols)), default=0)
    if max_a * max_b * max(1, mid) * max(1, ring.n) < 4_000_000_000_000_000:
        n = ring.n
        A_pad = np.zeros((rows, mid, 2 * n), dtype=np.float64)
        B_pad = np.zeros((mid, cols, 2 * n), dtype=np.float64)
        for i in range(rows):
            for j in range(mid):
                A_pad[i, j, :n] = A_cent[i][j].astype(np.float64, copy=False)
        for i in range(mid):
            for j in range(cols):
                B_pad[i, j, :n] = B_cent[i][j].astype(np.float64, copy=False)
        conv = np.rint(
            np.fft.irfft(
                np.einsum(
                    "imf,mjf->ijf",
                    np.fft.rfft(A_pad, axis=2),
                    np.fft.rfft(B_pad, axis=2),
                    optimize=True,
                ),
                n=2 * n,
                axis=2,
            )
        ).astype(np.int64)
        out_coeffs = conv[:, :, :n].copy()
        if n > 1:
            out_coeffs[:, :, : n - 1] -= conv[:, :, n : 2 * n - 1]
        out_coeffs %= ring.q
        return [[Poly(ring, out_coeffs[i, j]) for j in range(cols)] for i in range(rows)]
    out = zero_matrix(ring, rows, cols)
    for i in range(rows):
        for j in range(cols):
            acc = np.zeros(ring.n, dtype=np.int64)
            for k in range(mid):
                acc += _convolution_arrays(A_cent[i][k], B_cent[k][j], ring)
            out[i][j] = Poly(ring, acc % ring.q)
    return out


def mat_vec_mul(A: Sequence[Sequence[Poly]], x: Sequence[Poly]) -> List[Poly]:
    rows = len(A)
    cols = len(A[0])
    ring = A[0][0].ring
    out = []
    for i in range(rows):
        acc = ring.zero()
        for j in range(cols):
            acc = acc + A[i][j] * x[j]
        out.append(acc)
    return out


def mat_vec_mul_convolution(A: Sequence[Sequence[Poly]], x: Sequence[Poly]) -> List[Poly]:
    rows = len(A)
    cols = len(A[0])
    ring = A[0][0].ring
    A_cent = _centered_arrays_matrix(A)
    x_cent = _centered_arrays_vec(x)
    out = []
    for i in range(rows):
        acc = np.zeros(ring.n, dtype=np.int64)
        for j in range(cols):
            acc += _convolution_arrays(A_cent[i][j], x_cent[j], ring)
        out.append(Poly(ring, acc % ring.q))
    return out


def mat_transpose_vec_mul(A: Sequence[Sequence[Poly]], x: Sequence[Poly]) -> List[Poly]:
    rows = len(A)
    cols = len(A[0])
    ring = A[0][0].ring
    out = []
    for j in range(cols):
        acc = ring.zero()
        for i in range(rows):
            acc = acc + A[i][j] * x[i]
        out.append(acc)
    return out


def mat_transpose_vec_mul_convolution(A: Sequence[Sequence[Poly]], x: Sequence[Poly]) -> List[Poly]:
    rows = len(A)
    cols = len(A[0])
    ring = A[0][0].ring
    A_cent = _centered_arrays_matrix(A)
    x_cent = _centered_arrays_vec(x)
    out = []
    for j in range(cols):
        acc = np.zeros(ring.n, dtype=np.int64)
        for i in range(rows):
            acc += _convolution_arrays(A_cent[i][j], x_cent[i], ring)
        out.append(Poly(ring, acc % ring.q))
    return out


def concat_cols(A: Sequence[Sequence[Poly]], B: Sequence[Sequence[Poly]]) -> List[List[Poly]]:
    return [list(A[i]) + list(B[i]) for i in range(len(A))]


def as_column(v: Sequence[Poly]) -> List[List[Poly]]:
    return [[x] for x in v]


def first_column(A: Sequence[Sequence[Poly]]) -> List[Poly]:
    return [row[0] for row in A]


def poly_equal(a: Poly, b: Poly) -> bool:
    return a == b


def vec_equal(a: Sequence[Poly], b: Sequence[Poly]) -> bool:
    return len(a) == len(b) and all(a[i] == b[i] for i in range(len(a)))


def matrix_equal(A: Sequence[Sequence[Poly]], B: Sequence[Sequence[Poly]]) -> bool:
    return len(A) == len(B) and all(vec_equal(A[i], B[i]) for i in range(len(A)))


def poly_inf_norm(p: Poly) -> int:
    centered = p.centered()
    return int(np.max(np.abs(centered))) if centered.size else 0


def vec_inf_norm(v: Sequence[Poly]) -> int:
    return max((poly_inf_norm(p) for p in v), default=0)
