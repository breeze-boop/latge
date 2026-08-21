from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import List, Sequence

import numpy as np

from .decomposition import poly_base_decompose
from .linalg import centered_matrix, concat_cols, const_poly, mat_add, mat_mul, mat_mul_convolution, mat_mul_convolution_centered_left, mat_sub, matrix_equal, poly_inf_norm, random_matrix, small_matrix, zero_matrix
from .randomness import discrete_gaussian_coeffs, randomized_round, standard_normal_float64, uniform_mod_coeffs
from .ring import Poly, Ring


_COVARIANCE_ROOT_CACHE: dict[tuple[int, bytes, float, float, bool], np.ndarray] = {}
_PREFIX_LIFT_CACHE: dict[tuple[int, bytes], np.ndarray] = {}
_CONDITIONAL_PREFIX_ROOT_CACHE: dict[tuple[int, bytes, float, float], np.ndarray] = {}
_NEGACYCLIC_LEFT_MUL_CACHE: dict[int, tuple[np.ndarray, np.ndarray]] = {}
_NEGACYCLIC_FFT_TWIST_CACHE: dict[int, tuple[np.ndarray, np.ndarray]] = {}
_PREFIX_LIFT_FFT_CACHE: dict[tuple[int, bytes], np.ndarray] = {}
_CONDITIONAL_PREFIX_SPECTRAL_ROOT_CACHE: dict[tuple[int, bytes, float, float], np.ndarray] = {}
_STRICT_PERTURBATION_POOL: dict[tuple[int, bytes, float, float, int], list[tuple[List[List[Poly]], List[List[Poly]]]]] = {}


@dataclass
class TrapdoorKey:
    ring: Ring
    rows: int
    prefix_cols: int
    gadget_digits: int
    base: int
    A: List[List[Poly]]
    R: List[List[Poly]]
    G: List[List[Poly]]
    A_centered: List[List[np.ndarray]]
    R_centered: List[List[np.ndarray]]


@dataclass(frozen=True)
class TrapdoorRelationProof:
    tau: int
    challenge: int
    commitment: List[List[Poly]]
    response: List[List[Poly]]


@dataclass(frozen=True)
class CovarianceDiagnostics:
    coefficient_level: bool
    dimension: int
    min_eigenvalue: float
    max_eigenvalue: float
    trace: float
    root_error_inf: float


@dataclass(frozen=True)
class SampleDReport:
    target_sigma: float
    gadget_kernel_sigma: float
    matrix_covariance: bool
    coefficient_level_covariance: bool
    sample_rows: int
    sample_cols: int
    sample_inf_norm: int
    preimage_ok: bool
    covariance: CovarianceDiagnostics | None = None
    expected_covariance_dimension: int | None = None
    covariance_dimension_ok: bool | None = None
    covariance_psd_ok: bool | None = None
    covariance_root_ok: bool | None = None
    strict_coefficient_level_ok: bool = False


def _pack_matrix(M: Sequence[Sequence[Poly]]) -> bytes:
    return b"".join(poly.to_bytes() for row in M for poly in row)


def _trapdoor_fingerprint(key: TrapdoorKey) -> bytes:
    h = hashlib.sha256()
    h.update(str(key.ring.n).encode())
    h.update(b"|")
    h.update(str(key.ring.q).encode())
    h.update(b"|")
    h.update(str(key.rows).encode())
    h.update(b"|")
    h.update(str(key.prefix_cols).encode())
    h.update(b"|")
    h.update(str(key.gadget_digits).encode())
    h.update(b"|")
    h.update(str(key.base).encode())
    h.update(b"|")
    h.update(_pack_matrix(key.R))
    return h.digest()


def _hash_challenge(A: Sequence[Sequence[Poly]], commitment: Sequence[Sequence[Poly]], tau: int) -> int:
    digest = hashlib.sha256(b"Test2-trapdoor-proof|" + _pack_matrix(A) + _pack_matrix(commitment)).digest()
    return 1 + int.from_bytes(digest[:8], "little", signed=False) % int(tau)


def _matrix_scalar_mul(A: Sequence[Sequence[Poly]], c: int) -> List[List[Poly]]:
    return [[poly.scalar_mul(c) for poly in row] for row in A]


def _matrix_inf_norm(A: Sequence[Sequence[Poly]]) -> int:
    return max((poly_inf_norm(poly) for row in A for poly in row), default=0)


def gadget_matrix(ring: Ring, rows: int, digits: int, base: int) -> List[List[Poly]]:
    G = zero_matrix(ring, rows, rows * digits)
    for row in range(rows):
        for digit in range(digits):
            G[row][row * digits + digit] = const_poly(ring, base**digit)
    return G


def sparse_small_poly(ring: Ring, bound: int, weight: int) -> Poly:
    if bound <= 0 or weight <= 0:
        return ring.zero()
    weight = min(int(weight), ring.n)
    coeffs = np.zeros(ring.n, dtype=np.int64)
    used: set[int] = set()
    while len(used) < weight:
        idx = int(uniform_mod_coeffs(1, ring.n)[0])
        if idx in used:
            continue
        raw = int(uniform_mod_coeffs(1, 2 * bound)[0])
        value = raw - bound
        if value >= 0:
            value += 1
        coeffs[idx] = value
        used.add(idx)
    return Poly.from_ints(ring, coeffs)


def sparse_small_matrix(ring: Ring, rows: int, cols: int, bound: int, weight: int) -> List[List[Poly]]:
    if bound <= 0 or weight <= 0:
        return zero_matrix(ring, rows, cols)
    weight = min(int(weight), ring.n)
    total = int(rows) * int(cols)
    keys = uniform_mod_coeffs(total * ring.n, 1 << 60).reshape(total, ring.n)
    positions = np.argpartition(keys, weight - 1, axis=1)[:, :weight]
    raw = uniform_mod_coeffs(total * weight, 2 * int(bound)).reshape(total, weight)
    values = raw - int(bound)
    values = np.where(values >= 0, values + 1, values)
    out = []
    for row in range(rows):
        out_row = []
        for col in range(cols):
            idx = row * cols + col
            coeffs = np.zeros(ring.n, dtype=np.int64)
            coeffs[positions[idx]] = values[idx]
            out_row.append(Poly.from_ints(ring, coeffs))
        out.append(out_row)
    return out


def _mat_mul_convolution_fft_sparse_right(A: Sequence[Sequence[Poly]], B: Sequence[Sequence[Poly]]) -> List[List[Poly]]:
    rows = len(A)
    mid = len(A[0])
    cols = len(B[0])
    ring = A[0][0].ring
    n = ring.n
    q = ring.q
    B_int = np.empty((mid, cols, n), dtype=np.int64)
    max_abs_b = 0
    for i in range(mid):
        for j in range(cols):
            coeffs = B[i][j].centered().astype(np.int64, copy=False)
            B_int[i, j] = coeffs
            local = int(np.max(np.abs(coeffs))) if coeffs.size else 0
            if local > max_abs_b:
                max_abs_b = local
    if max_abs_b > 32:
        return mat_mul_convolution(A, B)
    A_pad = np.zeros((rows, mid, 2 * n), dtype=np.float64)
    B_pad = np.zeros((mid, cols, 2 * n), dtype=np.float64)
    for i in range(rows):
        for j in range(mid):
            A_pad[i, j, :n] = A[i][j].centered().astype(np.float64, copy=False)
    B_pad[:, :, :n] = B_int.astype(np.float64, copy=False)
    A_freq = np.fft.rfft(A_pad, axis=2)
    B_freq = np.fft.rfft(B_pad, axis=2)
    conv = np.rint(
        np.fft.irfft(np.einsum("imf,mjf->ijf", A_freq, B_freq, optimize=True), n=2 * n, axis=2)
    ).astype(np.int64)
    out_coeffs = conv[:, :, :n].copy()
    if n > 1:
        out_coeffs[:, :, : n - 1] -= conv[:, :, n : 2 * n - 1]
    out_coeffs %= q
    return [[Poly(ring, out_coeffs[i, j]) for j in range(cols)] for i in range(rows)]


def gen_trap(ring: Ring, rows: int, prefix_cols: int, gadget_digits: int, base: int, trapdoor_bound: int) -> TrapdoorKey:
    gadget_cols = rows * gadget_digits
    A_left = random_matrix(ring, rows, prefix_cols)
    trapdoor_weight = max(1, ring.n // 16)
    R = sparse_small_matrix(ring, prefix_cols, gadget_cols, trapdoor_bound, trapdoor_weight)
    G = gadget_matrix(ring, rows, gadget_digits, base)
    A_right = mat_sub(G, _mat_mul_convolution_fft_sparse_right(A_left, R))
    A = concat_cols(A_left, A_right)
    return TrapdoorKey(
        ring=ring,
        rows=rows,
        prefix_cols=prefix_cols,
        gadget_digits=gadget_digits,
        base=base,
        A=A,
        R=R,
        G=G,
        A_centered=centered_matrix(A),
        R_centered=centered_matrix(R),
    )


def gadget_decompose(key: TrapdoorKey, U: Sequence[Sequence[Poly]]) -> List[List[Poly]]:
    cols = len(U[0])
    z = zero_matrix(key.ring, key.rows * key.gadget_digits, cols)
    for row in range(key.rows):
        for col in range(cols):
            parts = poly_base_decompose(U[row][col], key.base, key.gadget_digits)
            for digit, part in enumerate(parts):
                z[row * key.gadget_digits + digit][col] = part
    return z


def randomized_gadget_decompose(key: TrapdoorKey, U: Sequence[Sequence[Poly]], kernel_bound: int) -> List[List[Poly]]:
    z = gadget_decompose(key, U)
    if key.gadget_digits <= 1 or kernel_bound <= 0:
        return z
    cols = len(U[0])
    for row in range(key.rows):
        for col in range(cols):
            carries = [small_matrix(key.ring, 1, 1, kernel_bound)[0][0] for _ in range(key.gadget_digits - 1)]
            z[row * key.gadget_digits][col] = z[row * key.gadget_digits][col] + carries[0].scalar_mul(key.base)
            for digit in range(1, key.gadget_digits - 1):
                z[row * key.gadget_digits + digit][col] = z[row * key.gadget_digits + digit][col] - carries[digit - 1] + carries[digit].scalar_mul(key.base)
            z[row * key.gadget_digits + key.gadget_digits - 1][col] = z[row * key.gadget_digits + key.gadget_digits - 1][col] - carries[-1]
    return z


def gaussian_gadget_decompose(key: TrapdoorKey, U: Sequence[Sequence[Poly]], kernel_sigma: float) -> List[List[Poly]]:
    z = gadget_decompose(key, U)
    if key.gadget_digits <= 1 or kernel_sigma <= 0:
        return z
    cols = len(U[0])
    for row in range(key.rows):
        for col in range(cols):
            carries = [discrete_gaussian_poly(key.ring, kernel_sigma) for _ in range(key.gadget_digits - 1)]
            z[row * key.gadget_digits][col] = z[row * key.gadget_digits][col] + carries[0].scalar_mul(key.base)
            for digit in range(1, key.gadget_digits - 1):
                z[row * key.gadget_digits + digit][col] = z[row * key.gadget_digits + digit][col] - carries[digit - 1] + carries[digit].scalar_mul(key.base)
            z[row * key.gadget_digits + key.gadget_digits - 1][col] = z[row * key.gadget_digits + key.gadget_digits - 1][col] - carries[-1]
    return z


def discrete_gaussian_poly(ring: Ring, sigma: float) -> Poly:
    if sigma <= 0:
        return ring.zero()
    return Poly(ring, discrete_gaussian_coeffs(ring.n, sigma))


def gaussian_matrix(ring: Ring, rows: int, cols: int, sigma: float) -> List[List[Poly]]:
    return [[discrete_gaussian_poly(ring, sigma) for _ in range(cols)] for _ in range(rows)]


def _poly_left_mul_real_matrix(poly: Poly) -> np.ndarray:
    ring = poly.ring
    coeffs = poly.centered().astype(np.float64, copy=False)
    cached = _NEGACYCLIC_LEFT_MUL_CACHE.get(ring.n)
    if cached is None:
        rows = np.arange(ring.n)[:, None]
        cols = np.arange(ring.n)[None, :]
        coeff_idx = (rows - cols) % ring.n
        signs = np.where(rows >= cols, 1.0, -1.0)
        cached = (coeff_idx.astype(np.intp, copy=False), signs)
        _NEGACYCLIC_LEFT_MUL_CACHE[ring.n] = cached
    coeff_idx, signs = cached
    return coeffs[coeff_idx] * signs


def _negacyclic_fft_twists(n: int) -> tuple[np.ndarray, np.ndarray]:
    cached = _NEGACYCLIC_FFT_TWIST_CACHE.get(n)
    if cached is not None:
        return cached
    idx = np.arange(n, dtype=np.float64)
    twist = np.exp(1j * np.pi * idx / float(n)).astype(np.complex128, copy=False)
    inv_twist = np.conjugate(twist)
    cached = (twist, inv_twist)
    _NEGACYCLIC_FFT_TWIST_CACHE[n] = cached
    return cached


def _coeff_channels_to_negacyclic_fft(values: np.ndarray, channels: int, n: int, cols: int) -> np.ndarray:
    twist, _ = _negacyclic_fft_twists(n)
    time_values = values.reshape(channels, n, cols).transpose(0, 2, 1)
    return np.fft.fft(time_values * twist[None, None, :], axis=2).transpose(2, 0, 1)


def _coeff_channels_from_negacyclic_fft(values: np.ndarray, n: int) -> np.ndarray:
    _, inv_twist = _negacyclic_fft_twists(n)
    time_values = np.fft.ifft(values.transpose(1, 2, 0), axis=2) * inv_twist[None, None, :]
    channels, cols, _ = time_values.shape
    return time_values.real.transpose(0, 2, 1).reshape(channels * n, cols)


def trapdoor_prefix_lift_fft(key: TrapdoorKey) -> np.ndarray:
    cache_key = (id(key), _trapdoor_fingerprint(key))
    cached = _PREFIX_LIFT_FFT_CACHE.get(cache_key)
    if cached is not None:
        return cached
    n = key.ring.n
    gadget_cols = key.rows * key.gadget_digits
    twist, _ = _negacyclic_fft_twists(n)
    coeffs = np.empty((key.prefix_cols, gadget_cols, n), dtype=np.float64)
    for row in range(key.prefix_cols):
        for col in range(gadget_cols):
            coeffs[row, col] = key.R[row][col].centered().astype(np.float64, copy=False)
    out = np.fft.fft(coeffs * twist[None, None, :], axis=2).transpose(2, 0, 1).copy()
    _PREFIX_LIFT_FFT_CACHE[cache_key] = out
    return out


def _apply_prefix_lift_float(key: TrapdoorKey, values: np.ndarray, cols: int) -> np.ndarray:
    n = key.ring.n
    gadget_cols = key.rows * key.gadget_digits
    freq = _coeff_channels_to_negacyclic_fft(values, gadget_cols, n, cols)
    lifted = np.einsum("frg,fgc->frc", trapdoor_prefix_lift_fft(key), freq, optimize=True)
    return _coeff_channels_from_negacyclic_fft(lifted, n)


def trapdoor_lift_matrix(key: TrapdoorKey) -> np.ndarray:
    ring = key.ring
    gadget_cols = key.rows * key.gadget_digits
    total_cols = key.prefix_cols + gadget_cols
    out = np.zeros((total_cols * ring.n, gadget_cols * ring.n), dtype=np.float64)
    for row in range(key.prefix_cols):
        row_slice = slice(row * ring.n, (row + 1) * ring.n)
        for col in range(gadget_cols):
            col_slice = slice(col * ring.n, (col + 1) * ring.n)
            out[row_slice, col_slice] = _poly_left_mul_real_matrix(key.R[row][col])
    offset = key.prefix_cols * ring.n
    for col in range(gadget_cols):
        row_slice = slice(offset + col * ring.n, offset + (col + 1) * ring.n)
        col_slice = slice(col * ring.n, (col + 1) * ring.n)
        out[row_slice, col_slice] = np.eye(ring.n, dtype=np.float64)
    return out


def trapdoor_prefix_lift_matrix(key: TrapdoorKey) -> np.ndarray:
    cache_key = (id(key), _trapdoor_fingerprint(key))
    cached = _PREFIX_LIFT_CACHE.get(cache_key)
    if cached is not None:
        return cached
    ring = key.ring
    gadget_cols = key.rows * key.gadget_digits
    out = np.zeros((key.prefix_cols * ring.n, gadget_cols * ring.n), dtype=np.float64)
    for row in range(key.prefix_cols):
        row_slice = slice(row * ring.n, (row + 1) * ring.n)
        for col in range(gadget_cols):
            col_slice = slice(col * ring.n, (col + 1) * ring.n)
            out[row_slice, col_slice] = _poly_left_mul_real_matrix(key.R[row][col])
    _PREFIX_LIFT_CACHE[cache_key] = out
    return out


def trapdoor_block_lift_matrix(key: TrapdoorKey) -> np.ndarray:
    gadget_cols = key.rows * key.gadget_digits
    total_cols = key.prefix_cols + gadget_cols
    out = np.zeros((total_cols, gadget_cols), dtype=np.float64)
    for row in range(key.prefix_cols):
        for col in range(gadget_cols):
            centered = key.R[row][col].centered().astype(np.float64, copy=False)
            out[row, col] = float(np.sqrt(np.dot(centered, centered)))
    out[key.prefix_cols :, :] = np.eye(gadget_cols, dtype=np.float64)
    return out


def perturbation_covariance_matrix(
    key: TrapdoorKey,
    target_sigma: float,
    gadget_sigma: float,
    coefficient_level: bool = False,
) -> np.ndarray:
    T = trapdoor_lift_matrix(key) if coefficient_level else trapdoor_block_lift_matrix(key)
    dim = T.shape[0]
    covariance = (float(target_sigma) ** 2) * np.eye(dim, dtype=np.float64)
    covariance -= (float(gadget_sigma) ** 2) * (T @ T.T)
    return (covariance + covariance.T) * 0.5


def covariance_square_root(covariance: np.ndarray, tolerance: float = 1e-8) -> np.ndarray:
    vals, vecs = np.linalg.eigh(covariance)
    min_val = float(vals[0]) if vals.size else 0.0
    scale = max(1.0, float(np.max(np.diag(covariance))) if covariance.size else 1.0)
    if min_val < -tolerance * scale:
        raise ValueError(f"perturbation covariance is not positive semidefinite: min_eigenvalue={min_val}")
    vals = np.maximum(vals, 0.0)
    return vecs @ (np.sqrt(vals)[:, None] * vecs.T)


def covariance_diagnostics(
    key: TrapdoorKey,
    target_sigma: float,
    gadget_sigma: float,
    coefficient_level: bool = False,
) -> CovarianceDiagnostics:
    covariance = perturbation_covariance_matrix(key, target_sigma, gadget_sigma, coefficient_level)
    vals = np.linalg.eigvalsh(covariance)
    root = cached_covariance_square_root(key, target_sigma, gadget_sigma, coefficient_level)
    root_error = root @ root.T - covariance
    return CovarianceDiagnostics(
        coefficient_level=bool(coefficient_level),
        dimension=int(covariance.shape[0]),
        min_eigenvalue=float(vals[0]) if vals.size else 0.0,
        max_eigenvalue=float(vals[-1]) if vals.size else 0.0,
        trace=float(np.trace(covariance)),
        root_error_inf=float(np.max(np.abs(root_error))) if root_error.size else 0.0,
    )


def covariance_report_ok(diag: CovarianceDiagnostics, tolerance: float = 1e-7) -> bool:
    scale = max(1.0, abs(diag.max_eigenvalue), abs(diag.trace))
    return diag.min_eigenvalue >= -float(tolerance) * scale and diag.root_error_inf <= float(tolerance) * scale


def expected_covariance_dimension(key: TrapdoorKey, coefficient_level: bool = False) -> int:
    total_cols = key.prefix_cols + key.rows * key.gadget_digits
    return total_cols * key.ring.n if coefficient_level else total_cols


def cached_covariance_square_root(
    key: TrapdoorKey,
    target_sigma: float,
    gadget_sigma: float,
    coefficient_level: bool = False,
) -> np.ndarray:
    cache_key = (
        id(key),
        _trapdoor_fingerprint(key),
        round(float(target_sigma), 15),
        round(float(gadget_sigma), 15),
        bool(coefficient_level),
    )
    root = _COVARIANCE_ROOT_CACHE.get(cache_key)
    if root is None:
        covariance = perturbation_covariance_matrix(key, target_sigma, gadget_sigma, coefficient_level)
        root = covariance_square_root(covariance)
        _COVARIANCE_ROOT_CACHE[cache_key] = root
    return root


def cached_conditional_prefix_square_root(
    key: TrapdoorKey,
    target_sigma: float,
    gadget_sigma: float,
) -> np.ndarray:
    s2 = float(target_sigma) ** 2
    r2 = float(gadget_sigma) ** 2
    denom = s2 - r2
    if denom <= 0:
        raise ValueError("coefficient covariance requires gadget_sigma < target_sigma")
    cache_key = (
        id(key),
        _trapdoor_fingerprint(key),
        round(float(target_sigma), 15),
        round(float(gadget_sigma), 15),
    )
    root = _CONDITIONAL_PREFIX_ROOT_CACHE.get(cache_key)
    if root is not None:
        return root
    R = trapdoor_prefix_lift_matrix(key)
    covariance = s2 * np.eye(R.shape[0], dtype=np.float64)
    covariance -= (r2 * s2 / denom) * (R @ R.T)
    covariance = (covariance + covariance.T) * 0.5
    try:
        root = np.linalg.cholesky(covariance)
    except np.linalg.LinAlgError:
        root = covariance_square_root(covariance)
    _CONDITIONAL_PREFIX_ROOT_CACHE[cache_key] = root
    return root


def cached_conditional_prefix_spectral_roots(
    key: TrapdoorKey,
    target_sigma: float,
    gadget_sigma: float,
) -> np.ndarray:
    s2 = float(target_sigma) ** 2
    r2 = float(gadget_sigma) ** 2
    denom = s2 - r2
    if denom <= 0:
        raise ValueError("coefficient covariance requires gadget_sigma < target_sigma")
    cache_key = (
        id(key),
        _trapdoor_fingerprint(key),
        round(float(target_sigma), 15),
        round(float(gadget_sigma), 15),
    )
    cached = _CONDITIONAL_PREFIX_SPECTRAL_ROOT_CACHE.get(cache_key)
    if cached is not None:
        return cached
    freq_lift = trapdoor_prefix_lift_fft(key)
    scale = r2 * s2 / denom
    eye = s2 * np.eye(key.prefix_cols, dtype=np.complex128)
    roots = np.empty((key.ring.n, key.prefix_cols, key.prefix_cols), dtype=np.complex128)
    for idx in range(key.ring.n):
        block = freq_lift[idx]
        covariance = eye - scale * (block @ block.conj().T)
        covariance = (covariance + covariance.conj().T) * 0.5
        try:
            roots[idx] = np.linalg.cholesky(covariance)
        except np.linalg.LinAlgError:
            vals, vecs = np.linalg.eigh(covariance)
            min_val = float(vals[0]) if vals.size else 0.0
            block_scale = max(1.0, float(np.max(np.real(np.diag(covariance)))) if covariance.size else 1.0)
            if min_val < -1e-8 * block_scale:
                raise ValueError(f"conditional prefix covariance is not positive semidefinite: min_eigenvalue={min_val}")
            vals = np.maximum(vals, 0.0)
            roots[idx] = vecs @ (np.sqrt(vals)[:, None] * vecs.conj().T)
    _CONDITIONAL_PREFIX_SPECTRAL_ROOT_CACHE[cache_key] = roots
    return roots


def _sample_conditional_prefix_noise(
    key: TrapdoorKey,
    target_sigma: float,
    gadget_sigma: float,
    cols: int,
) -> np.ndarray:
    n = key.ring.n
    prefix_dim = key.prefix_cols * n
    roots = cached_conditional_prefix_spectral_roots(key, target_sigma, gadget_sigma)
    freq = _coeff_channels_to_negacyclic_fft(
        standard_normal_float64(prefix_dim * cols).reshape(prefix_dim, cols),
        key.prefix_cols,
        n,
        cols,
    )
    sampled = np.einsum("fij,fjc->fic", roots, freq, optimize=True)
    return _coeff_channels_from_negacyclic_fft(sampled, n)


def _strict_perturbation_pool_key(
    key: TrapdoorKey,
    target_sigma: float,
    gadget_sigma: float,
    cols: int,
) -> tuple[int, bytes, float, float, int]:
    return (
        id(key),
        _trapdoor_fingerprint(key),
        round(float(target_sigma), 15),
        round(float(gadget_sigma), 15),
        int(cols),
    )


def coefficient_covariance_gaussian_matrix(
    key: TrapdoorKey,
    rows: int,
    cols: int,
    target_sigma: float,
    gadget_sigma: float,
) -> List[List[Poly]]:
    ring = key.ring
    gadget_cols = key.rows * key.gadget_digits
    total_cols = key.prefix_cols + gadget_cols
    if rows != total_cols:
        raise ValueError("coefficient covariance row mismatch")
    s2 = float(target_sigma) ** 2
    r2 = float(gadget_sigma) ** 2
    denom = s2 - r2
    if denom <= 0:
        raise ValueError("coefficient covariance requires gadget_sigma < target_sigma")
    prefix_dim = key.prefix_cols * ring.n
    gadget_dim = gadget_cols * ring.n
    bottom_sigma = float(np.sqrt(denom))
    bottom = bottom_sigma * standard_normal_float64(gadget_dim * cols).reshape(gadget_dim, cols)
    top_noise = _sample_conditional_prefix_noise(key, target_sigma, gadget_sigma, cols)
    top = top_noise - (r2 / denom) * _apply_prefix_lift_float(key, bottom, cols)
    sample = randomized_round(np.vstack([top, bottom]))
    out = zero_matrix(ring, rows, cols)
    for col in range(cols):
        for row in range(rows):
            out[row][col] = Poly(ring, sample[row * ring.n : (row + 1) * ring.n, col])
    return out


def precompute_strict_perturbations(
    key: TrapdoorKey,
    sigma: float,
    cols: int,
    count: int = 1,
    gadget_kernel_sigma: float | None = None,
) -> None:
    if count <= 0:
        return
    effective_sigma = effective_gadget_kernel_sigma(key, sigma, gadget_kernel_sigma, True)
    pool_key = _strict_perturbation_pool_key(key, sigma, effective_sigma, cols)
    pool = _STRICT_PERTURBATION_POOL.setdefault(pool_key, [])
    rows = len(key.A[0])
    while len(pool) < count:
        perturbation = coefficient_covariance_gaussian_matrix(key, rows, cols, sigma, effective_sigma)
        image = mat_mul_convolution_centered_left(key.A_centered, perturbation, key.ring)
        pool.append((perturbation, image))


def take_precomputed_strict_perturbation(
    key: TrapdoorKey,
    sigma: float,
    gadget_sigma: float,
    cols: int,
) -> tuple[List[List[Poly]], List[List[Poly]]] | None:
    pool_key = _strict_perturbation_pool_key(key, sigma, gadget_sigma, cols)
    pool = _STRICT_PERTURBATION_POOL.get(pool_key)
    if not pool:
        return None
    return pool.pop()


def matrix_covariance_gaussian_matrix(
    key: TrapdoorKey,
    rows: int,
    cols: int,
    target_sigma: float,
    gadget_sigma: float,
    coefficient_level: bool = False,
) -> List[List[Poly]]:
    ring = key.ring
    if coefficient_level:
        return coefficient_covariance_gaussian_matrix(key, rows, cols, target_sigma, gadget_sigma)
    root = cached_covariance_square_root(key, target_sigma, gadget_sigma, coefficient_level)
    out = zero_matrix(ring, rows, cols)
    if root.shape[0] != rows:
        raise ValueError("block covariance dimension mismatch")
    for coeff_idx in range(ring.n):
        for col in range(cols):
            sample = randomized_round(root @ standard_normal_float64(root.shape[1]))
            for row in range(rows):
                out[row][col].coeffs[coeff_idx] = int(sample[row]) % ring.q
    return out


def trapdoor_frobenius_bound(key: TrapdoorKey) -> float:
    total = 0
    for row in key.R:
        for poly in row:
            centered = poly.centered().astype(np.int64, copy=False)
            total += int(np.dot(centered, centered))
    return float(np.sqrt(total))


def sample_pre(
    key: TrapdoorKey,
    U: Sequence[Sequence[Poly]],
    perturbation_sigma: float = 1.0,
    gadget_kernel_bound: int = 1,
    gadget_kernel_sigma: float = 0.0,
    matrix_covariance: bool = False,
    coefficient_level_covariance: bool = False,
    slow_trapdoor_lift: bool = False,
) -> List[List[Poly]]:
    cols = len(U[0])
    if matrix_covariance and gadget_kernel_sigma > 0:
        cached = (
            take_precomputed_strict_perturbation(key, perturbation_sigma, gadget_kernel_sigma, cols)
            if coefficient_level_covariance
            else None
        )
        if cached is None:
            perturbation = matrix_covariance_gaussian_matrix(
                key,
                len(key.A[0]),
                cols,
                perturbation_sigma,
                gadget_kernel_sigma,
                coefficient_level_covariance,
            )
            perturbation_image = mat_mul_convolution_centered_left(key.A_centered, perturbation, key.ring)
        else:
            perturbation, perturbation_image = cached
    else:
        perturbation = gaussian_matrix(key.ring, len(key.A[0]), cols, perturbation_sigma)
        perturbation_image = mat_mul_convolution_centered_left(key.A_centered, perturbation, key.ring)
    adjusted = mat_sub(U, perturbation_image)
    if gadget_kernel_sigma > 0:
        z = gaussian_gadget_decompose(key, adjusted, gadget_kernel_sigma)
    else:
        z = randomized_gadget_decompose(key, adjusted, gadget_kernel_bound)
    top = mat_mul_convolution(key.R, z) if slow_trapdoor_lift else mat_mul_convolution_centered_left(key.R_centered, z, key.ring)
    centered = list(top) + list(z)
    return mat_add(centered, perturbation)


def sample_d(
    key: TrapdoorKey,
    U: Sequence[Sequence[Poly]],
    sigma: float,
    gadget_kernel_sigma: float | None = None,
    gadget_kernel_bound: int = 0,
    matrix_covariance: bool = False,
    coefficient_level_covariance: bool = False,
    slow_trapdoor_lift: bool = False,
) -> List[List[Poly]]:
    gadget_kernel_sigma = effective_gadget_kernel_sigma(key, sigma, gadget_kernel_sigma, matrix_covariance)
    return sample_pre(
        key,
        U,
        perturbation_sigma=sigma,
        gadget_kernel_bound=gadget_kernel_bound,
        gadget_kernel_sigma=gadget_kernel_sigma,
        matrix_covariance=matrix_covariance,
        coefficient_level_covariance=coefficient_level_covariance,
        slow_trapdoor_lift=slow_trapdoor_lift,
    )


def sample_d_strict(
    key: TrapdoorKey,
    U: Sequence[Sequence[Poly]],
    sigma: float,
    gadget_kernel_sigma: float | None = None,
    slow_trapdoor_lift: bool = False,
) -> List[List[Poly]]:
    return sample_d(
        key,
        U,
        sigma=sigma,
        gadget_kernel_sigma=gadget_kernel_sigma,
        gadget_kernel_bound=0,
        matrix_covariance=True,
        coefficient_level_covariance=True,
        slow_trapdoor_lift=slow_trapdoor_lift,
    )


def effective_gadget_kernel_sigma(
    key: TrapdoorKey,
    sigma: float,
    gadget_kernel_sigma: float | None,
    matrix_covariance: bool,
) -> float:
    if gadget_kernel_sigma is not None:
        return float(gadget_kernel_sigma)
    if matrix_covariance:
        return float(sigma) / max(trapdoor_frobenius_bound(key) + 1.0, 1.0)
    return float(sigma)


def precompute_strict_sampler(
    key: TrapdoorKey,
    sigma: float,
    gadget_kernel_sigma: float | None = None,
    perturbation_cols: Sequence[int] = (),
) -> float:
    effective_sigma = effective_gadget_kernel_sigma(key, sigma, gadget_kernel_sigma, True)
    cached_conditional_prefix_spectral_roots(key, sigma, effective_sigma)
    for cols in perturbation_cols:
        precompute_strict_perturbations(key, sigma, int(cols), 1, effective_sigma)
    return effective_sigma


def sample_d_with_report(
    key: TrapdoorKey,
    U: Sequence[Sequence[Poly]],
    sigma: float,
    gadget_kernel_sigma: float | None = None,
    gadget_kernel_bound: int = 0,
    matrix_covariance: bool = False,
    coefficient_level_covariance: bool = False,
    include_covariance_diagnostics: bool = True,
) -> tuple[List[List[Poly]], SampleDReport]:
    effective_sigma = effective_gadget_kernel_sigma(key, sigma, gadget_kernel_sigma, matrix_covariance)
    S = sample_d(
        key,
        U,
        sigma=sigma,
        gadget_kernel_sigma=effective_sigma,
        gadget_kernel_bound=gadget_kernel_bound,
        matrix_covariance=matrix_covariance,
        coefficient_level_covariance=coefficient_level_covariance,
    )
    diagnostics = None
    expected_dim = expected_covariance_dimension(key, coefficient_level_covariance) if matrix_covariance else None
    dimension_ok = None
    psd_ok = None
    root_ok = None
    if matrix_covariance and effective_sigma > 0 and include_covariance_diagnostics:
        diagnostics = covariance_diagnostics(
            key,
            target_sigma=sigma,
            gadget_sigma=effective_sigma,
            coefficient_level=coefficient_level_covariance,
        )
        scale = max(1.0, abs(diagnostics.max_eigenvalue), abs(diagnostics.trace))
        dimension_ok = diagnostics.dimension == expected_dim
        psd_ok = diagnostics.min_eigenvalue >= -1e-7 * scale
        root_ok = diagnostics.root_error_inf <= 1e-7 * scale
    report = SampleDReport(
        target_sigma=float(sigma),
        gadget_kernel_sigma=float(effective_sigma),
        matrix_covariance=bool(matrix_covariance),
        coefficient_level_covariance=bool(coefficient_level_covariance),
        sample_rows=len(S),
        sample_cols=len(S[0]) if S else 0,
        sample_inf_norm=_matrix_inf_norm(S),
        preimage_ok=verify_preimage(key, U, S),
        covariance=diagnostics,
        expected_covariance_dimension=expected_dim,
        covariance_dimension_ok=dimension_ok,
        covariance_psd_ok=psd_ok,
        covariance_root_ok=root_ok,
        strict_coefficient_level_ok=bool(matrix_covariance)
        and bool(coefficient_level_covariance)
        and bool(dimension_ok)
        and bool(psd_ok)
        and bool(root_ok)
        and verify_preimage(key, U, S),
    )
    return S, report


def sample_d_strict_with_report(
    key: TrapdoorKey,
    U: Sequence[Sequence[Poly]],
    sigma: float,
    gadget_kernel_sigma: float | None = None,
) -> tuple[List[List[Poly]], SampleDReport]:
    return sample_d_with_report(
        key,
        U,
        sigma=sigma,
        gadget_kernel_sigma=gadget_kernel_sigma,
        gadget_kernel_bound=0,
        matrix_covariance=True,
        coefficient_level_covariance=True,
        include_covariance_diagnostics=True,
    )


def sample_d_strict_self_test() -> bool:
    try:
        ring = Ring(8, 257)
        key = gen_trap(ring, rows=1, prefix_cols=2, gadget_digits=3, base=4, trapdoor_bound=1)
        target = [[small_matrix(ring, 1, 1, 1)[0][0]]]
        S, report = sample_d_strict_with_report(key, target, sigma=40.0, gadget_kernel_sigma=1.0)
        return (
            verify_preimage(key, target, S)
            and report.preimage_ok
            and report.matrix_covariance
            and report.coefficient_level_covariance
            and report.expected_covariance_dimension == (key.prefix_cols + key.rows * key.gadget_digits) * ring.n
            and report.covariance_dimension_ok is True
            and report.covariance_psd_ok is True
            and report.covariance_root_ok is True
            and report.strict_coefficient_level_ok is True
        )
    except Exception:
        return False


def prove_trapdoor_relation(key: TrapdoorKey, randomness_bound: int, tau: int) -> TrapdoorRelationProof:
    A_left = [row[: key.prefix_cols] for row in key.A]
    witness = small_matrix(key.ring, key.prefix_cols, key.rows * key.gadget_digits, randomness_bound)
    commitment = mat_mul_convolution(A_left, witness)
    challenge = _hash_challenge(key.A, commitment, tau)
    response = mat_add(witness, _matrix_scalar_mul(key.R, challenge))
    return TrapdoorRelationProof(tau=tau, challenge=challenge, commitment=commitment, response=response)


def verify_trapdoor_relation(ring: Ring, A: Sequence[Sequence[Poly]], rows: int, prefix_cols: int, gadget_digits: int, base: int, proof: TrapdoorRelationProof, randomness_bound: int) -> bool:
    if len(A) != rows or any(len(row) != prefix_cols + rows * gadget_digits for row in A):
        return False
    if len(proof.commitment) != rows or any(len(row) != rows * gadget_digits for row in proof.commitment):
        return False
    if len(proof.response) != prefix_cols or any(len(row) != rows * gadget_digits for row in proof.response):
        return False
    if proof.tau <= 0 or proof.challenge != _hash_challenge(A, proof.commitment, proof.tau):
        return False
    if _matrix_inf_norm(proof.response) > randomness_bound + proof.tau:
        return False
    A_left = [list(row[:prefix_cols]) for row in A]
    A_right = [list(row[prefix_cols:]) for row in A]
    G = gadget_matrix(ring, rows, gadget_digits, base)
    relation = mat_sub(G, A_right)
    lhs = mat_mul_convolution(A_left, proof.response)
    rhs = mat_add(proof.commitment, _matrix_scalar_mul(relation, proof.challenge))
    return matrix_equal(lhs, rhs)


def verify_trapdoor(key: TrapdoorKey) -> bool:
    A_left = [row[: key.prefix_cols] for row in key.A]
    A_right = [row[key.prefix_cols :] for row in key.A]
    return matrix_equal(mat_sub(key.G, mat_mul_convolution(A_left, key.R)), A_right)


def verify_preimage(key: TrapdoorKey, U: Sequence[Sequence[Poly]], S: Sequence[Sequence[Poly]]) -> bool:
    return matrix_equal(mat_mul(key.A, S), U)
