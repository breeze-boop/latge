from __future__ import annotations

import hashlib
from functools import lru_cache
from typing import List

from .ring import Poly, Ring


def _shake256(data: bytes, outlen: int) -> bytes:
    return hashlib.shake_256(data).digest(outlen)


@lru_cache(maxsize=4096)
def hash_to_poly(ring: Ring, data: bytes, domain: bytes) -> Poly:

    q = ring.q
    n = ring.n
    blen = (q.bit_length() + 7) // 8
    raw = _shake256(domain + data, n * blen)
    coeffs = [int.from_bytes(raw[i * blen : (i + 1) * blen], "little") % q for i in range(n)]
    return Poly.from_ints(ring, coeffs)


@lru_cache(maxsize=512)
def H2(ring: Ring, vk_bytes: bytes) -> Poly:

    return hash_to_poly(ring, vk_bytes, domain=b"GE:H2")


@lru_cache(maxsize=512)
def H2_vec(ring: Ring, vk_bytes: bytes, alpha1: int) -> List[Poly]:

    return [hash_to_poly(ring, vk_bytes + i.to_bytes(4, "little"), domain=b"GE:H2V") for i in range(alpha1)]


@lru_cache(maxsize=512)
def H1(ring: Ring, vk_bytes: bytes, k: int) -> List[Poly]:

    out: List[Poly] = []
    for i in range(k):
        out.append(hash_to_poly(ring, vk_bytes + i.to_bytes(4, "little"), domain=b"GE:H1"))
    return out


@lru_cache(maxsize=512)
def H1_mat(ring: Ring, vk_bytes: bytes, alpha1: int, k: int) -> List[List[Poly]]:

    mat: List[List[Poly]] = []
    for r in range(alpha1):
        row: List[Poly] = []
        for c in range(k):
            row.append(hash_to_poly(ring, vk_bytes + r.to_bytes(2, "little") + c.to_bytes(4, "little"), domain=b"GE:H1M"))
        mat.append(row)
    return mat
