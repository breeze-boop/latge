from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from functools import lru_cache
from typing import List, Sequence, Tuple

from .ring import Poly, Ring


@dataclass(frozen=True)
class OTSParams:
    n: int = 128
    m: int = 32
    p: int = 1 << 46
    b: int = 2
    w: int = 64


DEFAULT_OTS_PARAMS = OTSParams()
_USED_SIGNING_KEYS: set[bytes] = set()


@dataclass(frozen=True)
class OTSKeypair:
    sk_bytes: bytes
    vk_bytes: bytes


def _shake(data: bytes, outlen: int) -> bytes:
    return hashlib.shake_256(data).digest(outlen)


def _pack_u32(x: int) -> bytes:
    return int(x).to_bytes(4, "little", signed=False)


def _ring_for_params(params: OTSParams) -> Ring:
    return Ring(params.n, params.p)


def _poly_step(ring: Ring) -> int:
    return len(ring.zero().to_bytes())


def _poly_inf_norm(p: Poly) -> int:
    return max(abs(int(c)) for c in p.centered().tolist())


def _polyvec_inf_norm(v: Sequence[Poly]) -> int:
    return max((_poly_inf_norm(x) for x in v), default=0)


def _sample_bounded_poly(ring: Ring, bound: int, seed: bytes, nonce: int) -> Poly:
    if bound < 0:
        raise ValueError("bound must be non-negative")
    out = []
    ctr = 0
    span = 2 * bound + 1
    limit = (1 << 24) // span * span
    while len(out) < ring.n:
        buf = _shake(seed + _pack_u32(nonce) + _pack_u32(ctr), 3 * ring.n)
        ctr += 1
        pos = 0
        while pos + 3 <= len(buf) and len(out) < ring.n:
            x = int.from_bytes(buf[pos : pos + 3], "little")
            pos += 3
            if x < limit:
                out.append((x % span) - bound)
    return Poly.from_ints(ring, out)


def _sample_uniform_poly(ring: Ring, seed: bytes, nonce: int) -> Poly:
    q = ring.q
    blen = (q.bit_length() + 7) // 8
    limit = (1 << (8 * blen)) // q * q
    coeffs = []
    ctr = 0
    while len(coeffs) < ring.n:
        buf = _shake(seed + b"U" + _pack_u32(nonce) + _pack_u32(ctr), blen * ring.n)
        ctr += 1
        for i in range(0, len(buf), blen):
            x = int.from_bytes(buf[i : i + blen], "little")
            if x < limit:
                coeffs.append(x % q)
                if len(coeffs) == ring.n:
                    break
    return Poly.from_ints(ring, coeffs)


def _inner(row: Sequence[Poly], vec: Sequence[Poly]) -> Poly:
    if len(row) != len(vec):
        raise ValueError("dimension mismatch")
    ring = row[0].ring
    acc = ring.zero()
    for a, b in zip(row, vec):
        acc = acc + (a * b)
    return acc


def _encode_polyvec(v: Sequence[Poly]) -> bytes:
    return b"".join(p.to_bytes() for p in v)


def _decode_polyvec(ring: Ring, raw: bytes, count: int) -> List[Poly]:
    step = _poly_step(ring)
    want = count * step
    if len(raw) != want:
        raise ValueError("bad polyvec byte length")
    out = []
    pos = 0
    for _ in range(count):
        out.append(Poly.from_bytes(ring, raw[pos : pos + step]))
        pos += step
    return out


def _serialize_sk(params: OTSParams, seed_sk: bytes, vk_digest: bytes | None = None) -> bytes:
    header = (b"OTS3SK" if vk_digest is not None else b"OTS2SK") + _pack_u32(params.n) + _pack_u32(params.m)
    header += int(params.p).to_bytes(8, "little")
    header += _pack_u32(params.b) + _pack_u32(params.w)
    if len(seed_sk) != 32:
        raise ValueError("seed_sk must be 32 bytes")
    if vk_digest is None:
        return header + seed_sk
    if len(vk_digest) != 32:
        raise ValueError("vk_digest must be 32 bytes")
    return header + seed_sk + vk_digest


def _serialize_vk(params: OTSParams, seed_h: bytes, khat1: Poly, khat2: Poly) -> bytes:
    header = b"OTS2VK" + _pack_u32(params.n) + _pack_u32(params.m)
    header += int(params.p).to_bytes(8, "little")
    header += _pack_u32(params.b) + _pack_u32(params.w)
    if len(seed_h) != 32:
        raise ValueError("seed_h must be 32 bytes")
    return header + seed_h + khat1.to_bytes() + khat2.to_bytes()


def _serialize_sig(params: OTSParams, sig_vec: Sequence[Poly], vk_digest: bytes | None = None) -> bytes:
    header = (b"OTS4SG" if vk_digest is not None else b"OTS3SG") + _pack_u32(params.n) + _pack_u32(params.m)
    header += int(params.p).to_bytes(8, "little")
    header += _pack_u32(params.b) + _pack_u32(params.w)
    if vk_digest is not None:
        if len(vk_digest) != 32:
            raise ValueError("vk_digest must be 32 bytes")
        header += vk_digest
    return header + _encode_polyvec(sig_vec)


@lru_cache(maxsize=256)
def _parse_sk(sk_bytes: bytes) -> Tuple[OTSParams, bytes, bytes | None]:
    if len(sk_bytes) < 62 or sk_bytes[:6] not in (b"OTS2SK", b"OTS3SK"):
        raise ValueError("bad OTS secret key")
    n = int.from_bytes(sk_bytes[6:10], "little")
    m = int.from_bytes(sk_bytes[10:14], "little")
    p = int.from_bytes(sk_bytes[14:22], "little")
    b = int.from_bytes(sk_bytes[22:26], "little")
    w = int.from_bytes(sk_bytes[26:30], "little")
    expected = 62 if sk_bytes[:6] == b"OTS2SK" else 94
    if len(sk_bytes) != expected:
        raise ValueError("bad OTS secret key length")
    params = OTSParams(n=n, m=m, p=p, b=b, w=w)
    vk_digest = None if sk_bytes[:6] == b"OTS2SK" else sk_bytes[62:94]
    return params, sk_bytes[30:62], vk_digest


@lru_cache(maxsize=256)
def _parse_vk(vk_bytes: bytes) -> Tuple[OTSParams, bytes, Poly, Poly]:
    if len(vk_bytes) < 62 or vk_bytes[:6] != b"OTS2VK":
        raise ValueError("bad OTS public key")
    n = int.from_bytes(vk_bytes[6:10], "little")
    m = int.from_bytes(vk_bytes[10:14], "little")
    p = int.from_bytes(vk_bytes[14:22], "little")
    b = int.from_bytes(vk_bytes[22:26], "little")
    w = int.from_bytes(vk_bytes[26:30], "little")
    params = OTSParams(n=n, m=m, p=p, b=b, w=w)
    ring = _ring_for_params(params)
    step = _poly_step(ring)
    expected = 62 + 2 * step
    if len(vk_bytes) != expected:
        raise ValueError("bad OTS public key length")
    seed_h = vk_bytes[30:62]
    khat1 = Poly.from_bytes(ring, vk_bytes[62 : 62 + step])
    khat2 = Poly.from_bytes(ring, vk_bytes[62 + step : 62 + 2 * step])
    return params, seed_h, khat1, khat2


@lru_cache(maxsize=256)
def _parse_sig(sig_bytes: bytes) -> Tuple[OTSParams, bytes | None, List[Poly]]:
    if len(sig_bytes) < 22 or sig_bytes[:6] not in (b"OTS2SG", b"OTS3SG", b"OTS4SG"):
        raise ValueError("bad OTS signature")
    n = int.from_bytes(sig_bytes[6:10], "little")
    m = int.from_bytes(sig_bytes[10:14], "little")
    p = int.from_bytes(sig_bytes[14:22], "little")
    vk_digest = None
    if sig_bytes[:6] in (b"OTS3SG", b"OTS4SG"):
        if len(sig_bytes) < 30:
            raise ValueError("bad OTS signature")
        b = int.from_bytes(sig_bytes[22:26], "little")
        w = int.from_bytes(sig_bytes[26:30], "little")
        offset = 30
        if sig_bytes[:6] == b"OTS4SG":
            if len(sig_bytes) < 62:
                raise ValueError("bad OTS signature")
            vk_digest = sig_bytes[30:62]
            offset = 62
    else:
        b = DEFAULT_OTS_PARAMS.b
        w = DEFAULT_OTS_PARAMS.w
        offset = 22
    params = OTSParams(n=n, m=m, p=p, b=b, w=w)
    ring = _ring_for_params(params)
    sig_vec = _decode_polyvec(ring, sig_bytes[offset:], params.m)
    return params, vk_digest, sig_vec


@lru_cache(maxsize=256)
def _expand_secret(params: OTSParams, seed_sk: bytes) -> Tuple[List[Poly], List[Poly]]:
    ring = _ring_for_params(params)
    k1 = [_sample_bounded_poly(ring, params.b, seed_sk + b"K1", i) for i in range(params.m)]
    k2 = [_sample_bounded_poly(ring, params.w * params.b, seed_sk + b"K2", i) for i in range(params.m)]
    return k1, k2


@lru_cache(maxsize=256)
def _expand_public_row(params: OTSParams, seed_h: bytes) -> List[Poly]:
    ring = _ring_for_params(params)
    return [_sample_uniform_poly(ring, seed_h + b"H", i) for i in range(params.m)]


def hash_ct_components_to_digest(c_rec1: Sequence[Poly], c_rec2: Sequence[Poly], c_oa1: Sequence[Poly], c_oa2: Poly) -> bytes:
    h = hashlib.sha256()
    h.update(b"Test2-OTS-msg")
    for p in c_rec1:
        h.update(p.to_bytes())
    for p in c_rec2:
        h.update(p.to_bytes())
    for p in c_oa1:
        h.update(p.to_bytes())
    h.update(c_oa2.to_bytes())
    return h.digest()


@lru_cache(maxsize=2048)
def _digest_to_sparse_message_poly_cached(digest: bytes, params: OTSParams) -> Poly:
    ring = _ring_for_params(params)
    coeffs = [0] * params.n
    used = set()
    ctr = 0
    while len(used) < params.w:
        buf = _shake(digest + b"M" + _pack_u32(ctr), 8)
        ctr += 1
        idx = int.from_bytes(buf[:4], "little") % params.n
        if idx in used:
            continue
        coeffs[idx] = 1 if (buf[4] & 1) else -1
        used.add(idx)
    return Poly.from_ints(ring, coeffs)


def hash_digest_to_message_poly(digest: bytes, params: OTSParams = DEFAULT_OTS_PARAMS) -> Poly:
    return _digest_to_sparse_message_poly_cached(bytes(digest), params)


def ots_gen(params: OTSParams = DEFAULT_OTS_PARAMS, seed: bytes | None = None) -> OTSKeypair:
    if seed is None:
        seed = os.urandom(32)
    if len(seed) < 32:
        seed = _shake(seed, 32)
    seed_sk = _shake(seed + b"SK", 32)
    seed_h = _shake(seed + b"VH", 32)

    k1, k2 = _expand_secret(params, seed_sk)
    H = _expand_public_row(params, seed_h)
    khat1 = _inner(H, k1)
    khat2 = _inner(H, k2)
    vk_bytes = _serialize_vk(params, seed_h, khat1, khat2)
    return OTSKeypair(
        sk_bytes=_serialize_sk(params, seed_sk, hashlib.sha256(vk_bytes).digest()),
        vk_bytes=vk_bytes,
    )


def _ots_sign_raw(sk_bytes: bytes, msg: bytes) -> bytes:
    params, seed_sk, vk_digest = _parse_sk(sk_bytes)
    k1, k2 = _expand_secret(params, seed_sk)
    m1 = hash_digest_to_message_poly(msg, params)
    sig_vec = [k1j * m1 + k2j for (k1j, k2j) in zip(k1, k2)]
    if _polyvec_inf_norm(sig_vec) > 2 * params.w * params.b:
        raise ValueError("OTS signing produced out-of-bound signature")
    return _serialize_sig(params, sig_vec, vk_digest)


def ots_sign(sk_bytes: bytes, msg: bytes, enforce_one_time: bool = True) -> bytes:
    key_id = hashlib.sha256(sk_bytes).digest()
    if enforce_one_time and key_id in _USED_SIGNING_KEYS:
        raise ValueError("OTS secret key has already been used")
    sig = _ots_sign_raw(sk_bytes, msg)
    if enforce_one_time:
        _USED_SIGNING_KEYS.add(key_id)
    return sig


@lru_cache(maxsize=512)
def _verify_cached(vk_bytes: bytes, sig: bytes, msg: bytes) -> bool:
    params, seed_h, khat1, khat2 = _parse_vk(vk_bytes)
    sig_params, vk_digest, sig_vec = _parse_sig(sig)
    if (sig_params.n, sig_params.m, sig_params.p, sig_params.b, sig_params.w) != (
        params.n,
        params.m,
        params.p,
        params.b,
        params.w,
    ):
        return False
    if vk_digest is None or vk_digest != hashlib.sha256(vk_bytes).digest():
        return False
    if _polyvec_inf_norm(sig_vec) > 2 * params.w * params.b:
        return False
    H = _expand_public_row(params, seed_h)
    m1 = hash_digest_to_message_poly(msg, params)
    lhs = _inner(H, sig_vec)
    rhs = khat1 * m1 + khat2
    return lhs == rhs


def ots_verify(vk_bytes: bytes, sig: bytes, msg: bytes) -> bool:
    try:
        return _verify_cached(vk_bytes, sig, bytes(msg))
    except Exception:
        return False


def ots_self_test() -> bool:
    try:
        params = OTSParams(n=16, m=8, p=257, b=1, w=4)
        kp = ots_gen(params, seed=b"Test2-OTS-self-test")
        sig = ots_sign(kp.sk_bytes, b"message", enforce_one_time=False)
        if sig[:6] != b"OTS4SG":
            return False
        if not ots_verify(kp.vk_bytes, sig, b"message"):
            return False
        if ots_verify(kp.vk_bytes, sig, b"tampered"):
            return False
        tampered_sig = bytearray(sig)
        tampered_sig[30] ^= 1
        if ots_verify(kp.vk_bytes, bytes(tampered_sig), b"message"):
            return False
        tampered_vk = bytearray(kp.vk_bytes)
        tampered_vk[-1] ^= 1
        if ots_verify(bytes(tampered_vk), sig, b"message"):
            return False
        return True
    except Exception:
        return False
