from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from typing import List, Tuple

from .ring import Poly, Ring


def _shake256(data: bytes, outlen: int) -> bytes:
    return hashlib.shake_256(data).digest(outlen)


def _i2b(i: int) -> bytes:
    return i.to_bytes(4, "little", signed=False)


def poly_inf_norm(p: Poly) -> int:

    cc = p.centered().tolist()
    m = 0
    for x in cc:
        ax = abs(int(x))
        if ax > m:
            m = ax
    return int(m)


def polyvec_inf_norm(v: List[Poly]) -> int:
    if not v:
        return 0
    m = 0
    for p in v:
        t = poly_inf_norm(p)
        if t > m:
            m = t
    return m


def _pack_poly_centered(p: Poly) -> bytes:
    cc = p.centered().tolist()
    out = bytearray()
    for x in cc:
        out += int(x).to_bytes(8, "little", signed=True)
    return bytes(out)


def _pack_polyvec_centered(v: List[Poly]) -> bytes:
    out = bytearray()
    for p in v:
        out += _pack_poly_centered(p)
    return bytes(out)


def _sample_small_poly(ring: Ring, B: int, seed: bytes, nonce: int) -> Poly:

    n = ring.n
    coeffs = []
    span = 2 * B + 1
    blen = max(2, ((span - 1).bit_length() + 7) // 8)
    limit = (1 << (8 * blen)) // span * span
    ctr = 0
    while len(coeffs) < n:
        stream = _shake256(seed + _i2b(nonce) + _i2b(ctr), blen * n)
        ctr += 1
        for pos in range(0, len(stream), blen):
            t = int.from_bytes(stream[pos : pos + blen], "little")
            if t < limit:
                coeffs.append((t % span) - B)
                if len(coeffs) == n:
                    break
    return Poly.from_ints(ring, coeffs)


def _sample_uniform_poly(ring: Ring, seed: bytes, nonce: int) -> Poly:
    n = ring.n
    q = ring.q
    blen = (q.bit_length() + 7) // 8
    limit = (1 << (8 * blen)) // q * q
    coeffs = []
    ctr = 0
    while len(coeffs) < n:
        stream = _shake256(seed + _i2b(nonce) + _i2b(ctr), blen * n)
        ctr += 1
        for pos in range(0, len(stream), blen):
            t = int.from_bytes(stream[pos : pos + blen], "little")
            if t < limit:
                coeffs.append(t % q)
                if len(coeffs) == n:
                    break
    return Poly.from_ints(ring, coeffs)


def _sample_small_polyvec(ring: Ring, L: int, B: int, seed: bytes, nonce0: int) -> List[Poly]:
    return [_sample_small_poly(ring, B, seed, nonce0 + i) for i in range(L)]


def _mat_vec_mul(A: List[List[Poly]], s: List[Poly]) -> List[Poly]:

    K = len(A)
    L = len(s)
    ring = s[0].ring
    out: List[Poly] = []
    for i in range(K):
        acc = ring.zero()
        for j in range(L):
            acc = acc + (A[i][j] * s[j])
        out.append(acc)
    return out


def _polyvec_add(a: List[Poly], b: List[Poly]) -> List[Poly]:
    return [a[i] + b[i] for i in range(len(a))]


def _polyvec_sub(a: List[Poly], b: List[Poly]) -> List[Poly]:
    return [a[i] - b[i] for i in range(len(a))]


def _poly_scalar_mul(p: Poly, s: int) -> Poly:
    if s == 0:
        return p.ring.zero()
    if s == 1:
        return p
    if s == -1:
        return -p
    coeffs = [int(x) * s for x in p.centered().tolist()]
    return Poly.from_ints(p.ring, coeffs)


def _polyvec_scalar_mul(v: List[Poly], s: int) -> List[Poly]:
    return [_poly_scalar_mul(p, s) for p in v]


def _hash_to_challenge(mu: bytes, w: List[Poly], tau: int) -> int:

    h = hashlib.sha256(mu + _pack_polyvec_centered(w)).digest()
    x = int.from_bytes(h[:4], "little", signed=False)
    return int((x % (2 * tau + 1)) - tau)


def _dm_public_key_digest(pk: "DMPublicKey") -> bytes:
    return hashlib.sha256(b"DM-PK|" + pk.to_bytes()).digest()


def _dm_message_digest(mu: bytes) -> bytes:
    return hashlib.sha256(b"DM-MSG|" + mu).digest()


def _dm_context_digest(pk: "DMPublicKey", beta: int, tau: int) -> bytes:
    ring = pk.t[0].ring
    K = len(pk.A)
    L = len(pk.A[0]) if pk.A else 0
    parts = [
        b"DM-CTX|",
        int(ring.n).to_bytes(4, "little"),
        int(ring.q).to_bytes(8, "little"),
        int(K).to_bytes(4, "little"),
        int(L).to_bytes(4, "little"),
        int(beta).to_bytes(8, "little"),
        int(tau).to_bytes(8, "little", signed=True),
    ]
    return hashlib.sha256(b"".join(parts)).digest()


@dataclass(frozen=True)
class DMPublicKey:
    A: List[List[Poly]]
    t: List[Poly]

    def to_bytes(self) -> bytes:
        out = bytearray()
        out += len(self.A).to_bytes(2, "little")
        out += (len(self.A[0]) if self.A else 0).to_bytes(2, "little")
        for row in self.A:
            for p in row:
                out += p.to_bytes()
        for p in self.t:
            out += p.to_bytes()
        return bytes(out)


@dataclass(frozen=True)
class DMSecretKey:
    s1: List[Poly]
    s2: List[Poly]
    pk: DMPublicKey


@dataclass(frozen=True)
class DMSignature:
    c: int
    z: List[Poly]
    v_tilde: List[Poly]
    pk_digest: bytes = b""
    mu_digest: bytes = b""
    context_digest: bytes = b""


def sig_to_bytes(sig: DMSignature) -> bytes:
    out = bytearray()
    out += int(sig.c).to_bytes(4, "little", signed=True)
    out += len(sig.z).to_bytes(2, "little")
    out += len(sig.v_tilde).to_bytes(2, "little")
    for p in sig.z:
        out += p.to_bytes()
    for p in sig.v_tilde:
        out += p.to_bytes()
    if sig.pk_digest or sig.mu_digest or sig.context_digest:
        if len(sig.pk_digest) != 32 or len(sig.mu_digest) != 32 or len(sig.context_digest) != 32:
            raise ValueError("bad DM signature context digest length")
        out += b"DMCTX1"
        out += sig.pk_digest
        out += sig.mu_digest
        out += sig.context_digest
    return bytes(out)


def sig_from_bytes(ring: Ring, b: bytes) -> DMSignature:
    if len(b) < 8:
        raise ValueError("bad DM signature bytes")
    c = int.from_bytes(b[0:4], "little", signed=True)
    L = int.from_bytes(b[4:6], "little")
    K = int.from_bytes(b[6:8], "little")
    pos = 8


    dummy = Poly.from_ints(ring, [0] * ring.n)
    step = len(dummy.to_bytes())

    z: List[Poly] = []
    for _ in range(L):
        z.append(Poly.from_bytes(ring, b[pos : pos + step]))
        pos += step

    v: List[Poly] = []
    for _ in range(K):
        v.append(Poly.from_bytes(ring, b[pos : pos + step]))
        pos += step

    pk_digest = b""
    mu_digest = b""
    context_digest = b""
    if pos < len(b):
        if b[pos : pos + 6] != b"DMCTX1" or len(b) != pos + 6 + 96:
            raise ValueError("bad DM signature context extension")
        pos += 6
        pk_digest = b[pos : pos + 32]
        pos += 32
        mu_digest = b[pos : pos + 32]
        pos += 32
        context_digest = b[pos : pos + 32]
        pos += 32
    if pos != len(b):
        raise ValueError("bad DM signature trailing bytes")

    return DMSignature(c=c, z=z, v_tilde=v, pk_digest=pk_digest, mu_digest=mu_digest, context_digest=context_digest)


def dm_keygen(ring: Ring, K: int, L: int, B: int, seed: bytes) -> Tuple[DMPublicKey, DMSecretKey]:

    A: List[List[Poly]] = []
    nonce = 1000
    for _i in range(K):
        row: List[Poly] = []
        for _j in range(L):
            row.append(_sample_uniform_poly(ring, seed + b"A", nonce))
            nonce += 1
        A.append(row)

    s1 = _sample_small_polyvec(ring, L, B, seed + b"s1", 2000)
    s2 = _sample_small_polyvec(ring, K, B, seed + b"s2", 3000)
    t = _polyvec_add(_mat_vec_mul(A, s1), s2)

    pk = DMPublicKey(A=A, t=t)
    sk = DMSecretKey(s1=s1, s2=s2, pk=pk)
    return pk, sk


def dm_sign(
    sk: DMSecretKey,
    mu: bytes,
    B: int,
    beta: int,
    tau: int,
    max_tries: int = 256,
    seed: bytes | None = None,
) -> DMSignature:

    if seed is None:
        seed = hashlib.sha256(os.urandom(32) + mu + b"DM_SIGN").digest()

    pk = sk.pk
    ring = pk.t[0].ring
    L = len(sk.s1)
    K = len(pk.t)

    for attempt in range(max_tries):
        y1 = _sample_small_polyvec(ring, L, B, seed + b"y1", 4000 + attempt * 100)
        y2 = _sample_small_polyvec(ring, K, B, seed + b"y2", 8000 + attempt * 100)
        w = _polyvec_add(_mat_vec_mul(pk.A, y1), y2)
        c = _hash_to_challenge(mu, w, tau)

        z = _polyvec_add(y1, _polyvec_scalar_mul(sk.s1, c))
        v_tilde = _polyvec_add(y2, _polyvec_scalar_mul(sk.s2, c))

        if polyvec_inf_norm(z) <= beta and polyvec_inf_norm(v_tilde) <= beta:
            return DMSignature(
                c=c,
                z=z,
                v_tilde=v_tilde,
                pk_digest=_dm_public_key_digest(pk),
                mu_digest=_dm_message_digest(mu),
                context_digest=_dm_context_digest(pk, beta, tau),
            )

    raise ValueError("dm_sign: rejection sampling failed (max_tries exceeded)")


def dm_verify(pk: DMPublicKey, mu: bytes, sig: DMSignature, beta: int, tau: int) -> bool:

    if sig.pk_digest != _dm_public_key_digest(pk):
        return False
    if sig.mu_digest != _dm_message_digest(mu):
        return False
    if sig.context_digest != _dm_context_digest(pk, beta, tau):
        return False
    if abs(int(sig.c)) > tau:
        return False
    if polyvec_inf_norm(sig.z) > beta:
        return False
    if polyvec_inf_norm(sig.v_tilde) > beta:
        return False

    Az = _mat_vec_mul(pk.A, sig.z)
    w_prime = _polyvec_sub(_polyvec_add(Az, sig.v_tilde), _polyvec_scalar_mul(pk.t, sig.c))
    c_prime = _hash_to_challenge(mu, w_prime, tau)
    return int(c_prime) == int(sig.c)


def dm_self_test() -> bool:
    try:
        ring = Ring(16, 257)
        pk, sk = dm_keygen(ring, K=2, L=2, B=2, seed=b"Test2-DM-self-test")
        mu = b"dm-self-test-message"
        sig = dm_sign(sk, mu, B=2, beta=64, tau=20, max_tries=512, seed=b"Test2-DM-sign-seed")
        if not dm_verify(pk, mu, sig, beta=64, tau=20):
            return False
        if dm_verify(pk, b"dm-self-test-tampered", sig, beta=64, tau=20):
            return False
        if dm_verify(
            pk,
            mu,
            DMSignature(
                c=sig.c,
                z=sig.z,
                v_tilde=sig.v_tilde,
                pk_digest=sig.pk_digest,
                mu_digest=b"\x00" * 32,
                context_digest=sig.context_digest,
            ),
            beta=64,
            tau=20,
        ):
            return False
        encoded = sig_to_bytes(sig)
        parsed = sig_from_bytes(ring, encoded)
        if not dm_verify(pk, mu, parsed, beta=64, tau=20):
            return False
        tampered = bytearray(encoded)
        tampered[-1] ^= 1
        try:
            parsed_tampered = sig_from_bytes(ring, bytes(tampered))
        except ValueError:
            return True
        return not dm_verify(pk, mu, parsed_tampered, beta=64, tau=20)
    except Exception:
        return False
