from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Dict, Any
import hashlib
import os

import numpy as np

from .ring import Poly, Ring
from .hash_to_ring import H1_mat, H2_vec
from .dm_sig import sig_to_bytes


def _delta_beta(beta: int) -> int:
    return int(beta + 1).bit_length()


def _g_beta(beta: int) -> List[int]:
    return [(beta + (1 << j) - 1) // (1 << j) for j in range(1, _delta_beta(beta) + 1)]


def _idec(value: int, beta: int) -> List[int]:
    rest = int(value)
    out = []
    for weight in _g_beta(beta):
        if rest >= weight:
            out.append(1)
            rest -= weight
        else:
            out.append(0)
    if rest != 0:
        raise ValueError("bad decomposition")
    return out


def _rdec_first(poly: Poly) -> Poly:
    beta = (poly.ring.q - 1) // 2
    centered = poly.centered().astype(np.int64, copy=False)
    sign = np.where(centered < 0, -1, 1).astype(np.int64)
    first = (np.abs(centered) >= _g_beta(beta)[0]).astype(np.int64)
    return Poly(poly.ring, (sign * first) % poly.ring.q)


def _shake(data: bytes, outlen: int) -> bytes:
    return hashlib.shake_256(data).digest(outlen)

def _pack_poly(p: Poly) -> bytes:
    return p.to_bytes()

def _pack_poly_list(v: List[Poly]) -> bytes:
    return b"".join(_pack_poly(p) for p in v)

def _pack_poly_list_list(M: List[List[Poly]]) -> bytes:
    return b"".join(_pack_poly_list(row) for row in M)

def _pack_int(x: int) -> bytes:
    return int(x).to_bytes(8, "little", signed=True)

def _pack_bytes(data: bytes) -> bytes:
    return len(data).to_bytes(4, "little") + data

def _pack_tuple(xs) -> bytes:
    return len(xs).to_bytes(4, "little") + b"".join(_pack_int(int(x)) for x in xs)

def _pack_cert(cert: Any) -> bytes:
    if cert is None:
        return b"cert:none"
    if not hasattr(cert, "tag"):
        parts = [
            b"cert-legacy|t|",
            getattr(cert, "t_bits", b""),
            b"|r|",
            _pack_poly_list(getattr(cert, "r_i")),
            b"|v|",
            _pack_poly_list(getattr(cert, "v_i")),
            b"|mu|",
            getattr(cert, "gm_mu", b""),
        ]
        return b"".join(parts)
    parts = [
        b"cert|tag|",
        _pack_tuple(getattr(cert, "tag")),
        b"|r|",
        _pack_poly_list(getattr(cert, "r_i")),
        b"|h|",
        _pack_poly_list(getattr(cert, "h_i")),
        b"|w|",
        _pack_poly_list(getattr(cert, "w_i")),
        b"|v|",
        _pack_poly_list(getattr(cert, "v_i")),
    ]
    gm_sig = getattr(cert, "gm_sig", None)
    if gm_sig is not None:
        parts.extend([b"|gm_mu|", getattr(cert, "gm_mu", b""), b"|gm_sig|", sig_to_bytes(gm_sig)])
    return b"".join(parts)

def _cert_digest(cert: Any) -> bytes:
    return hashlib.sha256(b"Test2-cert-digest|" + _pack_cert(cert)).digest()

def _relations_digest(stmt: "PiStatement", response_bound: int) -> bytes:
    labels = (
        b"relation_1_AR_m|"
        b"relation_2_h|"
        b"relation_3_uprime|"
        b"relation_4_cert|"
        b"relation_5_bounds|"
        b"relation_6_enc_rec|"
        b"relation_6_enc_oa|"
        b"ots"
    )
    return hashlib.sha256(
        b"Test2-paper-relations-digest|"
        + labels
        + stmt.to_bytes()
        + _pack_int(int(response_bound))
    ).digest()

def _pack_parcom_randomness_proof(proof: Any) -> bytes:
    if proof is None:
        return b"parcom-rk:none"
    parts = [
        b"parcom-rk|tau|",
        _pack_int(getattr(proof, "tau")),
        b"|c|",
        _pack_int(getattr(proof, "challenge")),
        b"|f|",
        _pack_poly(getattr(proof, "f")),
        b"|t1|",
        _pack_poly_list(getattr(proof, "commitment_1")),
        b"|z|",
        _pack_poly_list(getattr(proof, "response")),
        b"|mask|",
        _pack_int(getattr(proof, "mask_bound", 0)),
        b"|bound|",
        _pack_int(getattr(proof, "response_bound", 0)),
        b"|norm|",
        _pack_int(getattr(proof, "response_norm", 0)),
    ]
    return b"".join(parts)

def _pack_response_decomposition_proof(proof: Any) -> bytes:
    if proof is None:
        return b"resp-dec:none"
    parts = [
        b"resp-dec|bound|",
        _pack_int(getattr(proof, "bound")),
        b"|weights|",
        _pack_tuple(getattr(proof, "weights")),
        b"|coeffs|",
        _pack_int(getattr(proof, "coefficient_count")),
        b"|max|",
        _pack_int(getattr(proof, "max_abs")),
        b"|digest|",
        _pack_bytes(getattr(proof, "digest")),
    ]
    return b"".join(parts)

def _pack_audit(audit: Dict[str, Any]) -> bytes:
    parts = [b"Test2-paper-audit|"]
    pk = audit.get("pk_user_A")
    if pk is not None:
        parts.extend([b"|pk|", _pack_poly_list_list(pk)])
    parts.extend([b"|m|", _pack_poly_list(audit["m"])])
    parts.extend([b"|cert|", _pack_cert(audit.get("cert"))])
    parts.extend([b"|xrec|", _pack_poly_list(audit["x_rec"])])
    parts.extend([b"|yrec|", _pack_poly_list(audit["y_rec"])])
    parts.extend([b"|zrec|", _pack_poly_list(audit["z_rec"])])
    x_oa = audit["x_oa"]
    if isinstance(x_oa, list):
        parts.extend([b"|xoa-list|", _pack_poly_list(x_oa)])
    else:
        parts.extend([b"|xoa|", _pack_poly(x_oa)])
    parts.extend([b"|yoa|", _pack_poly_list(audit["y_oa"])])
    parts.extend([b"|zoa|", _pack_poly_list(audit["z_oa"])])
    return b"".join(parts)

def _make_audit_commitment(pub: Any, payload: bytes) -> Tuple[bytes, bytes]:
    parcom = getattr(pub, "parcom", None)
    if parcom is not None and hasattr(parcom, "commit_payload"):
        return parcom.commit_payload(payload)
    opening = os.urandom(32)
    commitment = hashlib.sha256(b"Test2-parCOM-audit|" + opening + payload).digest()
    return commitment, opening

def _make_audit_commitment_with_proof(pub: Any, payload: bytes, tau: int) -> Tuple[bytes, bytes, Any]:
    commitment, opening = _make_audit_commitment(pub, payload)
    parcom = getattr(pub, "parcom", None)
    proof = None
    if parcom is not None and hasattr(parcom, "prove_opening_randomness_knowledge"):
        proof = parcom.prove_opening_randomness_knowledge(commitment, opening, tau=tau)
    return commitment, opening, proof

def verify_audit_randomness_knowledge(pi: Any, parcom: Any = None) -> bool:
    proof = getattr(pi, "audit_randomness_proof", None)
    if parcom is None or not hasattr(parcom, "verify_opening_randomness_knowledge"):
        return proof is None
    if not getattr(pi, "audit_commitment", b"") or proof is None:
        return False
    return parcom.verify_opening_randomness_knowledge(pi.audit_commitment, proof)

def verify_audit_commitment(pi: Any, parcom: Any = None) -> bool:
    if not getattr(pi, "audit_commitment", b"") or not getattr(pi, "audit_opening", b"") or getattr(pi, "audit", None) is None:
        return False
    payload = _pack_audit(pi.audit)
    if parcom is not None and hasattr(parcom, "verify_payload"):
        return parcom.verify_payload(pi.audit_commitment, pi.audit_opening, payload)
    expected = hashlib.sha256(b"Test2-parCOM-audit|" + pi.audit_opening + payload).digest()
    return expected == pi.audit_commitment

def _hash_challenge(stmt_bytes: bytes, t_vec: List[Poly], tau: int) -> int:
    h = hashlib.sha256(stmt_bytes + _pack_poly_list(t_vec)).digest()
    x = int.from_bytes(h[:8], "little", signed=False)
    return int((x % (2 * tau + 1)) - tau)

def _uniform_bounded_coeffs(count: int, bound: int) -> np.ndarray:
    span = 2 * int(bound) + 1
    if span <= 0:
        raise ValueError("bound is too large")
    blen = max(1, ((span - 1).bit_length() + 7) // 8)
    modulus = 1 << (8 * blen)
    limit = (modulus // span) * span
    out = []
    while len(out) < count:
        buf = os.urandom(blen * max(16, count - len(out)))
        for pos in range(0, len(buf), blen):
            sample = int.from_bytes(buf[pos : pos + blen], "little")
            if sample < limit:
                out.append((sample % span) - bound)
                if len(out) == count:
                    break
    return np.array(out, dtype=np.int64)

def _ternary_poly(ring: Ring) -> Poly:
    return Poly(ring, _uniform_bounded_coeffs(ring.n, 1))

def _ternary_vec(ring: Ring, L: int) -> List[Poly]:
    return [_ternary_poly(ring) for _ in range(L)]

def _bounded_poly(ring: Ring, bound: int) -> Poly:
    return Poly(ring, _uniform_bounded_coeffs(ring.n, int(bound)))

def _bounded_vec(ring: Ring, L: int, bound: int) -> List[Poly]:
    return [_bounded_poly(ring, bound) for _ in range(L)]

def _inner_vec(a: List[Poly], b: List[Poly]) -> Poly:
    ring = a[0].ring
    acc = np.zeros(ring.n, dtype=np.int64)
    for x, y in zip(a, b):
        acc += _convolution_arrays(x.centered().astype(np.int64, copy=False), y.centered().astype(np.int64, copy=False), ring)
    return Poly(ring, acc % ring.q)

def _convolution_arrays(a: np.ndarray, b: np.ndarray, ring: Ring) -> np.ndarray:
    n = ring.n
    conv = np.convolve(a, b)
    out = np.array(conv[:n], dtype=np.int64, copy=True)
    if n > 1:
        out[: n - 1] -= conv[n:]
    out %= ring.q
    return out

def _matT_vec_mul(A: List[List[Poly]], z: List[Poly]) -> List[Poly]:
    alpha1 = len(A)
    cols = len(A[0])
    ring = A[0][0].ring
    A_cent = [[p.centered().astype(np.int64, copy=False) for p in row] for row in A]
    z_cent = [p.centered().astype(np.int64, copy=False) for p in z]
    out = []
    for j in range(cols):
        acc = np.zeros(ring.n, dtype=np.int64)
        for r in range(alpha1):
            acc += _convolution_arrays(A_cent[r][j], z_cent[r], ring)
        out.append(Poly(ring, acc % ring.q))
    return out

def _matT_vec_mul_rect(A: List[List[Poly]], z: List[Poly]) -> List[Poly]:
    alpha1 = len(A)
    k = len(A[0])
    ring = A[0][0].ring
    A_cent = [[p.centered().astype(np.int64, copy=False) for p in row] for row in A]
    z_cent = [p.centered().astype(np.int64, copy=False) for p in z]
    out = []
    for j in range(k):
        acc = np.zeros(ring.n, dtype=np.int64)
        for r in range(alpha1):
            acc += _convolution_arrays(A_cent[r][j], z_cent[r], ring)
        out.append(Poly(ring, acc % ring.q))
    return out

def _mat_vec_mul(A: List[List[Poly]], z: List[Poly]) -> List[Poly]:
    rows = len(A)
    cols = len(A[0])
    ring = A[0][0].ring
    A_cent = [[p.centered().astype(np.int64, copy=False) for p in row] for row in A]
    z_cent = [p.centered().astype(np.int64, copy=False) for p in z]
    out = []
    for i in range(rows):
        acc = np.zeros(ring.n, dtype=np.int64)
        for j in range(cols):
            acc += _convolution_arrays(A_cent[i][j], z_cent[j], ring)
        out.append(Poly(ring, acc % ring.q))
    return out

def _vec_add(a: List[Poly], b: List[Poly]) -> List[Poly]:
    return [a[i] + b[i] for i in range(len(a))]

def _vec_sub(a: List[Poly], b: List[Poly]) -> List[Poly]:
    return [a[i] - b[i] for i in range(len(a))]

def _vec_equal(a: List[Poly], b: List[Poly]) -> bool:
    return len(a) == len(b) and all(a[i] == b[i] for i in range(len(a)))

def _mat_equal(a: List[List[Poly]], b: List[List[Poly]]) -> bool:
    return len(a) == len(b) and all(_vec_equal(a[i], b[i]) for i in range(len(a)))

def _mat_add(a: List[List[Poly]], b: List[List[Poly]]) -> List[List[Poly]]:
    return [[a[i][j] + b[i][j] for j in range(len(a[i]))] for i in range(len(a))]

def _concat_cols(a: List[List[Poly]], b: List[List[Poly]]) -> List[List[Poly]]:
    return [list(a[i]) + list(b[i]) for i in range(len(a))]

def _encode_msg_scale(poly: Poly, q: int) -> Poly:
    return poly.scalar_mul(q // 4)

def _poly_inf_norm(p: Poly) -> int:
    cc = p.centered()
    return int(np.max(np.abs(cc))) if cc.size else 0

def _polyvec_inf_norm(v: List[Poly]) -> int:
    m = 0
    for p in v:
        t = _poly_inf_norm(p)
        if t > m:
            m = t
    return m

def _lsb_poly(a: Poly) -> Poly:
    return _rdec_first(a)

def _legacy_parity_lsb_poly(a: Poly) -> Poly:
    return Poly.from_ints(a.ring, [int(c) & 1 for c in a.coeffs.tolist()])

def _certificate_receiver_hint(pk_user: Any, cert: Any) -> Poly:
    if cert is not None and hasattr(cert, "h_i") and getattr(cert, "h_i"):
        return getattr(cert, "h_i")[0]
    if cert is not None and hasattr(cert, "t_bits"):
        return _legacy_parity_lsb_poly(pk_user.A[0][-1])
    return _lsb_poly(pk_user.A[0][-1])

def _rdec_all(poly: Poly, beta: int | None = None) -> List[Poly]:
    ring = poly.ring
    if beta is None:
        beta = (ring.q - 1) // 2
    centered = poly.centered().astype(np.int64, copy=False)
    rest = np.abs(centered).astype(np.int64, copy=True)
    signs = np.where(centered < 0, -1, 1).astype(np.int64)
    out = []
    for weight in _g_beta(beta):
        take = rest >= weight
        out.append(Poly(ring, (signs * take.astype(np.int64)) % ring.q))
        rest[take] -= weight
    if bool(np.any(rest != 0)):
        raise ValueError("bad ring decomposition")
    return out

def _rvdec(vec: List[Poly], beta: int) -> List[Poly]:
    out: List[Poly] = []
    for poly in vec:
        out.extend(_rdec_all(poly, beta))
    return out

def _tag_matrix(pub: Any, tag: Any) -> List[List[Poly]] | None:
    tags = getattr(pub, "A_tags", None)
    if tags is None or tag is None:
        return None
    out = [list(row) for row in tags[0]]
    for j, bit in enumerate(tuple(tag), start=1):
        if bit:
            out = _mat_add(out, tags[j])
    return out

def _certificate_target(pub: Any, cert: Any) -> Tuple[List[Poly], List[Poly]] | None:
    if not all(hasattr(pub, name) for name in ("B0", "B1", "B2", "u", "ring_q")):
        return None
    if cert is None or not hasattr(cert, "h_i") or not hasattr(cert, "r_i"):
        return None
    beta = (pub.ring_q.q - 1) // 2
    w_raw = _vec_add(_mat_vec_mul(pub.B0, cert.r_i), _mat_vec_mul(pub.B1, cert.h_i))
    w_i = _rvdec(w_raw, beta)
    u_prime = _vec_add(_mat_vec_mul(pub.B2, w_i), pub.u)
    return w_i, u_prime

def _private_norm_bounds_ok(pub: Any, witness: Dict[str, Any], wit: "PiWitness") -> bool:
    bound = _proof_secret_bound(pub)
    beta = int(getattr(pub.params, "beta", 95255))
    cert = witness.get("cert")
    checks = [
        _polyvec_inf_norm(wit.m) <= 1,
        _polyvec_inf_norm(wit.x_rec) <= bound,
        _polyvec_inf_norm(wit.y_rec) <= bound,
        _polyvec_inf_norm(wit.z_rec) <= bound,
        _polyvec_inf_norm(wit.y_oa) <= bound,
        _polyvec_inf_norm(wit.z_oa) <= bound,
    ]
    if isinstance(wit.x_oa, list):
        checks.append(_polyvec_inf_norm(wit.x_oa) <= bound)
    else:
        checks.append(_poly_inf_norm(wit.x_oa) <= bound)
    if cert is not None and hasattr(cert, "h_i") and hasattr(cert, "r_i"):
        checks.append(_polyvec_inf_norm(cert.r_i) <= 1)
    if cert is not None and hasattr(cert, "h_i") and hasattr(cert, "v_i"):
        checks.append(_polyvec_inf_norm(cert.v_i) <= beta)
    return all(checks)

def _private_certificate_relations_ok(pub: Any, gm_context: Any, pk_user: Any, cert: Any) -> bool:
    if cert is None:
        return False
    if hasattr(cert, "h_i"):
        try:
            expected_h = _rdec_all(pk_user.A[0][-1], (pub.ring_q.q - 1) // 2)
        except Exception:
            return False
        if not _vec_equal(cert.h_i, expected_h):
            return False
    target = _certificate_target(pub, cert)
    if target is not None and hasattr(cert, "w_i"):
        w_i, u_prime = target
        if not _vec_equal(w_i, cert.w_i):
            return False
        if hasattr(gm_context, "pk_gm") and hasattr(cert, "tag") and hasattr(cert, "v_i"):
            tag_matrix = _tag_matrix(pub, cert.tag)
            if tag_matrix is None:
                return False
            at = _concat_cols(gm_context.pk_gm.A, tag_matrix)
            if not _vec_equal(_mat_vec_mul(at, cert.v_i), u_prime):
                return False
    return True

def _precheck_private_witness(pub: Any, gm_context: Any, stmt: "PiStatement", wit: "PiWitness", witness: Dict[str, Any]) -> bool:
    try:
        if not _vec_equal(_apply_M(stmt, wit), _compute_y(stmt)):
            return False
        if not _private_norm_bounds_ok(pub, witness, wit):
            return False
        pk_user = witness.get("pk_user")
        cert = witness.get("cert")
        if pk_user is None or not _private_certificate_relations_ok(pub, gm_context, pk_user, cert):
            return False
    except Exception:
        return False
    return True

def _proof_secret_bound(pub: Any) -> int:
    return max(1, int(getattr(pub.params, "B", 1)))

def _proof_mask_bound(pub: Any) -> int:
    beta = int(getattr(pub.params, "beta", 95255))
    tau = int(getattr(pub.params, "tau", 1))
    secret_bound = _proof_secret_bound(pub)
    margin = tau * secret_bound
    if beta <= margin + 1:
        return max(1, secret_bound)
    return max(secret_bound + 1, min(beta // 4, beta - margin))

def _proof_response_bound(pub: Any, mask_bound: int | None = None) -> int:
    beta = int(getattr(pub.params, "beta", 95255))
    tau = int(getattr(pub.params, "tau", 1))
    secret_bound = _proof_secret_bound(pub)
    if mask_bound is None:
        mask_bound = _proof_mask_bound(pub)
    return min(beta, int(mask_bound) + tau * secret_bound)

def _proof_response_norm(
    z_m: List[Poly],
    z_xrec: List[Poly],
    z_yrec: List[Poly],
    z_zrec: List[Poly],
    z_xoa: Any,
    z_yoa: List[Poly],
    z_zoa: List[Poly],
) -> int:
    norm = _polyvec_inf_norm(z_m + z_xrec + z_yrec + z_zrec + z_yoa + z_zoa)
    if isinstance(z_xoa, list):
        norm = max(norm, _polyvec_inf_norm(z_xoa))
    else:
        norm = max(norm, _poly_inf_norm(z_xoa))
    return norm

@dataclass(frozen=True)
class ResponseDecompositionProof:
    bound: int
    weights: Tuple[int, ...]
    coefficient_count: int
    max_abs: int
    digest: bytes


def _response_poly_groups(
    z_m: List[Poly],
    z_xrec: List[Poly],
    z_yrec: List[Poly],
    z_zrec: List[Poly],
    z_xoa: Any,
    z_yoa: List[Poly],
    z_zoa: List[Poly],
):
    yield b"z_m", z_m
    yield b"z_xrec", z_xrec
    yield b"z_yrec", z_yrec
    yield b"z_zrec", z_zrec
    if isinstance(z_xoa, list):
        yield b"z_xoa_list", z_xoa
    else:
        yield b"z_xoa_scalar", [z_xoa]
    yield b"z_yoa", z_yoa
    yield b"z_zoa", z_zoa


def _response_decomposition_digest(
    bound: int,
    z_m: List[Poly],
    z_xrec: List[Poly],
    z_yrec: List[Poly],
    z_zrec: List[Poly],
    z_xoa: Any,
    z_yoa: List[Poly],
    z_zoa: List[Poly],
) -> Tuple[bytes, int, int]:
    weights = tuple(_g_beta(int(bound)))
    h = hashlib.sha256()
    h.update(b"Test2-response-decomposition|")
    h.update(_pack_int(int(bound)))
    h.update(_pack_tuple(weights))
    coefficient_count = 0
    max_abs = 0
    for name, vec in _response_poly_groups(z_m, z_xrec, z_yrec, z_zrec, z_xoa, z_yoa, z_zoa):
        h.update(name)
        h.update(len(vec).to_bytes(4, "little"))
        for poly in vec:
            centered = poly.centered().astype(np.int64, copy=False)
            h.update(poly.ring.n.to_bytes(4, "little"))
            for coeff in centered:
                value = int(coeff)
                abs_value = abs(value)
                if abs_value > int(bound):
                    raise ValueError("response coefficient outside decomposition bound")
                bits = _idec(abs_value, int(bound))
                sign = 1 if value < 0 else 0
                h.update(bytes([sign]))
                h.update(bytes(bits))
                coefficient_count += 1
                if abs_value > max_abs:
                    max_abs = abs_value
    return h.digest(), coefficient_count, max_abs


def _make_response_decomposition_proof(
    bound: int,
    z_m: List[Poly],
    z_xrec: List[Poly],
    z_yrec: List[Poly],
    z_zrec: List[Poly],
    z_xoa: Any,
    z_yoa: List[Poly],
    z_zoa: List[Poly],
) -> ResponseDecompositionProof:
    digest, coefficient_count, max_abs = _response_decomposition_digest(
        bound,
        z_m,
        z_xrec,
        z_yrec,
        z_zrec,
        z_xoa,
        z_yoa,
        z_zoa,
    )
    return ResponseDecompositionProof(
        bound=int(bound),
        weights=tuple(_g_beta(int(bound))),
        coefficient_count=coefficient_count,
        max_abs=max_abs,
        digest=digest,
    )


def verify_response_decomposition(pi: Any) -> bool:
    proof = getattr(pi, "response_decomposition_proof", None)
    if proof is None:
        return False
    if int(getattr(proof, "bound", -1)) != int(getattr(pi, "response_bound", -2)):
        return False
    if tuple(getattr(proof, "weights", ())) != tuple(_g_beta(int(pi.response_bound))):
        return False
    if int(getattr(proof, "max_abs", -1)) > int(pi.response_bound):
        return False
    try:
        digest, coefficient_count, max_abs = _response_decomposition_digest(
            int(pi.response_bound),
            pi.z_m,
            pi.z_xrec,
            pi.z_yrec,
            pi.z_zrec,
            pi.z_xoa,
            pi.z_yoa,
            pi.z_zoa,
        )
    except Exception:
        return False
    return (
        digest == getattr(proof, "digest", b"")
        and coefficient_count == int(getattr(proof, "coefficient_count", -1))
        and max_abs == int(getattr(proof, "max_abs", -1))
    )

def _extract_vec_from_responses(a: List[Poly], b: List[Poly], inv_delta: int) -> List[Poly]:
    if len(a) != len(b):
        raise ValueError("response vector length mismatch")
    return [(b[i] - a[i]).scalar_mul(inv_delta) for i in range(len(a))]


@dataclass(frozen=True)
class PiStatement:

    ring: Ring
    q: int
    k: int
    alpha1: int

    A_user: List[List[Poly]]
    A_oa: List[List[Poly]]
    vk_ots: bytes
    c_rec1: List[Poly]
    c_rec2: List[Poly]
    c_oa1: List[Poly]
    c_oa2: Any
    label: bytes
    cert_digest: bytes = b""
    receiver_hint: Poly | None = None
    AR: Any = None
    uR: Any = None

    def to_bytes(self) -> bytes:
        parts = [
            b"PiStmt|",
            self.label, b"|vk|", self.vk_ots, b"|",
            b"|cert|", _pack_bytes(self.cert_digest),
            b"|hint|", _pack_poly(self.receiver_hint) if self.receiver_hint is not None else b"hint:none",
            _pack_poly_list_list(self.A_user),
            _pack_poly_list_list(self.A_oa),
            _pack_poly_list(self.c_rec1),
            _pack_poly_list(self.c_rec2),
            _pack_poly_list(self.c_oa1),
        ]
        if isinstance(self.c_oa2, list):
            parts.append(_pack_poly_list(self.c_oa2))
        else:
            parts.append(_pack_poly(self.c_oa2))
        if self.AR is not None and self.uR is not None:
            parts.append(b"|AR|")
            parts.append(_pack_poly_list(self.AR))
            parts.append(b"|uR|")
            parts.append(_pack_poly(self.uR))
        return b"".join(parts)


@dataclass(frozen=True)
class PiWitness:

    m: List[Poly]
    x_rec: List[Poly]
    y_rec: List[Poly]
    z_rec: List[Poly]
    x_oa: Any
    y_oa: List[Poly]
    z_oa: List[Poly]


def _statement_from_inputs(pub, pk_oa, ct, label: bytes, witness: Dict[str, Any]) -> Tuple[PiStatement, PiWitness]:
    ring = pub.ring_q
    q = ring.q
    k = pub.params.k
    alpha1 = getattr(pub.params, "alpha1", 1)

    pk_user = witness["pk_user"]

    stmt = PiStatement(
        ring=ring,
        q=q,
        k=k,
        alpha1=alpha1,
        A_user=pk_user.A,
        A_oa=pk_oa.A,
        vk_ots=ct.vk_ots,
        c_rec1=ct.c_rec1,
        c_rec2=ct.c_rec2,
        c_oa1=ct.c_oa1,
        c_oa2=ct.c_oa2,
        label=label,
        cert_digest=_cert_digest(witness.get("cert")),
        receiver_hint=_certificate_receiver_hint(pk_user, witness.get("cert")),
        AR=getattr(pub, "AR", None),
        uR=getattr(pub, "uR", None),
    )
    wit = PiWitness(
        m=witness["m"],
        x_rec=witness["x_rec"],
        y_rec=witness["y_rec"],
        z_rec=witness["z_rec"],
        x_oa=witness["x_oa"],
        y_oa=witness["y_oa"],
        z_oa=witness["z_oa"],
    )
    return stmt, wit


def _statement_from_public(pub, pk_oa, ct, label: bytes, pk_user, cert: Any = None) -> PiStatement:
    return PiStatement(
        ring=pub.ring_q,
        q=pub.ring_q.q,
        k=pub.params.k,
        alpha1=getattr(pub.params, "alpha1", 1),
        A_user=pk_user.A,
        A_oa=pk_oa.A,
        vk_ots=ct.vk_ots,
        c_rec1=ct.c_rec1,
        c_rec2=ct.c_rec2,
        c_oa1=ct.c_oa1,
        c_oa2=ct.c_oa2,
        label=label,
        cert_digest=_cert_digest(cert),
        receiver_hint=_certificate_receiver_hint(pk_user, cert),
        AR=getattr(pub, "AR", None),
        uR=getattr(pub, "uR", None),
    )


def _compute_y(stmt: PiStatement) -> List[Poly]:

    ring = stmt.ring
    alpha1 = stmt.alpha1
    k = stmt.k

    Hvk = H1_mat(ring, stmt.vk_ots, alpha1, k)
    hvk = H2_vec(ring, stmt.vk_ots, alpha1)


    y: List[Poly] = []

    if stmt.AR is not None and stmt.uR is not None:
        y.append(stmt.uR)

    y.extend(stmt.c_rec1)

    y.extend(stmt.c_rec2)

    y.extend(stmt.c_oa1)


    hi0 = stmt.receiver_hint if stmt.receiver_hint is not None else _lsb_poly(stmt.A_user[0][-1])
    enc_hi0 = _encode_msg_scale(hi0, stmt.q)

    if isinstance(stmt.c_oa2, list):
        if len(stmt.c_oa2) != k:
            raise ValueError("c_oa2 list must have length k")

        y.append(stmt.c_oa2[0] - enc_hi0)
        for j in range(1, k):
            y.append(stmt.c_oa2[j])
    else:
        y.append(stmt.c_oa2 - enc_hi0)
    return y


def _apply_M(stmt: PiStatement, vec: PiWitness) -> List[Poly]:

    ring = stmt.ring
    q = stmt.q
    alpha1 = stmt.alpha1
    k = stmt.k

    Hvk = H1_mat(ring, stmt.vk_ots, alpha1, k)
    hvk = H2_vec(ring, stmt.vk_ots, alpha1)

    out: List[Poly] = []

    if stmt.AR is not None and getattr(stmt, "uR", None) is not None:
        out.append(_inner_vec(stmt.AR, vec.m))


    Auz = _matT_vec_mul(stmt.A_user, vec.z_rec)
    for j in range(7 * alpha1):
        out.append(Auz[j] + vec.y_rec[j])


    Htz = _matT_vec_mul_rect(Hvk, vec.z_rec)
    for j in range(k):
        out.append(Htz[j] + vec.x_rec[j] + _encode_msg_scale(vec.m[j], q))


    Aoz = _matT_vec_mul(stmt.A_oa, vec.z_oa)
    for j in range(7 * alpha1):
        out.append(Aoz[j] + vec.y_oa[j])


    base = _inner_vec(hvk, vec.z_oa)
    if isinstance(stmt.c_oa2, list):

        if not isinstance(vec.x_oa, list) or len(vec.x_oa) != k:
            raise ValueError("x_oa must be a list[Poly] of length k when c_oa2 is a list")
        for j in range(k):
            out.append(base + vec.x_oa[j])
    else:

        if isinstance(vec.x_oa, list):
            raise ValueError("x_oa must be a scalar Poly when c_oa2 is scalar")
        out.append(base + vec.x_oa)

    return out


@dataclass(frozen=True)
class PiProof:
    tau: int
    c: int
    t: List[Poly]
    z_m: List[Poly]
    z_xrec: List[Poly]
    z_yrec: List[Poly]
    z_zrec: List[Poly]
    z_xoa: Any
    z_yoa: List[Poly]
    z_zoa: List[Poly]
    audit_commitment: bytes
    audit_opening: bytes
    response_bound: int = 0
    cert_digest: bytes = b""
    relations_digest: bytes = b""
    audit_randomness_proof: Any = None
    response_decomposition_proof: ResponseDecompositionProof | None = None
    audit: Dict[str, Any] | None = None


    def to_bytes(self) -> bytes:
        parts = [b"PiProof|", str(self.tau).encode(), b"|", str(self.c).encode(), b"|"]
        parts.append(_pack_poly_list(self.t))
        parts.append(_pack_poly_list(self.z_m))
        parts.append(_pack_poly_list(self.z_xrec))
        parts.append(_pack_poly_list(self.z_yrec))
        parts.append(_pack_poly_list(self.z_zrec))
        if isinstance(self.z_xoa, list):
            parts.append(_pack_poly_list(self.z_xoa))
        else:
            parts.append(_pack_poly(self.z_xoa))
        parts.append(_pack_poly_list(self.z_yoa))
        parts.append(_pack_poly_list(self.z_zoa))
        parts.append(self.audit_commitment)
        parts.append(_pack_int(self.response_bound))
        parts.append(_pack_bytes(self.cert_digest))
        parts.append(_pack_bytes(self.relations_digest))
        parts.append(_pack_parcom_randomness_proof(self.audit_randomness_proof))
        parts.append(_pack_response_decomposition_proof(self.response_decomposition_proof))
        return b"".join(parts)

    def public_view(self) -> "PiProof":
        return PiProof(
            tau=self.tau,
            c=self.c,
            t=self.t,
            z_m=self.z_m,
            z_xrec=self.z_xrec,
            z_yrec=self.z_yrec,
            z_zrec=self.z_zrec,
            z_xoa=self.z_xoa,
            z_yoa=self.z_yoa,
            z_zoa=self.z_zoa,
            audit_commitment=self.audit_commitment,
            audit_opening=b"",
            response_bound=self.response_bound,
            cert_digest=self.cert_digest,
            relations_digest=self.relations_digest,
            audit_randomness_proof=self.audit_randomness_proof,
            response_decomposition_proof=self.response_decomposition_proof,
            audit=None,
        )


def is_public_proof(pi: Any) -> bool:
    return getattr(pi, "audit", None) is None and getattr(pi, "audit_opening", b"") == b""


def extract_linear_witness(first: PiProof, second: PiProof) -> PiWitness:
    if len(first.t) != len(second.t) or any(first.t[i] != second.t[i] for i in range(len(first.t))):
        raise ValueError("proofs must share the same commitment vector")
    if first.c == second.c:
        raise ValueError("proof challenges must differ")
    if first.z_xoa.__class__ is not second.z_xoa.__class__:
        raise ValueError("x_oa response type mismatch")
    ring = first.t[0].ring
    delta = (int(second.c) - int(first.c)) % ring.q
    try:
        inv_delta = pow(delta, -1, ring.q)
    except ValueError as exc:
        raise ValueError("challenge difference is not invertible modulo q") from exc
    if isinstance(first.z_xoa, list):
        z_xoa = _extract_vec_from_responses(first.z_xoa, second.z_xoa, inv_delta)
    else:
        z_xoa = (second.z_xoa - first.z_xoa).scalar_mul(inv_delta)
    return PiWitness(
        m=_extract_vec_from_responses(first.z_m, second.z_m, inv_delta),
        x_rec=_extract_vec_from_responses(first.z_xrec, second.z_xrec, inv_delta),
        y_rec=_extract_vec_from_responses(first.z_yrec, second.z_yrec, inv_delta),
        z_rec=_extract_vec_from_responses(first.z_zrec, second.z_zrec, inv_delta),
        x_oa=z_xoa,
        y_oa=_extract_vec_from_responses(first.z_yoa, second.z_yoa, inv_delta),
        z_oa=_extract_vec_from_responses(first.z_zoa, second.z_zoa, inv_delta),
    )


def verify_extracted_linear_witness(pub, pk_oa, ct, label: bytes, pk_user, witness: PiWitness, cert: Any = None) -> bool:
    stmt = _statement_from_public(pub, pk_oa, ct, label, pk_user, cert)
    return all(left == right for left, right in zip(_apply_M(stmt, witness), _compute_y(stmt)))


def simulate_linear_transcript(pub, pk_oa, ct, label: bytes, pk_user, challenge: int | None = None, cert: Any = None) -> PiProof:
    stmt = _statement_from_public(pub, pk_oa, ct, label, pk_user, cert)
    ring = stmt.ring
    k = stmt.k
    alpha1 = stmt.alpha1
    tau = int(getattr(pub.params, "tau", 1))
    if challenge is None:
        challenge = int(_uniform_bounded_coeffs(1, tau)[0])
    if abs(int(challenge)) > tau:
        raise ValueError("challenge outside allowed range")
    response_bound = _proof_response_bound(pub)
    z_m = _bounded_vec(ring, k, response_bound)
    z_xrec = _bounded_vec(ring, k, response_bound)
    z_yrec = _bounded_vec(ring, 7 * alpha1, response_bound)
    z_zrec = _bounded_vec(ring, alpha1, response_bound)
    if isinstance(stmt.c_oa2, list):
        z_xoa = _bounded_vec(ring, k, response_bound)
    else:
        z_xoa = _bounded_poly(ring, response_bound)
    z_yoa = _bounded_vec(ring, 7 * alpha1, response_bound)
    z_zoa = _bounded_vec(ring, alpha1, response_bound)
    response = PiWitness(
        m=z_m,
        x_rec=z_xrec,
        y_rec=z_yrec,
        z_rec=z_zrec,
        x_oa=z_xoa,
        y_oa=z_yoa,
        z_oa=z_zoa,
    )
    left = _apply_M(stmt, response)
    y = _compute_y(stmt)
    t_vec = [left[i] - y[i].scalar_mul(int(challenge)) for i in range(len(left))]
    return PiProof(
        tau=tau,
        c=int(challenge),
        t=t_vec,
        z_m=z_m,
        z_xrec=z_xrec,
        z_yrec=z_yrec,
        z_zrec=z_zrec,
        z_xoa=z_xoa,
        z_yoa=z_yoa,
        z_zoa=z_zoa,
        audit_commitment=b"simulated-linear-transcript",
        audit_opening=b"",
        response_bound=response_bound,
        cert_digest=stmt.cert_digest,
        relations_digest=_relations_digest(stmt, response_bound),
        audit_randomness_proof=None,
        response_decomposition_proof=_make_response_decomposition_proof(
            response_bound,
            z_m,
            z_xrec,
            z_yrec,
            z_zrec,
            z_xoa,
            z_yoa,
            z_zoa,
        ),
        audit=None,
    )


def verify_linear_transcript(pub, pk_oa, ct, label: bytes, transcript: PiProof, pk_user, cert: Any = None) -> bool:
    if abs(int(transcript.c)) > int(getattr(pub.params, "tau", 1)):
        return False
    beta = int(getattr(pub.params, "beta", 95255))
    expected_response_bound = _proof_response_bound(pub)
    if int(transcript.response_bound) != expected_response_bound or int(transcript.response_bound) > beta:
        return False
    if _proof_response_norm(
        transcript.z_m,
        transcript.z_xrec,
        transcript.z_yrec,
        transcript.z_zrec,
        transcript.z_xoa,
        transcript.z_yoa,
        transcript.z_zoa,
    ) > transcript.response_bound:
        return False
    if not verify_response_decomposition(transcript):
        return False
    stmt = _statement_from_public(pub, pk_oa, ct, label, pk_user, cert)
    if getattr(transcript, "cert_digest", b"") != stmt.cert_digest:
        return False
    if getattr(transcript, "relations_digest", b"") != _relations_digest(stmt, transcript.response_bound):
        return False
    response = PiWitness(
        m=transcript.z_m,
        x_rec=transcript.z_xrec,
        y_rec=transcript.z_yrec,
        z_rec=transcript.z_zrec,
        x_oa=transcript.z_xoa,
        y_oa=transcript.z_yoa,
        z_oa=transcript.z_zoa,
    )
    left = _apply_M(stmt, response)
    y = _compute_y(stmt)
    right = [transcript.t[i] + y[i].scalar_mul(int(transcript.c)) for i in range(len(transcript.t))]
    return len(left) == len(right) and all(left[i] == right[i] for i in range(len(left)))


def prove(pub, gm_vk_unused, pk_oa, ct, label: bytes, witness: Dict[str, Any]) -> PiProof:
    stmt, wit = _statement_from_inputs(pub, pk_oa, ct, label, witness)
    if not _precheck_private_witness(pub, gm_vk_unused, stmt, wit, witness):
        raise ValueError("zkp prove: invalid private witness relations")
    ring = stmt.ring
    k = stmt.k
    alpha1 = stmt.alpha1
    tau = int(getattr(pub.params, "tau", 1))
    mask_bound = _proof_mask_bound(pub)
    response_bound = _proof_response_bound(pub, mask_bound)

    def add_scaled(rlist: List[Poly], wlist: List[Poly], challenge: int) -> List[Poly]:
        return [rlist[i] + wlist[i].scalar_mul(challenge) for i in range(len(rlist))]

    for _attempt in range(128):
        r_m = _bounded_vec(ring, k, mask_bound)
        r_xrec = _bounded_vec(ring, k, mask_bound)
        r_yrec = _bounded_vec(ring, 7 * alpha1, mask_bound)
        r_zrec = _bounded_vec(ring, alpha1, mask_bound)

        if isinstance(stmt.c_oa2, list):
            r_xoa = _bounded_vec(ring, k, mask_bound)
        else:
            r_xoa = _bounded_poly(ring, mask_bound)
        r_yoa = _bounded_vec(ring, 7 * alpha1, mask_bound)
        r_zoa = _bounded_vec(ring, alpha1, mask_bound)

        r_wit = PiWitness(
            m=r_m,
            x_rec=r_xrec,
            y_rec=r_yrec,
            z_rec=r_zrec,
            x_oa=r_xoa,
            y_oa=r_yoa,
            z_oa=r_zoa,
        )

        t_vec = _apply_M(stmt, r_wit)
        c = _hash_challenge(stmt.to_bytes(), t_vec, tau)
        z_m = add_scaled(r_m, wit.m, c)
        z_xrec = add_scaled(r_xrec, wit.x_rec, c)
        z_yrec = add_scaled(r_yrec, wit.y_rec, c)
        z_zrec = add_scaled(r_zrec, wit.z_rec, c)
        if isinstance(stmt.c_oa2, list):
            z_xoa = add_scaled(r_xoa, wit.x_oa, c)
        else:
            z_xoa = r_xoa + wit.x_oa.scalar_mul(c)
        z_yoa = add_scaled(r_yoa, wit.y_oa, c)
        z_zoa = add_scaled(r_zoa, wit.z_oa, c)
        if _proof_response_norm(z_m, z_xrec, z_yrec, z_zrec, z_xoa, z_yoa, z_zoa) <= response_bound:
            break
    else:
        raise ValueError("zkp prove: rejection sampling failed")

    audit = {
        "m": wit.m,
        "pk_user_A": witness["pk_user"].A,
        "cert": witness.get("cert"),
        "x_rec": wit.x_rec,
        "y_rec": wit.y_rec,
        "z_rec": wit.z_rec,
        "x_oa": wit.x_oa,
        "y_oa": wit.y_oa,
        "z_oa": wit.z_oa,
    }
    audit_commitment, audit_opening, audit_randomness_proof = _make_audit_commitment_with_proof(pub, _pack_audit(audit), tau)
    response_decomposition_proof = _make_response_decomposition_proof(
        response_bound,
        z_m,
        z_xrec,
        z_yrec,
        z_zrec,
        z_xoa,
        z_yoa,
        z_zoa,
    )

    return PiProof(
        tau=tau,
        c=c,
        t=t_vec,
        z_m=z_m,
        z_xrec=z_xrec,
        z_yrec=z_yrec,
        z_zrec=z_zrec,
        z_xoa=z_xoa,
        z_yoa=z_yoa,
        z_zoa=z_zoa,
        audit_commitment=audit_commitment,
        audit_opening=audit_opening,
        response_bound=response_bound,
        cert_digest=stmt.cert_digest,
        relations_digest=_relations_digest(stmt, response_bound),
        audit_randomness_proof=audit_randomness_proof,
        response_decomposition_proof=response_decomposition_proof,
        audit=audit,
    )


def verify(pub, gm_vk_unused, pk_oa, ct, label: bytes, pi: PiProof, pk_user=None, cert: Any = None) -> bool:

    if pk_user is None:
        return False
    if not is_public_proof(pi):
        return False

    ring = pub.ring_q
    q = ring.q
    k = pub.params.k
    alpha1 = getattr(pub.params, "alpha1", 1)

    stmt = PiStatement(
        ring=ring,
        q=q,
        k=k,
        alpha1=alpha1,
        A_user=pk_user.A,
        A_oa=pk_oa.A,
        vk_ots=ct.vk_ots,
        c_rec1=ct.c_rec1,
        c_rec2=ct.c_rec2,
        c_oa1=ct.c_oa1,
        c_oa2=ct.c_oa2,
        label=label,
        cert_digest=_cert_digest(cert),
        receiver_hint=_certificate_receiver_hint(pk_user, cert),
        AR=getattr(pub, "AR", None),
        uR=getattr(pub, "uR", None),
    )


    if int(pi.tau) != int(getattr(pub.params, "tau", 1)):
        return False
    if getattr(pi, "cert_digest", b"") != stmt.cert_digest:
        return False
    if getattr(pi, "relations_digest", b"") != _relations_digest(stmt, int(pi.response_bound)):
        return False
    c = _hash_challenge(stmt.to_bytes(), pi.t, pi.tau)
    if int(c) != int(pi.c):
        return False
    if not verify_audit_randomness_knowledge(pi, getattr(pub, "parcom", None)):
        return False
    beta = int(getattr(pub.params, "beta", 95255))
    expected_response_bound = _proof_response_bound(pub)
    if int(pi.response_bound) != expected_response_bound or int(pi.response_bound) > beta:
        return False
    if _proof_response_norm(pi.z_m, pi.z_xrec, pi.z_yrec, pi.z_zrec, pi.z_xoa, pi.z_yoa, pi.z_zoa) > pi.response_bound:
        return False
    if not verify_response_decomposition(pi):
        return False


    z_wit = PiWitness(
        m=pi.z_m,
        x_rec=pi.z_xrec,
        y_rec=pi.z_yrec,
        z_rec=pi.z_zrec,
        x_oa=pi.z_xoa,
        y_oa=pi.z_yoa,
        z_oa=pi.z_zoa,
    )

    left = _apply_M(stmt, z_wit)
    y = _compute_y(stmt)


    right: List[Poly] = []
    for i in range(len(pi.t)):
        right.append(pi.t[i] + y[i].scalar_mul(c))

    return all(left[i] == right[i] for i in range(len(left)))
