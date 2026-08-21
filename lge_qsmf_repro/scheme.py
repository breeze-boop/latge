from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from .commitment import ParCOMParams, setup_parcom
from .decomposition import delta_beta, rdec, rvdec
from .dm_sig import DMPublicKey, DMSecretKey, DMSignature, dm_keygen, dm_sign, dm_verify, polyvec_inf_norm
from .hash_to_ring import H1_mat, H2_vec
from .linalg import (
    as_column,
    concat_cols,
    first_column,
    mat_add,
    mat_transpose_vec_mul,
    mat_transpose_vec_mul_convolution,
    mat_vec_mul,
    mat_vec_mul_convolution,
    matrix_equal,
    poly_inf_norm,
    random_matrix,
    random_vec,
    small_matrix,
    small_poly,
    small_vec,
    ternary_vec,
    vec_add,
    vec_equal,
    vec_inf_norm,
    vec_sub,
    zero_matrix,
    zero_vec,
)
from .ots import OTSParams, ots_gen, ots_sign, ots_verify
from .params import Params, table_ii_params
from .ring import Poly, Ring
from .trapdoor import TrapdoorKey, TrapdoorRelationProof, gen_trap, precompute_strict_sampler, prove_trapdoor_relation, sample_d_strict, verify_preimage, verify_trapdoor, verify_trapdoor_relation
from .zkp_test2 import prove as zkp_prove, verify as zkp_verify, verify_audit_randomness_knowledge


@dataclass
class FullPublicParams:
    params: Params
    ring_q: Ring
    s_full: int
    q1: int
    q2: int
    A_tags: List[List[List[Poly]]]
    B0: List[List[Poly]]
    B1: List[List[Poly]]
    B2: List[List[Poly]]
    u: List[Poly]
    parcom: ParCOMParams
    AR: List[Poly] | None = None
    uR: Poly | None = None


@dataclass
class FullPublicKey:
    A: List[List[Poly]]


@dataclass
class FullSecretKey:
    trapdoor: TrapdoorKey
    sig_sk: DMSecretKey | None = None


@dataclass
class FullUserPublicKeyProof:
    pi_pki: TrapdoorRelationProof


@dataclass
class FullGMKeypair:
    pk: FullPublicKey
    sk: FullSecretKey
    vk_sig: DMPublicKey


@dataclass
class FullOAKeypair:
    pk: FullPublicKey
    sk: FullSecretKey


@dataclass
class FullCertificate:
    tag: Tuple[int, ...]
    r_i: List[Poly]
    h_i: List[Poly]
    w_i: List[Poly]
    v_i: List[Poly]
    gm_sig: DMSignature
    gm_mu: bytes
    gm_sig_max_vtilde_inf: int


@dataclass
class FullRegistrationEntry:
    pk: FullPublicKey
    cert: FullCertificate


@dataclass
class FullRegistrationTable:
    entries: List[FullRegistrationEntry] = field(default_factory=list)
    open_index: Dict[bytes, FullPublicKey | None] = field(default_factory=dict)

    def add(self, pk: FullPublicKey, cert: FullCertificate) -> None:
        self.entries.append(FullRegistrationEntry(pk=pk, cert=cert))
        key = cert.h_i[0].to_bytes()
        self.open_index[key] = pk if key not in self.open_index else None


@dataclass
class FullGroupPublicKey:
    pp: FullPublicParams
    pk_gm: FullPublicKey
    pk_oa: FullPublicKey
    reg: FullRegistrationTable
    vk_gm_sig: DMPublicKey | None = None


@dataclass
class FullCiphertext:
    vk_ots: bytes
    c_rec1: List[Poly]
    c_rec2: List[Poly]
    c_oa1: List[Poly]
    c_oa2: Poly
    sig: bytes
    sig_verify_cache: set[bytes] = field(default_factory=set, repr=False, compare=False)


def _now_ns() -> int:
    return time.perf_counter_ns()


def _timed(fn, *args, **kwargs):
    t0 = _now_ns()
    out = fn(*args, **kwargs)
    t1 = _now_ns()
    return out, (t1 - t0) / 1e6


def _root_floor(q: int, degree: int) -> int:
    x = int(q ** (1.0 / degree))
    while (x + 1) ** degree <= q:
        x += 1
    while x**degree > q:
        x -= 1
    return x


def _root_ceil(q: int, degree: int) -> int:
    x = _root_floor(q, degree)
    return x if x**degree >= q else x + 1


def setup_init() -> FullPublicParams:
    params = table_ii_params()
    ring = Ring(params.n, params.q)
    alpha = params.alpha
    k = params.k
    s_full = params.s
    q1 = _root_floor(params.q, 3)
    q2 = _root_ceil(params.q, 5)
    A_tags = [random_matrix(ring, alpha, 3 * alpha) for _ in range(params.d + 1)]
    B0 = random_matrix(ring, alpha, k * alpha)
    B1 = random_matrix(ring, alpha, s_full)
    B2 = random_matrix(ring, alpha, s_full * alpha)
    u = random_vec(ring, alpha)
    return FullPublicParams(
        params=params,
        ring_q=ring,
        s_full=s_full,
        q1=q1,
        q2=q2,
        A_tags=A_tags,
        B0=B0,
        B1=B1,
        B2=B2,
        u=u,
        parcom=setup_parcom(b"8.14-Re", ring=ring, message_len=1, rand_cols=max(4, params.k), rand_bound=params.B),
    )


def setup_gm(pp: FullPublicParams) -> FullGMKeypair:
    alpha = pp.params.alpha
    td = gen_trap(pp.ring_q, alpha, 2 * alpha, 3, pp.q1, 1)
    precompute_strict_sampler(td, 40.0, perturbation_cols=(1,))
    dm_K = int(getattr(pp.params, "dm_K", alpha))
    dm_L = int(getattr(pp.params, "dm_L", alpha))
    seed = b"Test2-full-GM-DM|" + str(pp.params.n).encode() + b"|" + str(pp.params.q).encode()
    vk_sig, sk_sig = dm_keygen(pp.ring_q, dm_K, dm_L, pp.params.B, seed)
    return FullGMKeypair(pk=FullPublicKey(td.A), sk=FullSecretKey(td, sk_sig), vk_sig=vk_sig)


def setup_oa(pp: FullPublicParams) -> FullOAKeypair:
    alpha1 = pp.params.alpha1
    td = gen_trap(pp.ring_q, alpha1, 2 * alpha1, 5, pp.q2, 1)
    precompute_strict_sampler(td, 1.0, perturbation_cols=(1,))
    return FullOAKeypair(pk=FullPublicKey(td.A), sk=FullSecretKey(td))


def ukgen(pp: FullPublicParams) -> Tuple[FullPublicKey, FullSecretKey]:
    alpha1 = pp.params.alpha1
    td = gen_trap(pp.ring_q, alpha1, 2 * alpha1, 5, pp.q2, 1)
    return FullPublicKey(td.A), FullSecretKey(td)


def ukgen_with_sampler_precompute(
    pp: FullPublicParams,
    _gm: FullGMKeypair,
    _oa: FullOAKeypair,
) -> Tuple[FullPublicKey, FullSecretKey]:
    pk_i, sk_i = ukgen(pp)
    precompute_strict_sampler(sk_i.trapdoor, 1.0, perturbation_cols=(pp.params.k,))
    return pk_i, sk_i


def prove_user_public_key(pp: FullPublicParams, sk_i: FullSecretKey) -> FullUserPublicKeyProof:
    return FullUserPublicKeyProof(prove_trapdoor_relation(sk_i.trapdoor, pp.params.B, pp.params.tau))


def verify_user_public_key(pp: FullPublicParams, pk_i: FullPublicKey, proof: FullUserPublicKeyProof) -> bool:
    return verify_trapdoor_relation(
        pp.ring_q,
        pk_i.A,
        pp.params.alpha1,
        2 * pp.params.alpha1,
        5,
        pp.q2,
        proof.pi_pki,
        pp.params.B,
    )


def _tag_matrix(pp: FullPublicParams, tag: Tuple[int, ...]) -> List[List[Poly]]:
    out = [list(row) for row in pp.A_tags[0]]
    for j, bit in enumerate(tag, start=1):
        if bit:
            out = mat_add(out, pp.A_tags[j])
    return out


def _certificate_target(pp: FullPublicParams, h_i: List[Poly], r_i: List[Poly]) -> Tuple[List[Poly], List[Poly]]:
    w_raw = vec_add(mat_vec_mul_convolution(pp.B0, r_i), mat_vec_mul_convolution(pp.B1, h_i))
    w_i = rvdec(w_raw, (pp.ring_q.q - 1) // 2)
    u_prime = vec_add(mat_vec_mul_convolution(pp.B2, w_i), pp.u)
    return w_i, u_prime


def _pack_vec(v: List[Poly]) -> bytes:
    return b"".join(poly.to_bytes() for poly in v)


def _pack_matrix(M: List[List[Poly]]) -> bytes:
    return b"".join(_pack_vec(row) for row in M)


def _certificate_message(
    gpk: FullGroupPublicKey,
    pk_i: FullPublicKey,
    tag: Tuple[int, ...],
    r_i: List[Poly],
    h_i: List[Poly],
    w_i: List[Poly],
    v_i: List[Poly],
) -> bytes:
    pp = gpk.pp.params
    parts = [
        b"Test2-full-cert|",
        b"n=", str(pp.n).encode(), b"|q=", str(pp.q).encode(),
        b"|k=", str(pp.k).encode(), b"|alpha=", str(pp.alpha).encode(),
        b"|alpha1=", str(pp.alpha1).encode(), b"|d=", str(pp.d).encode(),
        b"|tag=", bytes(int(bit) & 1 for bit in tag),
        b"|pk_gm=", _pack_matrix(gpk.pk_gm.A),
        b"|pk_i=", _pack_matrix(pk_i.A),
        b"|r=", _pack_vec(r_i),
        b"|h=", _pack_vec(h_i),
        b"|w=", _pack_vec(w_i),
        b"|v=", _pack_vec(v_i),
    ]
    return b"".join(parts)


def join_issue(gpk: FullGroupPublicKey, gm_sk: FullSecretKey, pk_i: FullPublicKey, sk_i: FullSecretKey, pk_proof: FullUserPublicKeyProof | None = None) -> FullCertificate:
    pp = gpk.pp
    if gm_sk.sig_sk is None:
        raise ValueError("missing GM signing secret key")
    if pk_proof is None:
        pk_proof = prove_user_public_key(pp, sk_i)
    if not verify_user_public_key(pp, pk_i, pk_proof):
        raise ValueError("invalid user public-key proof")
    if any(matrix_equal(entry.pk.A, pk_i.A) for entry in gpk.reg.entries):
        raise ValueError("user public key already registered")
    tag_value = len(gpk.reg.entries)
    tag = tuple((tag_value >> j) & 1 for j in range(pp.params.d))
    h_i = rdec(pk_i.A[0][-1], (pp.ring_q.q - 1) // 2)
    At = concat_cols(gpk.pk_gm.A, _tag_matrix(pp, tag))
    v2 = zero_vec(pp.ring_q, 3 * pp.params.alpha)
    last_cert = None
    last_norm = -1
    for _ in range(1024):
        r_i = ternary_vec(pp.ring_q, pp.params.k * pp.params.alpha)
        w_i, u_prime = _certificate_target(pp, h_i, r_i)
        for _ in range(8):
            v1_matrix = sample_d_strict(gm_sk.trapdoor, as_column(u_prime), sigma=40.0)
            v1 = first_column(v1_matrix)
            v_i = v1 + v2
            last_norm = vec_inf_norm(v_i)
            if last_norm > pp.params.beta:
                continue
            if not vec_equal(mat_vec_mul(At, v_i), u_prime):
                continue
            mu = _certificate_message(gpk, pk_i, tag, r_i, h_i, w_i, v_i)
            gm_sig = dm_sign(gm_sk.sig_sk, mu, B=pp.params.B, beta=pp.params.beta, tau=pp.params.tau, max_tries=256)
            cert = FullCertificate(
                tag=tag,
                r_i=r_i,
                h_i=h_i,
                w_i=w_i,
                v_i=v_i,
                gm_sig=gm_sig,
                gm_mu=mu,
                gm_sig_max_vtilde_inf=polyvec_inf_norm(gm_sig.v_tilde),
            )
            last_cert = cert
            gpk.reg.add(pk_i, cert)
            return cert
    max_norm = vec_inf_norm(last_cert.v_i) if last_cert is not None else last_norm
    raise ValueError(f"issued certificate failed verification; last norm={max_norm}, beta={pp.params.beta}")


def verify_certificate(gpk: FullGroupPublicKey, pk_i: FullPublicKey, cert: FullCertificate) -> bool:
    pp = gpk.pp
    if gpk.vk_gm_sig is None:
        return False
    if len(cert.r_i) != pp.params.k * pp.params.alpha:
        return False
    if len(cert.h_i) != pp.s_full:
        return False
    if len(cert.v_i) != 8 * pp.params.alpha:
        return False
    if not vec_equal(rdec(pk_i.A[0][-1], (pp.ring_q.q - 1) // 2), cert.h_i):
        return False
    w_i, u_prime = _certificate_target(pp, cert.h_i, cert.r_i)
    if not vec_equal(w_i, cert.w_i):
        return False
    At = concat_cols(gpk.pk_gm.A, _tag_matrix(pp, cert.tag))
    lhs = mat_vec_mul(At, cert.v_i)
    if not (vec_equal(lhs, u_prime) and vec_inf_norm(cert.v_i) <= pp.params.beta):
        return False
    mu = _certificate_message(gpk, pk_i, cert.tag, cert.r_i, cert.h_i, cert.w_i, cert.v_i)
    if cert.gm_mu != mu:
        return False
    return dm_verify(gpk.vk_gm_sig, mu, cert.gm_sig, beta=pp.params.beta, tau=pp.params.tau)


def gr(pp: FullPublicParams) -> List[Poly]:
    pp.AR = random_vec(pp.ring_q, pp.params.k)
    return pp.AR


def sample_risis(pp: FullPublicParams) -> List[Poly]:
    if pp.AR is None:
        gr(pp)
    msg = ternary_vec(pp.ring_q, pp.params.k)
    acc = pp.ring_q.zero()
    for a, b in zip(pp.AR, msg):
        acc = acc + a * b
    pp.uR = acc
    return msg


def _encode_msg_scale(poly: Poly, q: int) -> Poly:
    return poly.scalar_mul(q // 4)


def _decode_scaled(poly: Poly, q: int, ternary: bool) -> Poly:
    centered = poly.centered().astype(int)
    rounded = [int(round(x * 4.0 / q)) for x in centered]
    if ternary:
        clipped = [max(-1, min(1, x)) for x in rounded]
    else:
        clipped = [0 if x <= 0 else 1 for x in rounded]
    return Poly.from_ints(poly.ring, clipped)


def _ots_message(ct: FullCiphertext, label: bytes) -> bytes:
    h = hashlib.sha256()
    h.update(b"Test2-full-OTS")
    h.update(label)
    for seq in (ct.c_rec1, ct.c_rec2, ct.c_oa1):
        for poly in seq:
            h.update(poly.to_bytes())
    h.update(ct.c_oa2.to_bytes())
    return h.digest()


def verify_sender_sig(ct: FullCiphertext, label: bytes) -> bool:
    msg = _ots_message(ct, label)
    cache_key = hashlib.sha256(b"Test2-sender-sig-cache|" + ct.sig + msg).digest()
    if cache_key in ct.sig_verify_cache:
        return True
    ok = ots_verify(ct.vk_ots, ct.sig, msg)
    if ok:
        ct.sig_verify_cache.add(cache_key)
    return ok


def enc(gpk: FullGroupPublicKey, pk_i: FullPublicKey, cert: FullCertificate, msg: List[Poly], label: bytes) -> Tuple[FullCiphertext, Dict[str, object]]:
    pp = gpk.pp
    if not verify_certificate(gpk, pk_i, cert):
        raise ValueError("invalid certificate")
    if len(msg) != pp.params.k:
        raise ValueError("invalid message length")
    ots = ots_gen(OTSParams(n=pp.params.n, p=pp.params.p))
    Hvk = H1_mat(pp.ring_q, ots.vk_bytes, pp.params.alpha1, pp.params.k)
    hvk = H2_vec(pp.ring_q, ots.vk_bytes, pp.params.alpha1)
    x_rec = small_vec(pp.ring_q, pp.params.k, pp.params.B)
    y_rec = small_vec(pp.ring_q, 7 * pp.params.alpha1, pp.params.B)
    z_rec = small_vec(pp.ring_q, pp.params.alpha1, pp.params.B)
    c_rec1 = vec_add(mat_transpose_vec_mul_convolution(pk_i.A, z_rec), y_rec)
    c_rec2 = vec_add(vec_add(mat_transpose_vec_mul_convolution(Hvk, z_rec), x_rec), [_encode_msg_scale(m, pp.params.q) for m in msg])
    hi0 = cert.h_i[0]
    x_oa = small_poly(pp.ring_q, pp.params.B)
    y_oa = small_vec(pp.ring_q, 7 * pp.params.alpha1, pp.params.B)
    z_oa = small_vec(pp.ring_q, pp.params.alpha1, pp.params.B)
    c_oa1 = vec_add(mat_transpose_vec_mul_convolution(gpk.pk_oa.A, z_oa), y_oa)
    base = pp.ring_q.zero()
    for a, b in zip(hvk, z_oa):
        base = base + a * b
    c_oa2 = base + x_oa + _encode_msg_scale(hi0, pp.params.q)
    ct = FullCiphertext(ots.vk_bytes, c_rec1, c_rec2, c_oa1, c_oa2, b"")
    ct.sig = ots_sign(ots.sk_bytes, _ots_message(ct, label))
    witness = {
        "m": msg,
        "pk_user": pk_i,
        "cert": cert,
        "x_rec": x_rec,
        "y_rec": y_rec,
        "z_rec": z_rec,
        "x_oa": x_oa,
        "y_oa": y_oa,
        "z_oa": z_oa,
    }
    return ct, witness


def dec(pp: FullPublicParams, sk_i: FullSecretKey, ct: FullCiphertext, label: bytes) -> List[Poly]:
    if not verify_sender_sig(ct, label):
        raise ValueError("invalid OTS signature")
    Hvk = H1_mat(pp.ring_q, ct.vk_ots, pp.params.alpha1, pp.params.k)
    S = sample_d_strict(sk_i.trapdoor, Hvk, sigma=1.0, slow_trapdoor_lift=True)
    masked = mat_transpose_vec_mul_convolution(S, ct.c_rec1)
    return [_decode_scaled(ct.c_rec2[j] - masked[j], pp.params.q, True) for j in range(pp.params.k)]


def open_receiver(gpk: FullGroupPublicKey, sk_oa: FullSecretKey, ct: FullCiphertext, label: bytes) -> FullPublicKey | None:
    pp = gpk.pp
    if not verify_sender_sig(ct, label):
        raise ValueError("invalid OTS signature")
    hvk = H2_vec(pp.ring_q, ct.vk_ots, pp.params.alpha1)
    S = sample_d_strict(sk_oa.trapdoor, as_column(hvk), sigma=1.0)
    s = first_column(S)
    masked = pp.ring_q.zero()
    for a, b in zip(s, ct.c_oa1):
        masked = masked + a * b
    hi0 = _decode_scaled(ct.c_oa2 - masked, pp.params.q, True)
    return gpk.reg.open_index.get(hi0.to_bytes())


def _coin_bounds_ok(audit: Dict[str, object], pp: FullPublicParams) -> bool:
    bound = pp.params.B
    cert = audit["cert"]
    checks = [
        vec_inf_norm(audit["m"]) <= 1,
        vec_inf_norm(cert.r_i) <= 1,
        vec_inf_norm(audit["x_rec"]) <= bound,
        vec_inf_norm(audit["y_rec"]) <= bound,
        vec_inf_norm(audit["z_rec"]) <= bound,
        vec_inf_norm(audit["y_oa"]) <= bound,
        vec_inf_norm(audit["z_oa"]) <= bound,
    ]
    x_oa = audit["x_oa"]
    checks.append(vec_inf_norm(x_oa) <= bound if isinstance(x_oa, list) else poly_inf_norm(x_oa) <= bound)
    return all(checks)


def _verify_paper_relations(
    gpk: FullGroupPublicKey,
    pk_i: FullPublicKey,
    ct: FullCiphertext,
    label: bytes,
    witness: Dict[str, object],
    proof,
) -> Dict[str, bool]:
    pp = gpk.pp
    cert = witness["cert"]
    msg = witness["m"]
    x_rec = witness["x_rec"]
    y_rec = witness["y_rec"]
    z_rec = witness["z_rec"]
    x_oa = witness["x_oa"]
    y_oa = witness["y_oa"]
    z_oa = witness["z_oa"]
    Hvk = H1_mat(pp.ring_q, ct.vk_ots, pp.params.alpha1, pp.params.k)
    hvk = H2_vec(pp.ring_q, ct.vk_ots, pp.params.alpha1)
    w_i, u_prime = _certificate_target(pp, cert.h_i, cert.r_i)
    At = concat_cols(gpk.pk_gm.A, _tag_matrix(pp, cert.tag))
    rec1 = vec_add(mat_transpose_vec_mul_convolution(pk_i.A, z_rec), y_rec)
    rec2 = vec_add(
        vec_add(mat_transpose_vec_mul_convolution(Hvk, z_rec), x_rec),
        [_encode_msg_scale(m, pp.params.q) for m in msg],
    )
    oa1 = vec_add(mat_transpose_vec_mul_convolution(gpk.pk_oa.A, z_oa), y_oa)
    oa_base = pp.ring_q.zero()
    for a, b in zip(hvk, z_oa):
        oa_base = oa_base + a * b
    oa2 = oa_base + x_oa + _encode_msg_scale(cert.h_i[0], pp.params.q)
    reg_matches = [
        entry
        for entry in gpk.reg.entries
        if matrix_equal(entry.pk.A, pk_i.A) and verify_certificate(gpk, entry.pk, entry.cert)
    ]
    return {
        "public_proof_has_no_audit": getattr(proof, "audit", None) is None and getattr(proof, "audit_opening", b"") == b"",
        "parcom_randomness_knowledge": verify_audit_randomness_knowledge(proof, pp.parcom),
        "pk_matches_witness": matrix_equal(witness["pk_user"].A, pk_i.A),
        "registered_cert": len(reg_matches) == 1,
        "relation_1_AR_m": pp.AR is not None and pp.uR is not None and _inner_public(pp.AR, msg) == pp.uR,
        "relation_2_h": vec_equal(cert.h_i, rdec(pk_i.A[0][-1], (pp.ring_q.q - 1) // 2)),
        "relation_3_uprime": vec_equal(w_i, cert.w_i) and vec_equal(vec_sub(u_prime, pp.u), mat_vec_mul_convolution(pp.B2, cert.w_i)),
        "relation_4_cert": verify_certificate(gpk, pk_i, cert) and vec_equal(mat_vec_mul(At, cert.v_i), u_prime),
        "relation_5_bounds": _coin_bounds_ok(witness, pp) and vec_inf_norm(cert.v_i) <= pp.params.beta,
        "relation_6_enc_rec": vec_equal(rec1, ct.c_rec1) and vec_equal(rec2, ct.c_rec2),
        "relation_6_enc_oa": vec_equal(oa1, ct.c_oa1) and oa2 == ct.c_oa2,
        "ots": verify_sender_sig(ct, label),
    }


def _inner_public(a: List[Poly], b: List[Poly]) -> Poly:
    acc = a[0].ring.zero()
    for left, right in zip(a, b):
        acc = acc + left * right
    return acc


def prove(gpk: FullGroupPublicKey, pk_i: FullPublicKey, ct: FullCiphertext, label: bytes, witness: Dict[str, object]):
    if not verify_certificate(gpk, pk_i, witness["cert"]):
        raise ValueError("invalid proof certificate")
    return zkp_prove(gpk.pp, gpk, gpk.pk_oa, ct, label, witness).public_view()


def verify(gpk: FullGroupPublicKey, pk_i: FullPublicKey, ct: FullCiphertext, label: bytes, proof) -> bool:
    if gpk.pp.AR is None or gpk.pp.uR is None:
        return False
    matching = [entry.cert for entry in gpk.reg.entries if matrix_equal(entry.pk.A, pk_i.A)]
    if len(matching) != 1 or not verify_certificate(gpk, pk_i, matching[0]):
        return False
    return zkp_verify(gpk.pp, gpk, gpk.pk_oa, ct, label, proof, pk_i, cert=matching[0])


def run_full_pipeline() -> Dict[str, object]:
    timings: Dict[str, float] = {}
    pp, timings["SETUPinit"] = _timed(setup_init)
    gm, timings["SETUPGM"] = _timed(setup_gm, pp)
    oa, timings["SETUPOA"] = _timed(setup_oa, pp)
    reg = FullRegistrationTable()
    gpk = FullGroupPublicKey(pp=pp, pk_gm=gm.pk, pk_oa=oa.pk, reg=reg, vk_gm_sig=gm.vk_sig)
    user_keys, timings["UKGEN"] = _timed(ukgen_with_sampler_precompute, pp, gm, oa)
    pk_i, sk_i = user_keys
    pk_proof, timings["JOIN_PROVE_PK"] = _timed(prove_user_public_key, pp, sk_i)
    cert, timings["JOIN_ISSUE"] = _timed(join_issue, gpk, gm.sk, pk_i, sk_i, pk_proof)
    _, timings["Gr"] = _timed(gr, pp)
    msg, timings["sampleR"] = _timed(sample_risis, pp)
    label = b"test2-full"
    ct_wit, timings["ENC"] = _timed(enc, gpk, pk_i, cert, msg, label)
    ct, witness = ct_wit
    pi, timings["P"] = _timed(prove, gpk, pk_i, ct, label, witness)
    ok, timings["V"] = _timed(verify, gpk, pk_i, ct, label, pi)
    m_out, timings["DEC"] = _timed(dec, pp, sk_i, ct, label)
    opened, timings["OPEN"] = _timed(open_receiver, gpk, oa.sk, ct, label)
    paper_relation_checks = _verify_paper_relations(gpk, pk_i, ct, label, witness, pi)
    checks = {
        "registered_users": len(gpk.reg.entries) == 1,
        "gm_trapdoor": verify_trapdoor(gm.sk.trapdoor),
        "oa_trapdoor": verify_trapdoor(oa.sk.trapdoor),
        "user_trapdoor": verify_trapdoor(sk_i.trapdoor),
        "user_pk_proof": verify_user_public_key(pp, pk_i, pk_proof),
        "certificate": verify_certificate(gpk, pk_i, cert),
        "zkp": bool(ok),
        "zkp_public_proof_redacted": getattr(pi, "audit", None) is None and getattr(pi, "audit_opening", b"") == b"",
        "zkp_response_bound": getattr(pi, "response_bound", 0) > 0,
        "paper_relations": all(paper_relation_checks.values()),
        "dec_matches": vec_equal(m_out, msg),
        "open_matches": opened is pk_i,
    }
    return {
        "timings_ms": timings,
        "checks": checks,
        "paper_relation_checks": paper_relation_checks,
    }


def main() -> None:
    report = run_full_pipeline()
    print("[actual_timings_ms]")
    for name, elapsed_ms in report["timings_ms"].items():
        print(f"{name}: {elapsed_ms:.3f} ms")
    print("[checks]")
    for name, ok in report["checks"].items():
        print(f"{name}: {ok}")
    print("[paper_relation_checks]")
    for name, ok in report["paper_relation_checks"].items():
        print(f"{name}: {ok}")


if __name__ == "__main__":
    main()
