from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from typing import List

from .linalg import const_poly, mat_vec_mul_convolution, random_matrix, small_vec, vec_add, vec_inf_norm, vec_sub
from .ring import Poly, Ring

@dataclass(frozen=True)
class ParCOMParams:
    domain: bytes
    digest_size: int = 32
    ring: Ring | None = None
    message_len: int = 0
    rand_cols: int = 0
    rand_bound: int = 1
    A1: List[List[Poly]] | None = None
    A2: List[List[Poly]] | None = None

    def commit_payload(self, payload: bytes) -> tuple[bytes, bytes]:
        commitment = commit(self, payload)
        return commitment.value, commitment.opening

    def verify_payload(self, commitment_value: bytes, opening: bytes, payload: bytes) -> bool:
        return verify_opening(self, Commitment(commitment_value, opening), payload)

    def prove_opening_randomness_knowledge(self, commitment_value: bytes, opening: bytes, tau: int = 60) -> "RelaxedRandomnessKnowledgeProof":
        return prove_relaxed_randomness_knowledge(self, Commitment(commitment_value, opening), tau=tau)

    def verify_opening_randomness_knowledge(self, commitment_value: bytes, proof: "RelaxedRandomnessKnowledgeProof") -> bool:
        return verify_relaxed_randomness_knowledge(self, commitment_value, proof)


@dataclass(frozen=True)
class Commitment:
    value: bytes
    opening: bytes


@dataclass(frozen=True)
class RelaxedOpening:
    randomness: List[Poly]
    f: Poly


@dataclass(frozen=True)
class RelaxedOpeningKnowledgeProof:
    tau: int
    challenge: int
    f: Poly
    commitment_1: List[Poly]
    commitment_2: List[Poly]
    response: List[Poly]
    mask_bound: int = 0
    response_bound: int = 0
    response_norm: int = 0


@dataclass(frozen=True)
class RelaxedRandomnessKnowledgeProof:
    tau: int
    challenge: int
    f: Poly
    commitment_1: List[Poly]
    response: List[Poly]
    mask_bound: int = 0
    response_bound: int = 0
    response_norm: int = 0


def _pack_u32(x: int) -> bytes:
    return int(x).to_bytes(4, "little", signed=False)


def _pack_poly_list(v: List[Poly]) -> bytes:
    return b"".join(poly.to_bytes() for poly in v)


def _unpack_poly_list(ring: Ring, raw: bytes, count: int) -> List[Poly]:
    step = len(ring.zero().to_bytes())
    if len(raw) != count * step:
        raise ValueError("bad poly list length")
    return [Poly.from_bytes(ring, raw[i * step : (i + 1) * step]) for i in range(count)]


def _payload_to_message(params: ParCOMParams, payload: bytes) -> List[Poly]:
    if params.ring is None or params.message_len <= 0:
        raise ValueError("parCOM has no lattice parameters")
    ring = params.ring
    digest = hashlib.shake_256(params.domain + b"|msg|" + payload).digest(8 * ring.n * params.message_len)
    out: List[Poly] = []
    pos = 0
    for _ in range(params.message_len):
        coeffs = []
        for _ in range(ring.n):
            coeffs.append(int.from_bytes(digest[pos : pos + 8], "little") % ring.q)
            pos += 8
        out.append(Poly.from_ints(ring, coeffs))
    return out


def _serialize_lattice_commitment(c1: List[Poly], c2: List[Poly]) -> bytes:
    return b"PCOM1" + _pack_u32(len(c1)) + _pack_u32(len(c2)) + _pack_poly_list(c1) + _pack_poly_list(c2)


def _parse_lattice_commitment(params: ParCOMParams, raw: bytes) -> tuple[List[Poly], List[Poly]]:
    if params.ring is None or not raw.startswith(b"PCOM1"):
        raise ValueError("bad lattice commitment")
    step = len(params.ring.zero().to_bytes())
    n1 = int.from_bytes(raw[5:9], "little")
    n2 = int.from_bytes(raw[9:13], "little")
    body = raw[13:]
    if len(body) != (n1 + n2) * step:
        raise ValueError("bad lattice commitment length")
    c1 = _unpack_poly_list(params.ring, body[: n1 * step], n1)
    c2 = _unpack_poly_list(params.ring, body[n1 * step :], n2)
    return c1, c2


def _serialize_opening(r: List[Poly]) -> bytes:
    return b"POPEN1" + _pack_u32(len(r)) + _pack_poly_list(r)


def _serialize_relaxed_opening(opening: RelaxedOpening) -> bytes:
    return b"PROPN1" + _pack_u32(len(opening.randomness)) + opening.f.to_bytes() + _pack_poly_list(opening.randomness)


def _parse_opening(params: ParCOMParams, raw: bytes) -> List[Poly]:
    if params.ring is None or not raw.startswith(b"POPEN1"):
        raise ValueError("bad lattice opening")
    count = int.from_bytes(raw[6:10], "little")
    return _unpack_poly_list(params.ring, raw[10:], count)


def _parse_relaxed_opening(params: ParCOMParams, raw: bytes) -> RelaxedOpening:
    if params.ring is None:
        raise ValueError("parCOM has no lattice parameters")
    if raw.startswith(b"POPEN1"):
        return RelaxedOpening(randomness=_parse_opening(params, raw), f=params.ring.one())
    if not raw.startswith(b"PROPN1"):
        raise ValueError("bad relaxed lattice opening")
    step = len(params.ring.zero().to_bytes())
    count = int.from_bytes(raw[6:10], "little")
    start = 10
    f = Poly.from_bytes(params.ring, raw[start : start + step])
    randomness = _unpack_poly_list(params.ring, raw[start + step :], count)
    return RelaxedOpening(randomness=randomness, f=f)


def decode_relaxed_opening(params: ParCOMParams, opening: bytes) -> RelaxedOpening:
    return _parse_relaxed_opening(params, opening)


def _poly_vec_mul(poly: Poly, vec: List[Poly]) -> List[Poly]:
    return [poly * item for item in vec]


def _vec_scalar_mul(vec: List[Poly], scalar: int) -> List[Poly]:
    return [poly.scalar_mul(scalar) for poly in vec]


def _poly_l1_norm(poly: Poly) -> int:
    return int(sum(abs(int(x)) for x in poly.centered().tolist()))


def _default_mask_bound(params: ParCOMParams, mask_bound: int | None = None) -> int:
    if mask_bound is None:
        return max(1, int(params.rand_bound))
    return max(1, int(mask_bound))


def relaxed_response_bound(params: ParCOMParams, f: Poly, tau: int, mask_bound: int | None = None) -> int:
    mask = _default_mask_bound(params, mask_bound)
    opening_bound = _poly_l1_norm(f) * int(params.rand_bound)
    return int(mask + int(tau) * opening_bound)


def _response_norm(response: List[Poly]) -> int:
    return vec_inf_norm(response)


def _response_bound_fields_ok(params: ParCOMParams, proof) -> bool:
    mask_bound = int(getattr(proof, "mask_bound", 0))
    response_bound = int(getattr(proof, "response_bound", 0))
    response_norm = int(getattr(proof, "response_norm", -1))
    if mask_bound <= 0 or response_bound <= 0 or response_norm < 0:
        return False
    expected = relaxed_response_bound(params, proof.f, int(proof.tau), mask_bound)
    actual = _response_norm(proof.response)
    return response_bound == expected and response_norm == actual and actual <= response_bound


def _extract_vec_from_responses(first: List[Poly], second: List[Poly], inv_delta: int) -> List[Poly]:
    if len(first) != len(second):
        raise ValueError("response vector length mismatch")
    return [(second[i] - first[i]).scalar_mul(inv_delta) for i in range(len(first))]


def _challenge_inverse(first_challenge: int, second_challenge: int, ring: Ring) -> int:
    delta = (int(second_challenge) - int(first_challenge)) % ring.q
    if delta == 0:
        raise ValueError("proof challenges must differ")
    try:
        return pow(delta, -1, ring.q)
    except ValueError as exc:
        raise ValueError("challenge difference is not invertible modulo q") from exc


def _sample_challenge(tau: int) -> int:
    span = 2 * int(tau) + 1
    return int((int.from_bytes(os.urandom(8), "little") % span) - int(tau))


def _hash_opening_challenge(
    params: ParCOMParams,
    commitment: Commitment,
    message: bytes,
    f: Poly,
    commitment_1: List[Poly],
    commitment_2: List[Poly],
    tau: int,
) -> int:
    digest = hashlib.sha256(
        b"Test2-parCOM-opening-proof|"
        + params.domain
        + commitment.value
        + message
        + f.to_bytes()
        + _pack_poly_list(commitment_1)
        + _pack_poly_list(commitment_2)
    ).digest()
    return int((int.from_bytes(digest[:8], "little", signed=False) % (2 * int(tau) + 1)) - int(tau))


def _hash_randomness_challenge(
    params: ParCOMParams,
    commitment_value: bytes,
    f: Poly,
    commitment_1: List[Poly],
    tau: int,
) -> int:
    digest = hashlib.sha256(
        b"Test2-parCOM-randomness-proof|"
        + params.domain
        + commitment_value
        + f.to_bytes()
        + _pack_poly_list(commitment_1)
    ).digest()
    return int((int.from_bytes(digest[:8], "little", signed=False) % (2 * int(tau) + 1)) - int(tau))


def setup_parcom(label: bytes, ring: Ring | None = None, message_len: int = 1, rand_cols: int | None = None, rand_bound: int = 1) -> ParCOMParams:
    domain = hashlib.sha256(b"Test2-parCOM|" + label).digest()
    if ring is None:
        return ParCOMParams(domain=domain)
    if message_len <= 0:
        raise ValueError("message_len must be positive")
    if rand_cols is None:
        rand_cols = message_len + 2
    if rand_cols < message_len + 1:
        raise ValueError("rand_cols must leave room for the identity/message blocks")
    one = const_poly(ring, 1)
    A1_tail = random_matrix(ring, 1, rand_cols - 1)
    A1 = [[one] + A1_tail[0]]
    A2_tail_cols = rand_cols - message_len - 1
    A2_tail = random_matrix(ring, message_len, A2_tail_cols) if A2_tail_cols else [[] for _ in range(message_len)]
    A2: List[List[Poly]] = []
    for row in range(message_len):
        prefix = [ring.zero()]
        ident = [one if row == col else ring.zero() for col in range(message_len)]
        A2.append(prefix + ident + A2_tail[row])
    return ParCOMParams(
        domain=domain,
        ring=ring,
        message_len=message_len,
        rand_cols=rand_cols,
        rand_bound=rand_bound,
        A1=A1,
        A2=A2,
    )


def commit(params: ParCOMParams, message: bytes) -> Commitment:
    if params.ring is not None:
        if params.A1 is None or params.A2 is None:
            raise ValueError("missing parCOM matrices")
        x = _payload_to_message(params, message)
        r = small_vec(params.ring, params.rand_cols, params.rand_bound)
        c1 = mat_vec_mul_convolution(params.A1, r)
        c2 = vec_add(mat_vec_mul_convolution(params.A2, r), x)
        return Commitment(value=_serialize_lattice_commitment(c1, c2), opening=_serialize_opening(r))
    opening = os.urandom(params.digest_size)
    value = hashlib.sha256(params.domain + opening + message).digest()
    return Commitment(value=value, opening=opening)


def relaxed_open(params: ParCOMParams, commitment: Commitment, message: bytes, f: Poly | None = None) -> Commitment:
    if params.ring is None:
        return commitment
    if f is None:
        f = params.ring.one()
    r = _parse_opening(params, commitment.opening)
    opening = RelaxedOpening(randomness=_poly_vec_mul(f, r), f=f)
    return Commitment(value=commitment.value, opening=_serialize_relaxed_opening(opening))


def verify_relaxed_opening(params: ParCOMParams, commitment: Commitment, message: bytes) -> bool:
    if params.ring is None:
        return verify_opening(params, commitment, message)
    if params.A1 is None or params.A2 is None:
        return False
    try:
        c1, c2 = _parse_lattice_commitment(params, commitment.value)
        opening = _parse_relaxed_opening(params, commitment.opening)
    except Exception:
        return False
    if len(opening.randomness) != params.rand_cols:
        return False
    x = _payload_to_message(params, message)
    expected_c1 = mat_vec_mul_convolution(params.A1, opening.randomness)
    expected_c2 = vec_add(mat_vec_mul_convolution(params.A2, opening.randomness), _poly_vec_mul(opening.f, x))
    return _poly_vec_mul(opening.f, c1) == expected_c1 and _poly_vec_mul(opening.f, c2) == expected_c2


def verify_relaxed_opening_witness(
    params: ParCOMParams,
    commitment_value: bytes,
    message: bytes,
    opening: RelaxedOpening,
) -> bool:
    if params.ring is None or params.A1 is None or params.A2 is None:
        return False
    if len(opening.randomness) != params.rand_cols:
        return False
    try:
        c1, c2 = _parse_lattice_commitment(params, commitment_value)
    except Exception:
        return False
    x = _payload_to_message(params, message)
    lhs_1 = mat_vec_mul_convolution(params.A1, opening.randomness)
    lhs_2 = mat_vec_mul_convolution(params.A2, opening.randomness)
    rhs_1 = _poly_vec_mul(opening.f, c1)
    rhs_2 = vec_sub(_poly_vec_mul(opening.f, c2), _poly_vec_mul(opening.f, x))
    return lhs_1 == rhs_1 and lhs_2 == rhs_2


def verify_relaxed_randomness_witness(
    params: ParCOMParams,
    commitment_value: bytes,
    opening: RelaxedOpening,
) -> bool:
    if params.ring is None or params.A1 is None:
        return False
    if len(opening.randomness) != params.rand_cols:
        return False
    try:
        c1, _ = _parse_lattice_commitment(params, commitment_value)
    except Exception:
        return False
    lhs_1 = mat_vec_mul_convolution(params.A1, opening.randomness)
    rhs_1 = _poly_vec_mul(opening.f, c1)
    return lhs_1 == rhs_1


def prove_relaxed_opening_knowledge(
    params: ParCOMParams,
    commitment: Commitment,
    message: bytes,
    tau: int = 60,
    mask_bound: int | None = None,
    max_tries: int = 128,
) -> RelaxedOpeningKnowledgeProof:
    if params.ring is None or params.A1 is None or params.A2 is None:
        raise ValueError("parCOM has no lattice parameters")
    if not verify_relaxed_opening(params, commitment, message):
        raise ValueError("invalid relaxed opening")
    opening = _parse_relaxed_opening(params, commitment.opening)
    mask_bound = _default_mask_bound(params, mask_bound)
    response_bound = relaxed_response_bound(params, opening.f, tau, mask_bound)
    for _attempt in range(max_tries):
        y = small_vec(params.ring, params.rand_cols, mask_bound)
        commitment_1 = mat_vec_mul_convolution(params.A1, y)
        commitment_2 = mat_vec_mul_convolution(params.A2, y)
        challenge = _hash_opening_challenge(params, commitment, message, opening.f, commitment_1, commitment_2, tau)
        response = vec_add(y, _vec_scalar_mul(opening.randomness, challenge))
        response_norm = _response_norm(response)
        if response_norm <= response_bound:
            break
    else:
        raise ValueError("parCOM opening proof: rejection sampling failed")
    return RelaxedOpeningKnowledgeProof(
        tau=tau,
        challenge=challenge,
        f=opening.f,
        commitment_1=commitment_1,
        commitment_2=commitment_2,
        response=response,
        mask_bound=mask_bound,
        response_bound=response_bound,
        response_norm=response_norm,
    )


def verify_relaxed_opening_knowledge(
    params: ParCOMParams,
    commitment: Commitment,
    message: bytes,
    proof: RelaxedOpeningKnowledgeProof,
) -> bool:
    if params.ring is None or params.A1 is None or params.A2 is None:
        return False
    try:
        c1, c2 = _parse_lattice_commitment(params, commitment.value)
    except Exception:
        return False
    if proof.tau <= 0:
        return False
    if len(proof.response) != params.rand_cols:
        return False
    if not _response_bound_fields_ok(params, proof):
        return False
    if proof.challenge != _hash_opening_challenge(params, commitment, message, proof.f, proof.commitment_1, proof.commitment_2, proof.tau):
        return False
    x = _payload_to_message(params, message)
    target_1 = _poly_vec_mul(proof.f, c1)
    target_2 = vec_sub(_poly_vec_mul(proof.f, c2), _poly_vec_mul(proof.f, x))
    lhs_1 = mat_vec_mul_convolution(params.A1, proof.response)
    lhs_2 = mat_vec_mul_convolution(params.A2, proof.response)
    rhs_1 = vec_add(proof.commitment_1, _vec_scalar_mul(target_1, proof.challenge))
    rhs_2 = vec_add(proof.commitment_2, _vec_scalar_mul(target_2, proof.challenge))
    return lhs_1 == rhs_1 and lhs_2 == rhs_2


def verify_relaxed_opening_transcript(
    params: ParCOMParams,
    commitment: Commitment,
    message: bytes,
    proof: RelaxedOpeningKnowledgeProof,
) -> bool:
    if params.ring is None or params.A1 is None or params.A2 is None:
        return False
    try:
        c1, c2 = _parse_lattice_commitment(params, commitment.value)
    except Exception:
        return False
    if proof.tau <= 0 or abs(int(proof.challenge)) > int(proof.tau):
        return False
    if len(proof.response) != params.rand_cols:
        return False
    if not _response_bound_fields_ok(params, proof):
        return False
    x = _payload_to_message(params, message)
    target_1 = _poly_vec_mul(proof.f, c1)
    target_2 = vec_sub(_poly_vec_mul(proof.f, c2), _poly_vec_mul(proof.f, x))
    lhs_1 = mat_vec_mul_convolution(params.A1, proof.response)
    lhs_2 = mat_vec_mul_convolution(params.A2, proof.response)
    rhs_1 = vec_add(proof.commitment_1, _vec_scalar_mul(target_1, proof.challenge))
    rhs_2 = vec_add(proof.commitment_2, _vec_scalar_mul(target_2, proof.challenge))
    return lhs_1 == rhs_1 and lhs_2 == rhs_2


def simulate_relaxed_opening_transcript(
    params: ParCOMParams,
    commitment: Commitment,
    message: bytes,
    challenge: int | None = None,
    f: Poly | None = None,
    response_bound: int | None = None,
    tau: int = 60,
) -> RelaxedOpeningKnowledgeProof:
    if params.ring is None or params.A1 is None or params.A2 is None:
        raise ValueError("parCOM has no lattice parameters")
    if challenge is None:
        challenge = _sample_challenge(tau)
    if abs(int(challenge)) > int(tau):
        raise ValueError("challenge outside allowed range")
    if f is None:
        f = params.ring.one()
    if response_bound is None:
        response_bound = relaxed_response_bound(params, f, tau)
    c1, c2 = _parse_lattice_commitment(params, commitment.value)
    response = small_vec(params.ring, params.rand_cols, response_bound)
    response_norm = _response_norm(response)
    x = _payload_to_message(params, message)
    target_1 = _poly_vec_mul(f, c1)
    target_2 = vec_sub(_poly_vec_mul(f, c2), _poly_vec_mul(f, x))
    commitment_1 = vec_sub(mat_vec_mul_convolution(params.A1, response), _vec_scalar_mul(target_1, int(challenge)))
    commitment_2 = vec_sub(mat_vec_mul_convolution(params.A2, response), _vec_scalar_mul(target_2, int(challenge)))
    return RelaxedOpeningKnowledgeProof(
        tau=int(tau),
        challenge=int(challenge),
        f=f,
        commitment_1=commitment_1,
        commitment_2=commitment_2,
        response=response,
        mask_bound=_default_mask_bound(params),
        response_bound=response_bound,
        response_norm=response_norm,
    )


def extract_relaxed_opening_witness(
    params: ParCOMParams,
    first: RelaxedOpeningKnowledgeProof,
    second: RelaxedOpeningKnowledgeProof,
) -> RelaxedOpening:
    if params.ring is None:
        raise ValueError("parCOM has no lattice parameters")
    if first.f != second.f:
        raise ValueError("proofs must share the same relaxation polynomial")
    if first.commitment_1 != second.commitment_1 or first.commitment_2 != second.commitment_2:
        raise ValueError("proofs must share the same first-round commitments")
    inv_delta = _challenge_inverse(first.challenge, second.challenge, params.ring)
    return RelaxedOpening(
        randomness=_extract_vec_from_responses(first.response, second.response, inv_delta),
        f=first.f,
    )


def prove_relaxed_randomness_knowledge(
    params: ParCOMParams,
    commitment: Commitment,
    tau: int = 60,
    mask_bound: int | None = None,
    max_tries: int = 128,
) -> RelaxedRandomnessKnowledgeProof:
    if params.ring is None or params.A1 is None:
        raise ValueError("parCOM has no lattice parameters")
    opening = _parse_relaxed_opening(params, commitment.opening)
    mask_bound = _default_mask_bound(params, mask_bound)
    response_bound = relaxed_response_bound(params, opening.f, tau, mask_bound)
    for _attempt in range(max_tries):
        y = small_vec(params.ring, params.rand_cols, mask_bound)
        commitment_1 = mat_vec_mul_convolution(params.A1, y)
        challenge = _hash_randomness_challenge(params, commitment.value, opening.f, commitment_1, tau)
        response = vec_add(y, _vec_scalar_mul(opening.randomness, challenge))
        response_norm = _response_norm(response)
        if response_norm <= response_bound:
            break
    else:
        raise ValueError("parCOM randomness proof: rejection sampling failed")
    return RelaxedRandomnessKnowledgeProof(
        tau=tau,
        challenge=challenge,
        f=opening.f,
        commitment_1=commitment_1,
        response=response,
        mask_bound=mask_bound,
        response_bound=response_bound,
        response_norm=response_norm,
    )


def verify_relaxed_randomness_knowledge(
    params: ParCOMParams,
    commitment_value: bytes,
    proof: RelaxedRandomnessKnowledgeProof,
) -> bool:
    if params.ring is None or params.A1 is None:
        return False
    try:
        c1, _ = _parse_lattice_commitment(params, commitment_value)
    except Exception:
        return False
    if proof.tau <= 0:
        return False
    if len(proof.response) != params.rand_cols:
        return False
    if not _response_bound_fields_ok(params, proof):
        return False
    if proof.challenge != _hash_randomness_challenge(params, commitment_value, proof.f, proof.commitment_1, proof.tau):
        return False
    target_1 = _poly_vec_mul(proof.f, c1)
    lhs_1 = mat_vec_mul_convolution(params.A1, proof.response)
    rhs_1 = vec_add(proof.commitment_1, _vec_scalar_mul(target_1, proof.challenge))
    return lhs_1 == rhs_1


def verify_relaxed_randomness_transcript(
    params: ParCOMParams,
    commitment_value: bytes,
    proof: RelaxedRandomnessKnowledgeProof,
) -> bool:
    if params.ring is None or params.A1 is None:
        return False
    try:
        c1, _ = _parse_lattice_commitment(params, commitment_value)
    except Exception:
        return False
    if proof.tau <= 0 or abs(int(proof.challenge)) > int(proof.tau):
        return False
    if len(proof.response) != params.rand_cols:
        return False
    if not _response_bound_fields_ok(params, proof):
        return False
    target_1 = _poly_vec_mul(proof.f, c1)
    lhs_1 = mat_vec_mul_convolution(params.A1, proof.response)
    rhs_1 = vec_add(proof.commitment_1, _vec_scalar_mul(target_1, proof.challenge))
    return lhs_1 == rhs_1


def simulate_relaxed_randomness_transcript(
    params: ParCOMParams,
    commitment_value: bytes,
    challenge: int | None = None,
    f: Poly | None = None,
    response_bound: int | None = None,
    tau: int = 60,
) -> RelaxedRandomnessKnowledgeProof:
    if params.ring is None or params.A1 is None:
        raise ValueError("parCOM has no lattice parameters")
    if challenge is None:
        challenge = _sample_challenge(tau)
    if abs(int(challenge)) > int(tau):
        raise ValueError("challenge outside allowed range")
    if f is None:
        f = params.ring.one()
    if response_bound is None:
        response_bound = relaxed_response_bound(params, f, tau)
    c1, _ = _parse_lattice_commitment(params, commitment_value)
    response = small_vec(params.ring, params.rand_cols, response_bound)
    response_norm = _response_norm(response)
    target_1 = _poly_vec_mul(f, c1)
    commitment_1 = vec_sub(mat_vec_mul_convolution(params.A1, response), _vec_scalar_mul(target_1, int(challenge)))
    return RelaxedRandomnessKnowledgeProof(
        tau=int(tau),
        challenge=int(challenge),
        f=f,
        commitment_1=commitment_1,
        response=response,
        mask_bound=_default_mask_bound(params),
        response_bound=response_bound,
        response_norm=response_norm,
    )


def extract_relaxed_randomness_witness(
    params: ParCOMParams,
    first: RelaxedRandomnessKnowledgeProof,
    second: RelaxedRandomnessKnowledgeProof,
) -> RelaxedOpening:
    if params.ring is None:
        raise ValueError("parCOM has no lattice parameters")
    if first.f != second.f:
        raise ValueError("proofs must share the same relaxation polynomial")
    if first.commitment_1 != second.commitment_1:
        raise ValueError("proofs must share the same first-round commitment")
    inv_delta = _challenge_inverse(first.challenge, second.challenge, params.ring)
    return RelaxedOpening(
        randomness=_extract_vec_from_responses(first.response, second.response, inv_delta),
        f=first.f,
    )


def verify_opening(params: ParCOMParams, commitment: Commitment, message: bytes) -> bool:
    if params.ring is not None:
        return verify_relaxed_opening(params, commitment, message)
    expected = hashlib.sha256(params.domain + commitment.opening + message).digest()
    return expected == commitment.value
