from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Params:

    n: int
    q: int


    p: int


    k: int


    B: int


    ell: int
    alpha: int
    alpha1: int
    d: int
    beta: int
    kappa1: int
    kappa2: int


    tau: int = 60
    dm_K: int = 4
    dm_L: int = 4


    s: int = 2
    m_alpha: int = 2


def table_ii_params() -> Params:

    n = 128
    ell = 16
    q = 4294968833
    p = 70368744177679
    k = math.ceil(math.log(q, 3))
    s = math.ceil(math.log2(q)) - 1

    return Params(
        n=n,
        q=q,
        p=p,
        k=k,
        B=8,
        ell=ell,
        alpha=4,
        alpha1=6,
        d=2,
        beta=95255,
        kappa1=45,
        kappa2=52,
        tau=60,
        dm_K=4,
        dm_L=4,
        s=s,
        m_alpha=2,
    )


def legacy_timing_params() -> Params:

    n = 128
    ell = 16
    q = 6073984769
    p = q
    k = math.ceil(math.log(q, 3))

    return Params(
        n=n,
        q=q,
        p=p,
        k=k,
        B=8,
        ell=ell,
        alpha=4,
        alpha1=6,
        d=2,
        beta=95255,
        kappa1=45,
        kappa2=52,
        tau=60,
        dm_K=4,
        dm_L=4,
        s=2,
        m_alpha=2,
    )
