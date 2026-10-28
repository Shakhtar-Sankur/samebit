"""samebit spec v0.2 ops in NumPy (int64 throughout, plus an exact float64
matmul). Every op returns integers; nothing here rounds a float."""

import numpy as np

I64 = np.int64


def rshift_round(v, s):
    """floor((v + 2^(s-1)) / 2^s), elementwise; s may be an array."""
    v = np.asarray(v, dtype=I64)
    s = np.asarray(s, dtype=I64)
    half = np.where(s > 0, np.left_shift(I64(1), np.maximum(s - 1, 0)), 0)
    return np.right_shift(v + half, s)


def sat(v, bits):
    lo, hi = -(1 << (bits - 1)), (1 << (bits - 1)) - 1
    return np.clip(v, lo, hi)


def isqrt64(n):
    """floor(sqrt(n)) for 0 <= n < 2^62, digit by digit (31 fixed steps)."""
    n = np.asarray(n, dtype=I64).copy()
    res = np.zeros_like(n)
    bit = I64(1) << I64(60)
    for _ in range(31):
        take = n >= res + bit
        n = np.where(take, n - (res + bit), n)
        res = np.where(take, (res >> 1) + bit, res >> 1)
        bit >>= 2
    return res


def matmul(a, b, exact_float=True):
    """Exact integer matmul. With exact_float, uses float64 BLAS, which is
    exact because every partial sum is an integer below 2^53 (checked)."""
    a = np.asarray(a)
    b = np.asarray(b)
    bound = float(np.abs(a).max(initial=0)) * float(np.abs(b).max(initial=0)) * a.shape[-1]
    assert bound < 2.0**53, f"matmul bound {bound:.3g} >= 2^53"
    if exact_float:
        return np.rint(a.astype(np.float64) @ b.astype(np.float64)).astype(I64)
    return a.astype(I64) @ b.astype(I64)


def requant(x, m, s, bits, bias=None):
    """sat_bits(rshift_round((x + bias) * m, s)); m, s, bias per last-axis channel."""
    x = np.asarray(x, dtype=I64)
    if bias is not None:
        x = x + np.asarray(bias, dtype=I64)
    return sat(rshift_round(x * np.asarray(m, dtype=I64), s), bits)


def rmsnorm(x, g, eps_q24, s2):
    """x: int32 residual in Q12. Returns int16."""
    x = np.asarray(x, dtype=I64)
    d = x.shape[-1]
    assert np.abs(x).max(initial=0) < 2**26 and d <= 2048, "rmsnorm sum of squares bound"
    n = (x * x).sum(-1, keepdims=True) // d + eps_q24
    r = np.maximum(isqrt64(n), 1)
    inv = (I64(1) << I64(40)) // r
    u = rshift_round(x * inv, 26)
    return sat(rshift_round(u * np.asarray(g, dtype=I64), s2), 16)


def rope(x, cos, sin):
    """x: [T, H, D] int16; cos, sin: [T, D/2] Q14."""
    x = np.asarray(x, dtype=I64)
    h = x.shape[-1] // 2
    c, s = cos[:, None, :].astype(I64), sin[:, None, :].astype(I64)
    x1, x2 = x[..., :h], x[..., h:]
    return sat(
        np.concatenate([rshift_round(x1 * c - x2 * s, 14), rshift_round(x1 * s + x2 * c, 14)], -1),
        16,
    )


def softmax(scores, valid, m, s, lut):
    """scores int64 [..., S]; returns probabilities in Q15 (int64)."""
    sc = np.asarray(scores, dtype=I64)
    lo = np.iinfo(I64).min
    mx = np.where(valid, sc, lo).max(-1, keepdims=True)
    d = np.minimum(rshift_round(np.where(valid, mx - sc, 0) * I64(m), s), len(lut) - 1)
    e = np.where(valid, lut[d], 0)
    tot = e.sum(-1, keepdims=True)
    return np.minimum((e * (I64(1) << I64(16)) + tot) // (2 * tot), 32767)


def lut16(x, table):
    return table[np.asarray(x, dtype=I64) + 32768]


def mul_requant(a, b, m, s, bits):
    return sat(rshift_round(np.asarray(a, dtype=I64) * np.asarray(b, dtype=I64) * I64(m), s), bits)


def add_sat(a, b, bits):
    return sat(np.asarray(a, dtype=I64) + np.asarray(b, dtype=I64), bits)


def argmax(x):
    return np.argmax(x, axis=-1)
