"""Reference implementation of every samebit op, in plain Python integers.

This module is the executable form of SPEC.md. It uses no numpy arithmetic:
inputs are converted to nested lists of Python ints, every operation is done
with Python's arbitrary-precision integers, and each result is checked against
the int32/int64 bounds the spec promises. Backends are correct when they return
exactly the same integers.
"""

import numpy as np

INT8_MIN, INT8_MAX = -128, 127
INT32_MAX = 2**31 - 1
INT64_MAX = 2**63 - 1


def _check(value, bound, what):
    if not -bound - 1 <= value <= bound:
        raise OverflowError(f"{what} = {value} is outside the spec bound")
    return value


def rshift_round(v, s):
    """floor((v + 2**(s-1)) / 2**s): round half up, for any sign of v."""
    if s == 0:
        return v
    return (v + (1 << (s - 1))) >> s  # Python >> is floor division


def sat(v, lo=INT8_MIN, hi=INT8_MAX):
    return lo if v < lo else hi if v > hi else v


def isqrt(n):
    """floor(sqrt(n)) for 0 <= n < 2**32, digit by digit (16 fixed steps)."""
    res, bit = 0, 1 << 30
    for _ in range(16):
        if n >= res + bit:
            n -= res + bit
            res = (res >> 1) + bit
        else:
            res >>= 1
        bit >>= 2
    return res


def _requant_scalar(x, m, s):
    _check(x, INT32_MAX, "requantize input")
    return sat(rshift_round(x * m, s))


def _per_channel(p, n):
    p = np.asarray(p).reshape(-1).tolist()
    return p * n if len(p) == 1 else p


def matmul(a, b):
    a, b = np.asarray(a).tolist(), np.asarray(b).tolist()
    k, n = len(b), len(b[0])
    out = []
    for row in a:
        assert len(row) == k
        r = []
        for j in range(n):
            acc = 0
            for t in range(k):
                acc += row[t] * b[t][j]
            r.append(_check(acc, INT32_MAX, "matmul accumulator"))
        out.append(r)
    return np.array(out, dtype=np.int32).reshape(len(a), n)


def requantize(x, m, s):
    x = np.asarray(x)
    flat, n = x.reshape(-1, x.shape[-1]).tolist(), x.shape[-1]
    ms, ss = _per_channel(m, n), _per_channel(s, n)
    out = [[_requant_scalar(v, ms[j], ss[j]) for j, v in enumerate(row)] for row in flat]
    return np.array(out, dtype=np.int8).reshape(x.shape)


def rmsnorm(x, g, m, s):
    x = np.asarray(x)
    d = x.shape[-1]
    gs, ms, ss = _per_channel(g, d), _per_channel(m, d), _per_channel(s, d)
    out = []
    for row in x.reshape(-1, d).tolist():
        sq = sum(v * v for v in row)
        n = _check((sq << 16) // d, 2**32 - 1, "rmsnorm mean square")
        r = max(isqrt(n), 1)
        inv = (1 << 30) // r
        out.append(
            [
                _requant_scalar(rshift_round(v * inv, 14) * gs[j], ms[j], ss[j])
                for j, v in enumerate(row)
            ]
        )
    return np.array(out, dtype=np.int8).reshape(x.shape)


def rope(x, cos, sin):
    """x: [T, H, D] int8; cos, sin: [T, D//2] Q14. Half-split rotation."""
    x = np.asarray(x)
    t, h, d = x.shape
    half = d // 2
    xl, cl, sl = x.tolist(), np.asarray(cos).tolist(), np.asarray(sin).tolist()
    out = []
    for i in range(t):
        rows = []
        for head in xl[i]:
            y = [0] * d
            for j in range(half):
                x1, x2, c, sn = head[j], head[j + half], cl[i][j], sl[i][j]
                y[j] = sat(rshift_round(x1 * c - x2 * sn, 14))
                y[j + half] = sat(rshift_round(x1 * sn + x2 * c, 14))
            rows.append(y)
        out.append(rows)
    return np.array(out, dtype=np.int8).reshape(t, h, d)


def softmax(scores, valid, shift, exp_lut):
    """scores: [R, S] int32; valid: [R, S] bool. Returns probabilities in
    [0, 255] with scale 1/256 (int32)."""
    lut = np.asarray(exp_lut).tolist()
    last = len(lut) - 1
    out = []
    for row, ok in zip(np.asarray(scores).tolist(), np.asarray(valid).tolist()):
        mx = max(v for v, k in zip(row, ok) if k)
        e = [lut[min(rshift_round(mx - v, shift), last)] if k else 0 for v, k in zip(row, ok)]
        total = sum(e)
        out.append([min((v * 512 + total) // (2 * total), 255) for v in e])
    return np.array(out, dtype=np.int32).reshape(np.asarray(scores).shape)


def lut(x, table):
    table = np.asarray(table).tolist()
    x = np.asarray(x)
    return np.array([table[v + 128] for v in x.reshape(-1).tolist()], dtype=np.int8).reshape(
        x.shape
    )


def mul_requant(a, b, m, s):
    a, b = np.asarray(a), np.asarray(b)
    prod = [p * q for p, q in zip(a.reshape(-1).tolist(), b.reshape(-1).tolist())]
    return requantize(np.array(prod, dtype=np.int64).reshape(a.shape), m, s)


def add_requant(a, b, ma, mb, s):
    a, b = np.asarray(a), np.asarray(b)
    out = [
        sat(rshift_round(_check(p * ma + q * mb, INT64_MAX, "add accumulator"), s))
        for p, q in zip(a.reshape(-1).tolist(), b.reshape(-1).tolist())
    ]
    return np.array(out, dtype=np.int8).reshape(a.shape)


def argmax(x):
    """Index of the largest value in each row; ties go to the lowest index."""
    out = []
    for row in np.asarray(x).tolist():
        best = 0
        for j, v in enumerate(row):
            if v > row[best]:
                best = j
        out.append(best)
    return np.array(out, dtype=np.int64)
