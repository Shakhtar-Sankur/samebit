"""NumPy backend: vectorised integer kernels, int32 matmul accumulation."""

import numpy as np

name = "numpy"


def _rshift_round(v, s):
    s = np.asarray(s, dtype=np.int64)
    half = np.where(s > 0, np.left_shift(np.int64(1), np.maximum(s - 1, 0)), 0)
    return np.right_shift(v + half, s)  # arithmetic shift on int64 is floor


def _sat8(v):
    return np.clip(v, -128, 127).astype(np.int8)


def _isqrt(n):
    n = n.astype(np.int64)
    res = np.zeros_like(n)
    bit = np.int64(1 << 30)
    for _ in range(16):
        take = n >= res + bit
        n = np.where(take, n - (res + bit), n)
        res = np.where(take, (res >> 1) + bit, res >> 1)
        bit >>= 2
    return res


def matmul(a, b):
    return np.matmul(np.asarray(a, dtype=np.int32), np.asarray(b, dtype=np.int32))


def requantize(x, m, s):
    x = np.asarray(x, dtype=np.int64)
    return _sat8(_rshift_round(x * np.asarray(m, dtype=np.int64), s))


def rmsnorm(x, g, m, s):
    x = np.asarray(x, dtype=np.int64)
    sq = (x * x).sum(axis=-1, keepdims=True)
    r = np.maximum(_isqrt((sq << 16) // x.shape[-1]), 1)
    inv = (1 << 30) // r
    v = _rshift_round(x * inv, 14) * np.asarray(g, dtype=np.int64)
    return requantize(v, m, s)


def rope(x, cos, sin):
    x = np.asarray(x, dtype=np.int64)
    half = x.shape[-1] // 2
    c = np.asarray(cos, dtype=np.int64)[:, None, :]
    sn = np.asarray(sin, dtype=np.int64)[:, None, :]
    x1, x2 = x[..., :half], x[..., half:]
    y1 = _rshift_round(x1 * c - x2 * sn, 14)
    y2 = _rshift_round(x1 * sn + x2 * c, 14)
    return _sat8(np.concatenate([y1, y2], axis=-1))


def softmax(scores, valid, shift, exp_lut):
    scores = np.asarray(scores, dtype=np.int64)
    lut = np.asarray(exp_lut, dtype=np.int64)
    mx = np.where(valid, scores, np.iinfo(np.int64).min).max(axis=-1, keepdims=True)
    d = np.minimum(_rshift_round(np.where(valid, mx - scores, 0), shift), len(lut) - 1)
    e = np.where(valid, lut[d], 0)
    total = e.sum(axis=-1, keepdims=True)
    return np.minimum((e * 512 + total) // (2 * total), 255).astype(np.int32)


def lut(x, table):
    return np.asarray(table, dtype=np.int8)[np.asarray(x, dtype=np.int64) + 128]


def mul_requant(a, b, m, s):
    return requantize(np.asarray(a, dtype=np.int64) * np.asarray(b, dtype=np.int64), m, s)


def add_requant(a, b, ma, mb, s):
    v = np.asarray(a, dtype=np.int64) * ma + np.asarray(b, dtype=np.int64) * mb
    return _sat8(_rshift_round(v, s))


def argmax(x):
    return np.argmax(np.asarray(x), axis=-1).astype(np.int64)  # first maximum wins
