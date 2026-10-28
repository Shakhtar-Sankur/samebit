"""MLX backend: the integer kernels on MLX arrays (Metal on Apple silicon,
CPU elsewhere). MLX matmul only accepts floating types, so the integer
product is a broadcast multiply and an int32 sum, which is exact in any order.
"""

import os

import mlx.core as mx
import numpy as np

name = "mlx"
if os.environ.get("SAMEBIT_MLX_DEVICE") == "cpu":
    mx.set_default_device(mx.cpu)


def _a(x, dtype=mx.int64):
    return mx.array(np.asarray(x)).astype(dtype)


def _np(x, dtype):
    return np.array(x).astype(dtype)


def _rshift_round(v, s):
    s = _a(s)
    one = mx.ones_like(s)
    half = mx.where(s > 0, mx.left_shift(one, mx.maximum(s - 1, 0)), 0)
    return mx.right_shift(v + half, s)


def _sat8(v):
    return _np(mx.clip(v, -128, 127), np.int8)


def _isqrt(n):
    res = mx.zeros_like(n)
    bit = 1 << 30
    for _ in range(16):
        take = n >= res + bit
        n = mx.where(take, n - (res + bit), n)
        res = mx.where(take, mx.right_shift(res, 1) + bit, mx.right_shift(res, 1))
        bit >>= 2
    return res


def matmul(a, b, block=64):
    a, b = _a(a, mx.int32), _a(b, mx.int32)
    cols = [
        (a[:, :, None] * b[None, :, j : j + block]).sum(axis=1) for j in range(0, b.shape[1], block)
    ]
    return _np(mx.concatenate(cols, axis=1), np.int32)


def _requant(x, m, s):
    return mx.clip(_rshift_round(x * _a(m), s), -128, 127)


def requantize(x, m, s):
    return _np(_requant(_a(x), m, s), np.int8)


def rmsnorm(x, g, m, s):
    x = _a(x)
    sq = (x * x).sum(axis=-1, keepdims=True)
    r = mx.maximum(_isqrt(mx.floor_divide(mx.left_shift(sq, 16), x.shape[-1])), 1)
    inv = mx.floor_divide(mx.full(r.shape, 1 << 30, dtype=mx.int64), r)
    v = _rshift_round(x * inv, 14) * _a(g)
    return _np(_requant(v, m, s), np.int8)


def rope(x, cos, sin):
    x = _a(x)
    half = x.shape[-1] // 2
    c, sn = _a(cos)[:, None, :], _a(sin)[:, None, :]
    x1, x2 = x[..., :half], x[..., half:]
    y = mx.concatenate(
        [_rshift_round(x1 * c - x2 * sn, 14), _rshift_round(x1 * sn + x2 * c, 14)], axis=-1
    )
    return _sat8(y)


def softmax(scores, valid, shift, exp_lut):
    scores, ok, lut_a = _a(scores), _a(valid, mx.bool_), _a(exp_lut)
    low = mx.array(np.iinfo(np.int64).min, dtype=mx.int64)
    mxv = mx.where(ok, scores, low).max(axis=-1, keepdims=True)
    d = mx.minimum(_rshift_round(mx.where(ok, mxv - scores, 0), shift), lut_a.shape[0] - 1)
    e = mx.where(ok, mx.take(lut_a, d), 0)
    total = e.sum(axis=-1, keepdims=True)
    p = mx.minimum(mx.floor_divide(e * 512 + total, 2 * total), 255)
    return _np(p, np.int32)


def lut(x, table):
    return _np(mx.take(_a(table), _a(x) + 128), np.int8)


def mul_requant(a, b, m, s):
    return _np(_requant(_a(a) * _a(b), m, s), np.int8)


def add_requant(a, b, ma, mb, s):
    return _sat8(_rshift_round(_a(a) * ma + _a(b) * mb, s))


def argmax(x):
    x = _a(x)
    hit = x == x.max(axis=-1, keepdims=True)
    idx = mx.broadcast_to(mx.arange(x.shape[-1]), x.shape)
    return _np(mx.where(hit, idx, x.shape[-1]).min(axis=-1), np.int64)
