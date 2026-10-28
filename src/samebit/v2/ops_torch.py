"""samebit spec v0.2 ops in PyTorch int64, written separately from ops.py.
Takes and returns NumPy arrays so the same runner can drive either set."""

import numpy as np
import torch

I64 = torch.int64


def _t(v):
    return torch.as_tensor(np.asarray(v)).to(I64)


def _rshift_round(v, s):
    s = _t(s)
    half = torch.where(s > 0, torch.ones_like(s) << (s - 1).clamp_min(0), torch.zeros_like(s))
    return (v + half) >> s


def _sat(v, bits):
    return v.clamp(-(1 << (bits - 1)), (1 << (bits - 1)) - 1)


def _isqrt64(n):
    n = n.clone()
    res = torch.zeros_like(n)
    bit = 1 << 60
    for _ in range(31):
        take = n >= res + bit
        n = torch.where(take, n - (res + bit), n)
        res = torch.where(take, (res >> 1) + bit, res >> 1)
        bit >>= 2
    return res


def rshift_round(v, s):
    return _rshift_round(_t(v), s).numpy()


def sat(v, bits):
    return _sat(_t(v), bits).numpy()


def isqrt64(n):
    return _isqrt64(_t(n)).numpy()


def matmul(a, b):
    """int64 matmul in torch; exact because every sum stays below 2^63."""
    return (_t(a) @ _t(b)).numpy()


def requant(x, m, s, bits, bias=None):
    x = _t(x)
    if bias is not None:
        x = x + _t(bias)
    return _sat(_rshift_round(x * _t(m), s), bits).numpy()


def rmsnorm(x, g, eps_q24, s2):
    x = _t(x)
    n = torch.div((x * x).sum(-1, keepdim=True), x.shape[-1], rounding_mode="floor") + eps_q24
    r = _isqrt64(n).clamp_min(1)
    inv = torch.div(torch.full_like(r, 1 << 40), r, rounding_mode="floor")
    u = _rshift_round(x * inv, 26)
    return _sat(_rshift_round(u * _t(g), s2), 16).numpy()


def rope(x, cos, sin):
    x = _t(x)
    h = x.shape[-1] // 2
    c, s = _t(cos)[:, None, :], _t(sin)[:, None, :]
    x1, x2 = x[..., :h], x[..., h:]
    out = torch.cat([_rshift_round(x1 * c - x2 * s, 14), _rshift_round(x1 * s + x2 * c, 14)], -1)
    return _sat(out, 16).numpy()


def softmax(scores, valid, m, s, lut):
    sc, ok, lut = _t(scores), torch.as_tensor(np.asarray(valid)), _t(lut)
    mx = torch.where(ok, sc, torch.iinfo(I64).min).amax(-1, keepdim=True)
    d = _rshift_round(torch.where(ok, mx - sc, 0) * int(m), s).clamp_max(len(lut) - 1)
    e = torch.where(ok, lut[d], 0)
    tot = e.sum(-1, keepdim=True)
    return torch.div(e * (1 << 16) + tot, 2 * tot, rounding_mode="floor").clamp_max(32767).numpy()


def lut16(x, table):
    return _t(table)[_t(x) + 32768].numpy()


def mul_requant(a, b, m, s, bits):
    return _sat(_rshift_round(_t(a) * _t(b) * int(m), s), bits).numpy()


def add_sat(a, b, bits):
    return _sat(_t(a) + _t(b), bits).numpy()


def argmax(x):
    return torch.argmax(_t(x), -1).numpy()
