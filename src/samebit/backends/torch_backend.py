"""PyTorch backend: the same integer kernels on torch tensors (CPU)."""

import numpy as np
import torch

name = "torch"
device = torch.device("cpu")


def _t(x, dtype=torch.int64):
    return torch.as_tensor(np.asarray(x), device=device).to(dtype)


def _np(x, dtype):
    return x.cpu().numpy().astype(dtype)


def _rshift_round(v, s):
    s = _t(s)
    half = torch.where(s > 0, torch.ones_like(s) << torch.clamp(s - 1, min=0), 0)
    return (v + half) >> s


def _sat8(v):
    return _np(torch.clamp(v, -128, 127), np.int8)


def _isqrt(n):
    res = torch.zeros_like(n)
    bit = 1 << 30
    for _ in range(16):
        take = n >= res + bit
        n = torch.where(take, n - (res + bit), n)
        res = torch.where(take, (res >> 1) + bit, res >> 1)
        bit >>= 2
    return res


def matmul(a, b):
    return _np(_t(a, torch.int32) @ _t(b, torch.int32), np.int32)


def _requant(x, m, s):
    return torch.clamp(_rshift_round(x * _t(m), s), -128, 127)


def requantize(x, m, s):
    return _np(_requant(_t(x), m, s), np.int8)


def rmsnorm(x, g, m, s):
    x = _t(x)
    sq = (x * x).sum(dim=-1, keepdim=True)
    r = torch.clamp(_isqrt(torch.div(sq << 16, x.shape[-1], rounding_mode="floor")), min=1)
    inv = torch.div(torch.full_like(r, 1 << 30), r, rounding_mode="floor")
    v = _rshift_round(x * inv, 14) * _t(g)
    return _np(_requant(v, m, s), np.int8)


def rope(x, cos, sin):
    x = _t(x)
    half = x.shape[-1] // 2
    c, sn = _t(cos)[:, None, :], _t(sin)[:, None, :]
    x1, x2 = x[..., :half], x[..., half:]
    y = torch.cat(
        [_rshift_round(x1 * c - x2 * sn, 14), _rshift_round(x1 * sn + x2 * c, 14)], dim=-1
    )
    return _sat8(y)


def softmax(scores, valid, shift, exp_lut):
    scores, ok, lut_t = _t(scores), _t(valid, torch.bool), _t(exp_lut)
    mx = torch.where(ok, scores, torch.iinfo(torch.int64).min).amax(dim=-1, keepdim=True)
    d = torch.clamp(_rshift_round(torch.where(ok, mx - scores, 0), shift), max=len(lut_t) - 1)
    e = torch.where(ok, lut_t[d], 0)
    total = e.sum(dim=-1, keepdim=True)
    p = torch.clamp(torch.div(e * 512 + total, 2 * total, rounding_mode="floor"), max=255)
    return _np(p, np.int32)


def lut(x, table):
    return _np(_t(table)[_t(x) + 128], np.int8)


def mul_requant(a, b, m, s):
    return _np(_requant(_t(a) * _t(b), m, s), np.int8)


def add_requant(a, b, ma, mb, s):
    return _sat8(_rshift_round(_t(a) * ma + _t(b) * mb, s))


def argmax(x):
    # torch.argmax does not promise the first maximum, so take it explicitly.
    x = _t(x)
    hit = x == x.amax(dim=-1, keepdim=True)
    idx = torch.arange(x.shape[-1]).expand_as(x)
    return _np(torch.where(hit, idx, x.shape[-1]).amin(dim=-1), np.int64)
