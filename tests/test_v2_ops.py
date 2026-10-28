"""Spec v0.2 ops: the NumPy and PyTorch implementations must agree exactly,
and the float64-BLAS matmul must equal the int64 matmul."""

import math

import numpy as np
import pytest

from samebit.v2 import ops

torch_ops = pytest.importorskip("samebit.v2.ops_torch")
rng = np.random.default_rng(20261028)


def i16(*shape):
    return rng.integers(-32768, 32768, size=shape, dtype=np.int64)


def test_isqrt64_exact():
    n = np.concatenate([rng.integers(0, 2**62, 5000), [0, 1, 2, 3, 4, 2**62 - 1]])
    r = ops.isqrt64(n)
    assert all(int(a) == math.isqrt(int(b)) for a, b in zip(r, n))
    assert np.array_equal(r, torch_ops.isqrt64(n))


def test_rshift_round_negative_halves():
    v = np.array([-5, -4, -3, -2, -1, 0, 1, 2, 3, 5], dtype=np.int64)
    assert ops.rshift_round(v, 1).tolist() == [-2, -2, -1, -1, 0, 0, 1, 1, 2, 3]
    assert np.array_equal(ops.rshift_round(v, 1), torch_ops.rshift_round(v, 1))


def test_matmul_blas_equals_int64():
    a, b = i16(33, 896), rng.integers(-127, 128, size=(896, 70))
    ref = a @ b
    assert np.array_equal(ops.matmul(a, b), ref)
    assert np.array_equal(ops.matmul(a, b, exact_float=False), ref)
    assert np.array_equal(torch_ops.matmul(a, b), ref)


def test_matmul_rejects_inexact_bound():
    with pytest.raises(AssertionError):
        ops.matmul(np.full((1, 8), 2**25), np.full((8, 1), 2**25))  # bound 2^53


def test_requant_rmsnorm_rope_agree():
    x = rng.integers(-(2**24), 2**24, size=(9, 64))
    m, s, b = rng.integers(-(2**23), 2**23, size=64), 30, rng.integers(-1000, 1000, size=64)
    assert np.array_equal(ops.requant(x, m, s, 16, b), torch_ops.requant(x, m, s, 16, b))
    g = rng.integers(-(2**30), 2**30, size=64)
    assert np.array_equal(ops.rmsnorm(x, g, 17, 34), torch_ops.rmsnorm(x, g, 17, 34))
    q = i16(9, 4, 64)
    ang = np.arange(9)[:, None] * 0.37 ** np.arange(32)[None, :]
    cos, sin = (
        np.rint(np.cos(ang) * 16384).astype(np.int64),
        np.rint(np.sin(ang) * 16384).astype(np.int64),
    )
    assert np.array_equal(ops.rope(q, cos, sin), torch_ops.rope(q, cos, sin))


def test_softmax_lut_agree():
    lut = np.rint((1 << 24) * np.exp(-np.arange(4096) / 128)).astype(np.int64)
    sc = rng.integers(-(2**36), 2**36, size=(16, 16))
    valid = np.tril(np.ones((16, 16), dtype=bool))
    p = ops.softmax(sc, valid, 3, 30, lut)
    assert np.array_equal(p, torch_ops.softmax(sc, valid, 3, 30, lut))
    assert (p[~valid] == 0).all() and (p.sum(-1) <= 32768 + 16).all()
    table = rng.integers(-32768, 32768, size=65536)
    a, c = i16(50), i16(50)
    assert np.array_equal(ops.lut16(a, table), torch_ops.lut16(a, table))
    assert np.array_equal(
        ops.mul_requant(a, c, 12345, 40, 16), torch_ops.mul_requant(a, c, 12345, 40, 16)
    )
    assert np.array_equal(
        ops.add_sat(a << 16, c << 16, 32), torch_ops.add_sat(a << 16, c << 16, 32)
    )
