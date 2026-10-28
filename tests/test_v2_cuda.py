"""Spec v0.2 CUDA ops must equal the NumPy ops exactly. Runs on a GPU with CuPy,
or anywhere with PYTHONPATH=tests/emulated_cupy."""

import numpy as np
import pytest

from samebit.v2 import ops

pytest.importorskip("cupy")
cu = pytest.importorskip("samebit.v2.ops_cuda")
rng = np.random.default_rng(7)


def i16(*shape):
    return rng.integers(-32768, 32768, size=shape, dtype=np.int64)


def same(a, b):
    return np.array_equal(np.asarray(a), cu.asnumpy(b))


@pytest.mark.parametrize("m,k,n", [(1, 1, 1), (9, 896, 70), (17, 130, 300), (3, 4864, 5)])
def test_cuda_v2_matmul(m, k, n):
    a = i16(m, k)
    w = rng.integers(-127, 128, size=(k, n)).astype(np.int8)
    assert same(a @ w.astype(np.int64), cu.matmul(a, w))
    b = i16(k, n)
    assert same(a @ b, cu.matmul(a, b))


def test_cuda_v2_elementwise():
    x = rng.integers(-(2**24), 2**24, size=(9, 64))
    m, b = rng.integers(-(2**23), 2**23, size=64), rng.integers(-1000, 1000, size=64)
    assert same(ops.requant(x, m, 30, 16, b), cu.requant(x, m, 30, 16, b))
    assert same(ops.requant(x, np.int64(-77), 3, 63), cu.requant(x, np.int64(-77), 3, 63))
    g = rng.integers(-(2**30), 2**30, size=64)
    assert same(ops.rmsnorm(x, g, 17, 34), cu.rmsnorm(x, g, 17, 34))
    assert same(ops.rmsnorm(x * 0, g, 17, 34), cu.rmsnorm(x * 0, g, 17, 34))
    q = i16(9, 4, 64)
    ang = np.arange(9)[:, None] * 0.37 ** np.arange(32)[None, :]
    cos = np.rint(np.cos(ang) * 16384).astype(np.int64)
    sin = np.rint(np.sin(ang) * 16384).astype(np.int64)
    assert same(ops.rope(q, cos, sin), cu.rope(q, cos, sin))
    lut = np.rint((1 << 24) * np.exp(-np.arange(4096) / 128)).astype(np.int64)
    sc = rng.integers(-(2**36), 2**36, size=(16, 16))
    valid = np.tril(np.ones((16, 16), dtype=bool))
    assert same(ops.softmax(sc, valid, 3, 30, lut), cu.softmax(sc, valid, 3, 30, lut))
    table = rng.integers(-32768, 32768, size=65536)
    a, c = i16(50), i16(50)
    assert same(ops.lut16(a, table), cu.lut16(a, table))
    assert same(ops.mul_requant(a, c, 12345, 40, 16), cu.mul_requant(a, c, 12345, 40, 16))
    assert same(ops.add_sat(a << 16, c << 16, 32), cu.add_sat(a << 16, c << 16, 32))
    ties = np.array([[3, 9, 9, 1], [-5, -5, -5, -5]])
    assert same(ops.argmax(ties), cu.argmax(ties))
