"""Every backend must return exactly the reference integers, op by op."""

import numpy as np
import pytest

from samebit import reference, tables

rng = np.random.default_rng(0)


def i8(*shape):
    x = rng.integers(-128, 128, size=shape, dtype=np.int64)
    x.flat[:: max(x.size // 7, 1)] = -128  # always include the int8 minimum
    return x.astype(np.int8)


def same(a, b):
    np.testing.assert_array_equal(np.asarray(a, dtype=np.int64), np.asarray(b, dtype=np.int64))


def test_rshift_round_examples():
    # round half up, also for negative values
    assert [reference.rshift_round(v, 1) for v in (-3, -2, -1, 0, 1, 2, 3)] == [
        -1,
        -1,
        0,
        0,
        1,
        1,
        2,
    ]
    assert reference.rshift_round(-5, 2) == -1  # -1.25 -> -1
    assert reference.rshift_round(-6, 2) == -1  # -1.5 -> -1 (half up)
    assert reference.rshift_round(-7, 2) == -2


def test_isqrt_exact():
    import math

    for n in (
        list(range(5000))
        + [2**32 - 1, 2**30, 2**31 + 12345]
        + rng.integers(0, 2**32, 2000).tolist()
    ):
        assert reference.isqrt(n) == math.isqrt(n)


@pytest.mark.parametrize("m,k,n", [(1, 1, 1), (3, 17, 5), (8, 64, 33), (16, 128, 256)])
def test_matmul(be, ref, m, k, n):
    a, b = i8(m, k), i8(k, n)
    same(be.matmul(a, b), ref.matmul(a, b))


def test_matmul_worst_case_bound(be):
    # K = 131071 is the largest depth at which int8 x int8 cannot overflow
    # int32 in any summation order (see docs/BOUNDS.md).
    k = 131071
    a = np.full((1, k), -128, np.int8)
    b = np.full((k, 2), -128, np.int8)
    b[:, 1] = 127
    same(be.matmul(a, b), [[16384 * k, -16256 * k]])


def test_requantize(be, ref):
    x = rng.integers(-(2**31), 2**31, size=(6, 40), dtype=np.int64)
    x[0, :4] = [-(2**31), 2**31 - 1, -1, 1]
    same(be.requantize(x, 1 << 30, 37), ref.requantize(x, 1 << 30, 37))
    ms = rng.integers(1 << 30, 1 << 31, size=40)
    ss = rng.integers(0, 50, size=40)
    same(be.requantize(x, ms, ss), ref.requantize(x, ms, ss))


def test_rmsnorm(be, ref):
    x = i8(9, 64)
    x[0] = 0  # all-zero row takes the r = max(r, 1) path
    x[1] = -128
    g = i8(64)
    same(be.rmsnorm(x, g, 1 << 30, 39), ref.rmsnorm(x, g, 1 << 30, 39))


def test_rope(be, ref):
    cos, sin = tables.rope_tables(5, 16)
    x = i8(5, 3, 16)
    same(be.rope(x, cos, sin), ref.rope(x, cos, sin))


def test_softmax(be, ref):
    lut = tables.exp_lut()
    scores = rng.integers(-60000, 60000, size=(7, 23), dtype=np.int64)
    scores[0] = 5  # all ties
    valid = rng.random((7, 23)) < 0.7
    valid[:, 0] = True
    for shift in (0, 3, 9):
        same(be.softmax(scores, valid, shift, lut), ref.softmax(scores, valid, shift, lut))


def test_lut(be, ref):
    table = tables.silu_lut()
    x = np.arange(-128, 128, dtype=np.int8)
    same(be.lut(x, table), ref.lut(x, table))


def test_mul_and_add_requant(be, ref):
    a, b = i8(4, 50), i8(4, 50)
    same(be.mul_requant(a, b, 1 << 30, 35), ref.mul_requant(a, b, 1 << 30, 35))
    same(be.add_requant(a, b, 12345, 23456, 15), ref.add_requant(a, b, 12345, 23456, 15))


def test_argmax_ties_go_to_lowest_index(be, ref):
    x = np.array([[3, 9, 9, 1], [-5, -5, -5, -5], [0, 1, 2, 2]], dtype=np.int64)
    same(be.argmax(x), [1, 0, 2])
    same(ref.argmax(x), [1, 0, 2])
