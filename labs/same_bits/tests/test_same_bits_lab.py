import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import engine as e  # noqa: E402


def test_addition_order_changes_bits():
    for dtype in ("float16", "float32"):
        r = e.addition_orders(dtype=dtype)
        assert r["distinct"] > 1
        assert r["rows"][0]["ulps_from_left_to_right"] == 0


def test_tree_and_blocked_sums_are_sums():
    x = np.arange(1, 101, dtype=np.float32)
    assert e.sum_tree(x) == e.sum_sequential(x) == e.sum_blocked(x, 7) == 5050


def test_ulp_distance():
    one = np.array([1.0], np.float32)
    nxt = np.nextafter(one, np.float32(2))
    assert e.ulp_distance(one, one) == 0 and e.ulp_distance(one, nxt) == 1
    assert e.ulp_distance(np.array([-0.0], np.float32), np.array([0.0], np.float32)) == 0


def test_matmul_ordered_matches_float64_closely():
    rng = np.random.default_rng(0)
    a, w = rng.standard_normal((3, 64)), rng.standard_normal((64, 5))
    got = e.matmul_ordered(a, w, 4, np.float32)
    assert np.allclose(got, a @ w, atol=1e-4)


def test_batch_changes_bits_unless_invariant():
    assert e.batch_contrast(1)["outputs_differing"] == 0
    assert e.batch_contrast(32)["outputs_differing"] > 0
    assert e.batch_contrast(32, policy="invariant")["outputs_differing"] == 0


def test_serving_divergence_and_both_fixes():
    assert e.serving_contrast(16)["changed"] > 0
    assert e.serving_contrast(16, policy="invariant")["changed"] == 0
    assert e.serving_contrast(16, mode="integer")["changed"] == 0


def test_integer_path_is_order_free():
    m = e.TinyLM()
    h = np.tanh(np.random.default_rng(3).standard_normal((8, m.width)))
    outs = {e.digest(e.logits(m, h[i:i + 1], "integer", "throughput")[0]) for i in range(1)}
    batched = e.logits(m, h, "integer", "throughput")[0]
    assert outs == {e.digest(batched)}


def test_integer_agreement_is_reported():
    r = e.integer_agreement()
    assert 0.5 < r["token_agreement"] <= 1.0
