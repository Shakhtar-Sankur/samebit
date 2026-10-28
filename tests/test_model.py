"""End-to-end conformance: identical bits for every tensor on every backend."""

import json
from pathlib import Path

import numpy as np

import samebit
from samebit import model
from samebit.backends import load

GOLDEN = json.loads((Path(samebit.__file__).parent / "golden.json").read_text())
MODEL = model.build()


def run(be):
    return model.generate(be, MODEL, list(GOLDEN["prompt"].encode()), GOLDEN["steps"])


def test_matches_golden(be):
    tokens, trace = run(be)
    assert tokens == GOLDEN["tokens"]
    assert len(trace.digests) == GOLDEN["tensors"]
    assert trace.digest() == GOLDEN["digest"]


def test_reference_matches_golden(ref):
    tokens, trace = run(ref)
    assert tokens == GOLDEN["tokens"]
    assert trace.digest() == GOLDEN["digest"]


def test_prefill_equals_incremental(be):
    # Batch invariance: processing the prompt at once or one token at a time
    # gives the same logits bit for bit.
    prompt = list(b"batch invariance")
    whole = model.forward(be, MODEL, prompt, 0, {}, model.Trace())[-1]
    cache = {}
    for p, tok in enumerate(prompt):
        last = model.forward(be, MODEL, [tok], p, cache, model.Trace())[-1]
    np.testing.assert_array_equal(whole, last)


def test_c_accumulation_order_does_not_matter():
    c = load("c")
    digests = set()
    try:
        for order in (0, 1, 2):
            c.ORDER = order
            digests.add(run(c)[1].digest())
    finally:
        c.ORDER = 0
    assert digests == {GOLDEN["digest"]}
