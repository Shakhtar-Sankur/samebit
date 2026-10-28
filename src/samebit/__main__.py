"""Conformance check: python -m samebit [backend ...]

Runs the reference model on each backend and compares the digest of every
intermediate tensor with the published golden digest.
"""

import json
import sys
import time
from pathlib import Path

from . import model
from .backends import available, load

GOLDEN = json.loads((Path(__file__).parent / "golden.json").read_text())


def run(name):
    be = load(name)
    mdl = model.build()
    start = time.perf_counter()
    tokens, trace = model.generate(be, mdl, list(GOLDEN["prompt"].encode()), GOLDEN["steps"])
    return tokens, trace.digest(), time.perf_counter() - start


def main(argv):
    names = argv or available()
    ok = True
    print(f"golden {GOLDEN['digest'][:16]}  tokens {GOLDEN['tokens']}")
    for name in names:
        dev = getattr(load(name), "device_name", None)
        if dev:
            print(f"{name}: {dev()}")
        tokens, digest, secs = run(name)
        same = digest == GOLDEN["digest"] and tokens == GOLDEN["tokens"]
        ok &= same
        print(f"{name:10s} {digest[:16]}  {'PASS' if same else 'FAIL'}  ({secs:.2f}s)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
