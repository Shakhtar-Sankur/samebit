"""Run the integer Qwen2.5-0.5B three ways and compare every residual stream
and the logits bit for bit:
  numpy-blas   float64 BLAS matmul (exact: all sums are integers below 2^53)
  numpy-int64  plain int64 matmul
  torch        separate PyTorch int64 implementation of every op
Usage: python determinism.py [n_tokens]"""

import hashlib
import pickle
import sys
import time
import types

import reference as Q
from transformers import AutoTokenizer

from samebit.v2 import ops, ops_torch, qwen2

PROMPT = "The same model, the same prompt, the same bits on every chip. " * 8

ops_int = types.SimpleNamespace(**{k: getattr(ops, k) for k in dir(ops) if not k.startswith("_")})
ops_int.matmul = lambda a, b: ops.matmul(a, b, exact_float=False)


def digest(arrays):
    h = hashlib.sha256()
    for a in arrays:
        h.update(a.astype("<i8").tobytes())
    return h.hexdigest()[:16]


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 32
    ids = AutoTokenizer.from_pretrained(Q.MODEL)(PROMPT)["input_ids"][:n]
    with open(Q.MODEL.parent / "qwen2.5-0.5b.samebit.pkl", "rb") as f:
        P = pickle.load(f)
    out = {}
    for name, o in [("numpy-blas", ops), ("numpy-int64", ops_int), ("torch", ops_torch)]:
        tr = []
        t0 = time.time()
        logits = qwen2.forward(P, ids, ops=o, trace=tr)
        out[name] = (digest(tr + [logits]), logits.argmax(-1))
        print(f"{name:12s} {out[name][0]}  ({time.time() - t0:.1f}s)", flush=True)
    ok = len({d for d, _ in out.values()}) == 1
    print(f"{len(ids)} tokens, 24 residual streams + logits: {'IDENTICAL' if ok else 'MISMATCH'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
