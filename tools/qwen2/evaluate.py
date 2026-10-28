"""Perplexity and top-1 agreement of integer vs float Qwen2.5-0.5B on WikiText-2 test.
Usage: python evaluate.py [n_windows]   (256 tokens per window)"""

import hashlib
import math
import pickle
import sys
import time

import numpy as np
import pyarrow.parquet as pq
import reference as Q
import torch
from transformers import AutoTokenizer

from samebit.v2 import qwen2 as IM

N, T = int(sys.argv[1]) if len(sys.argv) > 1 else 2, 256
tok = AutoTokenizer.from_pretrained(Q.MODEL)
text = "".join(pq.read_table(Q.MODEL.parent / "data/wt2-test.parquet").column("text").to_pylist())
ids = tok(text)["input_ids"]
cfg, w = Q.load()
with open(Q.MODEL.parent / "qwen2.5-0.5b.samebit.pkl", "rb") as f:
    P = pickle.load(f)
nll_f = nll_i = agree = n = 0
for j in range(N):
    win = ids[j * T : (j + 1) * T + 1]
    x, y = win[:-1], torch.tensor(win[1:])
    with torch.no_grad():
        lf = Q.forward(cfg, w, x).double()
    t0 = time.time()
    li = IM.forward(P, x)
    dt = time.time() - t0
    lif = torch.from_numpy(li.astype(np.float64) * P["logit_unit"])
    nll_f += torch.nn.functional.cross_entropy(lf, y, reduction="sum").item()
    nll_i += torch.nn.functional.cross_entropy(lif, y, reduction="sum").item()
    agree += (lf.argmax(-1) == lif.argmax(-1)).sum().item()
    n += len(y)
    print(
        f"window {j}: int forward {dt:.1f}s"
        f"  ppl float {math.exp(nll_f / n):.3f}  int {math.exp(nll_i / n):.3f}"
        f"  top-1 agree {agree / n:.4f}  digest {hashlib.sha256(li.tobytes()).hexdigest()[:16]}",
        flush=True,
    )
