import collections

import pyarrow.parquet as pq
import reference as Q
import torch
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained(Q.MODEL)
txt = "".join(pq.read_table(Q.MODEL.parent / "data/wt2-train.parquet").column("text").to_pylist())
ids = tok(txt[:200000])["input_ids"]
cfg, w = Q.load()
st = collections.defaultdict(lambda: [0.0, None, []])


def hook(n, t):
    t = t.detach()
    s = st[n]
    a = t.abs()
    s[0] = max(s[0], a.max().item())
    ch = a.reshape(-1, a.shape[-1]).amax(0)
    s[1] = ch if s[1] is None else torch.maximum(s[1], ch)
    s[2].append(torch.quantile(a.reshape(-1)[:200000].float(), 0.999).item())


with torch.no_grad():
    for j in range(8):
        Q.forward(cfg, w, ids[j * 256 : (j + 1) * 256], hook)
for n in [
    "L0.resid_in",
    "L1.resid_in",
    "L3.resid_in",
    "L12.resid_in",
    "L23.resid_mid",
    "final_resid",
    "L0.qkv_in",
    "L12.qkv_in",
    "L12.mlp_in",
    "L1.down_in",
    "L12.down_in",
    "L23.down_in",
    "L12.q",
    "L12.k",
    "L12.scores",
    "head_in",
]:
    mx, ch, q = st[n]
    top = torch.topk(ch, 3)
    print(
        f"{n:14s} max={mx:9.2f} p99.9={max(q):8.2f}"
        f" top-ch={[(int(i), round(float(v), 1)) for v, i in zip(top.values, top.indices)]}"
        f" median-ch-max={ch.median().item():.2f}"
    )
