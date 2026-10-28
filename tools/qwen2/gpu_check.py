"""Integer Qwen2.5-0.5B on an NVIDIA GPU: compare its bits with the CPU, and
measure the speed against ordinary float16 / float32 inference.

Usage: python gpu_check.py [--tokens 256] [--windows 2]"""

import argparse
import hashlib
import pickle
import statistics
import time

import numpy as np
import pyarrow.parquet as pq
import reference as Q
import torch
from convert import fingerprint
from transformers import AutoModelForCausalLM, AutoTokenizer

from samebit.v2 import ops_cuda, qwen2

FINGERPRINT = "e1994bc2e4f9bff7"  # tools/qwen2/convert.py with calibration.json
# Logit digests of WikiText-2 test windows (256 tokens), x86-64 CPU, 28 Oct 2026
CPU_DIGESTS = ["9709de19df934eb4", "204ff68dead1d2ea", "1572058d9fedc3ec", "4edb3ff435f59651"]


def digest(a):
    return hashlib.sha256(np.ascontiguousarray(a, dtype="<i8").tobytes()).hexdigest()[:16]


def timed(fn, reps=3, sync=lambda: None):
    fn()
    sync()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        out = fn()
        sync()
        ts.append(time.perf_counter() - t0)
    return out, statistics.median(ts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=256)
    ap.add_argument("--windows", type=int, default=2)
    args = ap.parse_args()
    T = args.tokens
    report = [f"GPU: {ops_cuda.device_name()}"]

    with open(Q.MODEL.parent / "qwen2.5-0.5b.samebit.pkl", "rb") as f:
        P = pickle.load(f)
    fp = fingerprint(P)
    report.append(
        f"integer model fingerprint {fp} ({'same' if fp == FINGERPRINT else 'DIFFERENT'})"
    )
    tok = AutoTokenizer.from_pretrained(Q.MODEL)
    text = "".join(
        pq.read_table(Q.MODEL.parent / "data/wt2-test.parquet").column("text").to_pylist()
    )
    ids = tok(text)["input_ids"]
    windows = [ids[j * T : (j + 1) * T] for j in range(args.windows)]
    G = ops_cuda.prepare(P)

    def gpu(x):
        return qwen2.forward(G, x, ops=ops_cuda)

    # 1. Same bits: GPU vs CPU on WikiText-2 windows
    report.append("\n1. Same bits, GPU vs CPU (logits of every position):")
    ok = True
    t_cpu = None
    for j, x in enumerate(windows):
        t0 = time.perf_counter()
        d_cpu = digest(qwen2.forward(P, x))
        t_cpu = t_cpu or time.perf_counter() - t0
        d_gpu = digest(ops_cuda.asnumpy(gpu(x)))
        pub = CPU_DIGESTS[j] if T == 256 and j < len(CPU_DIGESTS) and fp == FINGERPRINT else None
        same = d_gpu == d_cpu and (pub is None or pub == d_gpu)
        ok &= same
        extra = f"  published x86 CPU {pub}" if pub else ""
        report.append(
            f"   window {j}: GPU {d_gpu}  CPU {d_cpu}{extra}  {'IDENTICAL' if same else 'MISMATCH'}"
        )

    # 2. Summation order: split every dot product into 1..64 blocks, atomicAdd in any order
    report.append("\n2. Same bits under any summation order (split-K blocks + atomicAdd):")
    x = windows[0]
    ds = set()
    for k in (1, 4, 16, 64):
        ops_cuda.SPLITK = k
        d = digest(ops_cuda.asnumpy(gpu(x)))
        ds.add(d)
        report.append(f"   split-K {k:>2}: {d}")
    ops_cuda.SPLITK = 4
    ok &= len(ds) == 1

    # 3. Shape invariance: the first half's logits must not change when the input grows,
    # or when other sequences share the batch. Ordinary float inference usually changes.
    h = T // 2
    full = ops_cuda.asnumpy(gpu(x))[:h]
    half = ops_cuda.asnumpy(gpu(x[:h]))
    report.append(f"\n3. The first {h} positions: alone vs inside a longer input or a batch")
    report.append(
        f"   samebit integers, GPU, alone vs inside {T} tokens: max |difference| {np.abs(full - half).max()}"
    )
    ok &= np.array_equal(full, half)

    _, t_gpu = timed(lambda: gpu(x), sync=ops_cuda.sync)
    rows = [("samebit integers, GPU", t_gpu), ("samebit integers, CPU (NumPy)", t_cpu)]
    if torch.cuda.is_available():
        xt = torch.tensor([x], device="cuda")
        batch = torch.tensor([ids[j * T : (j + 1) * T] for j in range(8)], device="cuda")
        for dtype in (torch.float16, torch.float32):
            name = str(dtype).split(".")[1]
            m = AutoModelForCausalLM.from_pretrained(Q.MODEL, torch_dtype=dtype).cuda().eval()
            with torch.no_grad():
                lg, t = timed(lambda m=m: m(xt).logits[0].float(), sync=torch.cuda.synchronize)
                rows.append((f"PyTorch {name}, GPU", t))
                pre = (lg[:h] - m(xt[:, :h]).logits[0].float()).abs().max().item()
                inb = (lg[:h] - m(batch).logits[0, :h].float()).abs().max().item()
            report.append(
                f"   PyTorch {name}, GPU, alone vs inside {T} tokens: max |difference| {pre:.3g}"
            )
            report.append(
                f"   PyTorch {name}, GPU, alone vs in a batch of 8:   max |difference| {inb:.3g}"
            )
            del m
            torch.cuda.empty_cache()
    cfg, w = Q.load()
    with torch.no_grad():
        _, t = timed(lambda: Q.forward(cfg, w, x), reps=1)
    rows.append(("PyTorch float32, CPU", t))

    # 4. Speed, prompt processing (prefill) of T tokens
    report.append(f"\n4. Speed, processing one {T}-token prompt (median of 3 runs):")
    for name, t in rows:
        report.append(f"   {name:32s} {t * 1000:9.1f} ms  {T / t:8.0f} tokens/s")
    report.append(f"\nRESULT: {'PASS, identical bits everywhere' if ok else 'FAIL'}")
    print("\n".join(report))


if __name__ == "__main__":
    main()
