"""An integer-only, Llama-shaped decoder that runs on any samebit backend.

All arithmetic goes through the backend under test. The runner itself only
moves data (gather, slice, concatenate), which cannot change any value.
Weights come from a SplitMix64 generator written in plain Python, so every
machine builds the same model without depending on a library's RNG.
"""

import hashlib
from dataclasses import dataclass

import numpy as np

from . import tables


@dataclass(frozen=True)
class Config:
    vocab: int = 256  # byte-level tokens
    dim: int = 64
    heads: int = 4
    ffn: int = 128
    layers: int = 2
    max_pos: int = 64
    seed: int = 2026

    @property
    def head_dim(self):
        return self.dim // self.heads


class SplitMix64:
    MASK = (1 << 64) - 1

    def __init__(self, seed):
        self.state = seed & self.MASK

    def next(self):
        self.state = (self.state + 0x9E3779B97F4A7C15) & self.MASK
        z = self.state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & self.MASK
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & self.MASK
        return z ^ (z >> 31)

    def ints(self, shape, lo, hi):
        n = int(np.prod(shape))
        span = hi - lo + 1
        return np.array([lo + self.next() % span for _ in range(n)], dtype=np.int64).reshape(shape)


def _shift_for(k, a_rms, b_rms, target_rms=24):
    """Right shift (on top of a 2**30-scale multiplier) that brings an int32
    accumulator of rms a_rms * b_rms * sqrt(k) down to about target_rms."""
    from math import isqrt

    ratio = max(a_rms * b_rms * isqrt(k) // target_rms, 1)
    return 30 + ratio.bit_length() - 1


def build(cfg=None):
    cfg = cfg or Config()
    rng = SplitMix64(cfg.seed)
    d, f, hd = cfg.dim, cfg.ffn, cfg.head_dim
    w_rms = 37  # rms of uniform [-64, 63]

    def lin(k, n):
        w = rng.ints((k, n), -64, 63).astype(np.int8)
        return {"w": w, "m": int((1 << 30) + rng.next() % (1 << 29)), "s": _shift_for(k, 32, w_rms)}

    def norm(n):
        return {"g": rng.ints((n,), 32, 96).astype(np.int8), "m": 1 << 30, "s": 30 + 9}

    layers = []
    for _ in range(cfg.layers):
        layers.append(
            {
                "norm1": norm(d),
                "q": lin(d, d),
                "k": lin(d, d),
                "v": lin(d, d),
                "o": lin(d, d),
                "norm2": norm(d),
                "gate": lin(d, f),
                "up": lin(d, f),
                "down": lin(f, d),
                "res": {"ma": 1 << 14, "mb": 1 << 14, "s": 15},
            }
        )
    cos, sin = tables.rope_tables(cfg.max_pos, hd)
    return {
        "cfg": cfg,
        "embed": rng.ints((cfg.vocab, d), -64, 63).astype(np.int8),
        "layers": layers,
        "norm_f": norm(d),
        "out": rng.ints((d, cfg.vocab), -64, 63).astype(np.int8),
        "cos": np.array(cos, dtype=np.int64),
        "sin": np.array(sin, dtype=np.int64),
        "exp_lut": np.array(tables.exp_lut(), dtype=np.int64),
        "silu_lut": np.array(tables.silu_lut(), dtype=np.int8),
        # scores = q.k over head_dim int8 values; this shift maps them to
        # exp-table steps.
        "score_shift": _shift_for(hd, 32, 32, target_rms=16) - 30,
        "pv": {"m": 1 << 30, "s": 30 + 8},  # probabilities carry a 1/256 scale
    }


class Trace:
    """Records a sha256 of every intermediate tensor, as int64 little-endian."""

    def __init__(self):
        self.digests = []

    def add(self, label, x):
        data = np.ascontiguousarray(np.asarray(x), dtype="<i8").tobytes()
        self.digests.append((label, hashlib.sha256(data).hexdigest()))
        return x

    def digest(self):
        h = hashlib.sha256()
        for label, d in self.digests:
            h.update(f"{label}:{d};".encode())
        return h.hexdigest()


def _linear(be, x, p):
    return be.requantize(be.matmul(x, p["w"]), p["m"], p["s"])


def forward(be, model, tokens, start, cache, trace):
    """Run positions start .. start + len(tokens) - 1. cache holds int8 K and V
    per layer ([S, heads, head_dim]) and is extended in place. Returns int32
    logits [T, vocab]."""
    cfg = model["cfg"]
    t, h, hd = len(tokens), cfg.heads, cfg.head_dim
    pos = np.arange(start, start + t)
    cos, sin = model["cos"][pos], model["sin"][pos]
    x = model["embed"][np.asarray(tokens)]
    for i, layer in enumerate(model["layers"]):
        n = layer["norm1"]
        a = trace.add(f"L{i}.norm1", be.rmsnorm(x, n["g"], n["m"], n["s"]))
        q = be.rope(_linear(be, a, layer["q"]).reshape(t, h, hd), cos, sin)
        k = be.rope(_linear(be, a, layer["k"]).reshape(t, h, hd), cos, sin)
        v = _linear(be, a, layer["v"]).reshape(t, h, hd)
        ck, cv = cache.setdefault(i, [np.zeros((0, h, hd), np.int8)] * 2)
        ck, cv = np.concatenate([ck, k]), np.concatenate([cv, v])
        cache[i] = [ck, cv]
        s_len = ck.shape[0]
        valid = np.arange(s_len)[None, :] <= pos[:, None]  # causal mask
        heads = []
        for j in range(h):
            scores = be.matmul(q[:, j, :], ck[:, j, :].T)
            probs = be.softmax(scores, valid, model["score_shift"], model["exp_lut"])
            heads.append(
                be.requantize(be.matmul(probs, cv[:, j, :]), model["pv"]["m"], model["pv"]["s"])
            )
        attn = trace.add(f"L{i}.attn", np.concatenate(heads, axis=-1))
        r = layer["res"]
        x = trace.add(
            f"L{i}.res1", be.add_requant(x, _linear(be, attn, layer["o"]), r["ma"], r["mb"], r["s"])
        )
        n = layer["norm2"]
        b = be.rmsnorm(x, n["g"], n["m"], n["s"])
        gate = be.lut(_linear(be, b, layer["gate"]), model["silu_lut"])
        hidden = be.mul_requant(gate, _linear(be, b, layer["up"]), 1 << 30, 30 + 5)
        x = trace.add(
            f"L{i}.res2",
            be.add_requant(x, _linear(be, hidden, layer["down"]), r["ma"], r["mb"], r["s"]),
        )
    n = model["norm_f"]
    logits = be.matmul(be.rmsnorm(x, n["g"], n["m"], n["s"]), model["out"])
    return trace.add("logits", logits)


def generate(be, model, prompt, steps, trace=None, prefill=True):
    """Greedy decoding. With prefill=False the prompt is fed one token at a
    time; integer kernels make both paths bit-identical."""
    trace = trace or Trace()
    cache, tokens = {}, list(prompt)
    if prefill:
        logits = forward(be, model, tokens, 0, cache, trace)
    else:
        for p, tok in enumerate(tokens):
            logits = forward(be, model, [tok], p, cache, Trace())
    for _ in range(steps):
        nxt = int(be.argmax(logits[-1:])[0])
        tokens.append(nxt)
        logits = forward(be, model, [nxt], len(tokens) - 1, cache, trace)
    return tokens[len(prompt) :], trace
