"""Measured experiments for the Same-Bits lab.

Every number here is computed, not modelled: numpy's float16 and float32 are IEEE 754 binary16 and
binary32, rounded after every operation, so the bit patterns below are the ones a chip that adds in
the same order produces. Each kernel fixes its reduction order explicitly instead of calling BLAS,
whose order is not specified.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

DTYPES = {"float16": np.float16, "float32": np.float32}


def digest(a: np.ndarray) -> str:
    """First 12 hex digits of the SHA-256 of an array's bytes: equal digests mean equal bits."""
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()[:12]


def ulp_distance(a, b) -> int:
    """How many representable floats lie between a and b (0 means the same bits)."""
    a, b = np.asarray(a), np.asarray(b)
    itype = {2: np.int16, 4: np.int32}[a.dtype.itemsize]
    ia, ib = a.view(itype).astype(np.int64), b.view(itype).astype(np.int64)
    # Map sign-magnitude to a monotone integer line so the subtraction counts steps.
    top = np.int64(1) << (8 * a.dtype.itemsize - 1)
    ia = np.where(ia < 0, -top - ia, ia)
    ib = np.where(ib < 0, -top - ib, ib)
    return int(np.max(np.abs(ia - ib)))


# Part A: one sum, three orders -------------------------------------------------------------

def sum_sequential(x: np.ndarray) -> np.generic:
    """Left to right, rounding after every addition (a single thread's loop)."""
    return np.add.accumulate(x, dtype=x.dtype)[-1]


def sum_tree(x: np.ndarray) -> np.generic:
    """Pairwise tree, the shape of a GPU block reduction."""
    while x.size > 1:
        if x.size % 2:
            x = np.concatenate([x, np.zeros(1, x.dtype)])
        x = (x[0::2] + x[1::2]).astype(x.dtype)
    return x[0]


def sum_blocked(x: np.ndarray, blocks: int) -> np.generic:
    """Split into `blocks` contiguous chunks summed in parallel, then add the partials in order."""
    parts = [sum_sequential(c) for c in np.array_split(x, blocks)]
    return sum_sequential(np.array(parts, dtype=x.dtype))


def addition_orders(n: int = 4096, dtype: str = "float32", seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n).astype(DTYPES[dtype])
    exact = float(np.sum(x.astype(np.float64)))
    results = {
        "left to right": sum_sequential(x),
        "pairwise tree": sum_tree(x),
        "8 blocks": sum_blocked(x, 8),
        "64 blocks": sum_blocked(x, 64),
        "reversed": sum_sequential(x[::-1].copy()),
    }
    ref = results["left to right"]
    return {
        "exact": exact,
        "rows": [
            {"order": k, "value": float(v), "bits": digest(np.array([v])),
             "ulps_from_left_to_right": ulp_distance(np.array([v]), np.array([ref]))}
            for k, v in results.items()
        ],
        "distinct": len({digest(np.array([v])) for v in results.values()}),
    }


# Part B: the batch decides the kernel ------------------------------------------------------

def matmul_ordered(a: np.ndarray, w: np.ndarray, splits: int, dtype) -> np.ndarray:
    """a @ w with the K dimension cut into `splits` chunks (split-K): each chunk accumulates
    left to right, then the partial outputs are added in chunk order. Rounding after every step."""
    a, w = a.astype(dtype), w.astype(dtype)
    partials = []
    for idx in np.array_split(np.arange(a.shape[1]), splits):
        acc = np.zeros((a.shape[0], w.shape[1]), dtype)
        for k in idx:
            acc = (acc + a[:, k:k + 1] * w[k:k + 1, :]).astype(dtype)
        partials.append(acc)
    out = partials[0]
    for p in partials[1:]:
        out = (out + p).astype(dtype)
    return out


def splits_for_batch(batch: int, policy: str) -> int:
    """'throughput': like tuned GPU libraries, split K when the batch is too small to fill the
    chip. 'invariant': the same split whatever the batch, so a row's sums never change order."""
    if policy == "invariant":
        return 4
    return 4 if batch < 4 else (2 if batch < 16 else 1)


def batch_contrast(batch: int, dtype: str = "float16", policy: str = "throughput", seed: int = 1,
                   k: int = 256, n: int = 128) -> dict:
    rng = np.random.default_rng(seed)
    dt = DTYPES[dtype]
    w = (rng.standard_normal((k, n)) / np.sqrt(k)).astype(dt)
    rows = rng.standard_normal((batch, k)).astype(dt)
    alone = matmul_ordered(rows[:1], w, splits_for_batch(1, policy), dt)[0]
    together = matmul_ordered(rows, w, splits_for_batch(batch, policy), dt)[0]
    return {
        "splits_alone": splits_for_batch(1, policy), "splits_batched": splits_for_batch(batch, policy),
        "digest_alone": digest(alone), "digest_batched": digest(together),
        "outputs_differing": int(np.sum(alone.view(np.uint16 if dt == np.float16 else np.uint32)
                                        != together.view(np.uint16 if dt == np.float16 else np.uint32))),
        "outputs": n, "max_ulps": ulp_distance(alone, together),
        "max_abs": float(np.max(np.abs(alone.astype(np.float64) - together.astype(np.float64)))),
    }


# Part C-E: a tiny language model, decoded greedily ----------------------------------------

SYLLABLES = [c + v for c in "bdfgklmnprstvz" for v in "aeiou"] + ["ka", "lo", "mi", "ru", "sa", "te"]


@dataclass
class TinyLM:
    """A seeded toy recurrent language model: big enough to have near-tied logits, small enough to
    decode in a browser. The weights are random; what matters is that the arithmetic is real."""
    vocab: int = 76
    width: int = 64
    seed: int = 7

    def __post_init__(self):
        rng = np.random.default_rng(self.seed)
        self.emb = rng.standard_normal((self.vocab, self.width)) * 0.8
        self.mix = rng.standard_normal((self.width, self.width)) / np.sqrt(self.width) * 1.1
        self.out = rng.standard_normal((self.width, self.vocab)) / np.sqrt(self.width) * 3.0


def _quantize(x: np.ndarray, axis: int):
    scale = np.max(np.abs(x), axis=axis, keepdims=True) / 127.0
    scale = np.where(scale == 0, 1.0, scale)
    return np.clip(np.rint(x / scale), -127, 127).astype(np.int8), scale


def logits(model: TinyLM, h: np.ndarray, mode: str, policy: str, dtype: str = "float16") -> np.ndarray:
    """Output projection for a batch of hidden states.
    mode 'float': split-K float kernel chosen by the batch policy.
    mode 'integer': int8 x int8 with exact int32 sums; any order gives the same bits."""
    batch = h.shape[0]
    if mode == "integer":
        hq, hs = _quantize(h, axis=1)
        wq, ws = _quantize(model.out, axis=0)
        acc = np.zeros((batch, model.vocab), np.int32)
        for idx in np.array_split(np.arange(model.width), splits_for_batch(batch, policy)):
            acc += hq[:, idx].astype(np.int32) @ wq[idx, :].astype(np.int32)
        return acc.astype(np.float64) * hs * ws
    return matmul_ordered(h, model.out, splits_for_batch(batch, policy), DTYPES[dtype]).astype(np.float64)


def decode(model: TinyLM, prompts: list[int], steps: int, batch: int, mode: str, policy: str,
           dtype: str = "float16") -> list[list[int]]:
    """Greedy (temperature 0) decoding of every prompt, served `batch` at a time."""
    outs = []
    for start in range(0, len(prompts), batch):
        group = prompts[start:start + batch]
        tok = np.array(group)
        h = np.zeros((len(group), model.width))
        seqs = [[t] for t in group]
        for _ in range(steps):
            h = np.tanh(h * 0.5 + model.emb[tok] @ model.mix)
            tok = np.argmax(logits(model, h, mode, policy, dtype), axis=1)
            for s, t in zip(seqs, tok):
                s.append(int(t))
        outs.extend(seqs)
    return outs


def words(seq: list[int]) -> str:
    return " ".join("".join(SYLLABLES[t] for t in seq[i:i + 3]) for i in range(0, len(seq), 3))


def serving_contrast(batch: int, mode: str = "float", policy: str = "throughput", dtype: str = "float16",
                     users: int = 64, steps: int = 40, seed: int = 7) -> dict:
    """The same 64 prompts at temperature 0, served one at a time and then `batch` at a time."""
    model = TinyLM(seed=seed)
    prompts = list(range(users))
    alone = decode(model, prompts, steps, 1, mode, policy, dtype)
    together = decode(model, prompts, steps, batch, mode, policy, dtype)
    changed = []
    for i, (a, b) in enumerate(zip(alone, together)):
        if a != b:
            first = next(j for j, (x, y) in enumerate(zip(a, b)) if x != y)
            changed.append({"user": i, "first_divergent_token": first,
                            "alone": words(a), "batched": words(b)})
    return {"users": users, "changed": len(changed), "examples": changed[:3],
            "splits_alone": splits_for_batch(1, policy), "splits_batched": splits_for_batch(batch, policy),
            "digest_alone": digest(np.array(alone)), "digest_batched": digest(np.array(together))}


def integer_agreement(steps: int = 40, users: int = 64, seed: int = 7) -> dict:
    """What integer arithmetic costs: how often its greedy token matches a float64 reference."""
    model = TinyLM(seed=seed)
    prompts = list(range(users))
    ref = decode(model, prompts, steps, 1, "float", "invariant", "float32")
    q = decode(model, prompts, steps, 1, "integer", "invariant")
    same = sum(a == b for r, s in zip(ref, q) for a, b in zip(r[1:], s[1:]))
    first = sum(r[1] == s[1] for r, s in zip(ref, q))
    return {"token_agreement": same / (users * steps), "first_token_agreement": first / users,
            "identical_sequences": sum(r == s for r, s in zip(ref, q)), "users": users}


def extra_reduction_adds(batch: int, policy: str, outputs: int = 128) -> int:
    """Additions spent combining split-K partials for one matmul: the price of a fixed split."""
    return (splits_for_batch(batch, policy) - 1) * batch * outputs
