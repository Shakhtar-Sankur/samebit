"""Convert Qwen2.5-0.5B to samebit spec v0.2 integers.

Float is used only here, offline, to choose constants. The result is a set of
integer tensors and integer (multiplier, shift) pairs; the runtime never sees
a float. Lookup tables are built with 60-digit decimal arithmetic.
"""

import collections
import hashlib
import json
import math
import pickle
import sys
from decimal import ROUND_HALF_EVEN, Decimal, localcontext
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import reference as Q
import torch
from transformers import AutoTokenizer

WORK = Q.MODEL.parent  # holds model/, data/ and the converted .pkl
OUT = WORK / "qwen2.5-0.5b.samebit.pkl"
CALIB = Path(__file__).parent / "calibration.json"
Q12 = 4096  # residual stream: value * 2^12
MARGIN = 2.0  # activation scale = MARGIN * calibrated max / 32767
M_BITS = 24  # requant multipliers are below 2^24
EXP_STEPS = 128  # softmax table steps per nat
EXP_LEN = 4096  # covers 32 nats
EXP_ONE = 1 << 24


def mult_shift(ratio, x_bound, m_bits=M_BITS):
    """Integer (m, s) with m / 2^s ~ ratio (array or scalar; may be negative), |m| < 2^m_bits,
    and |x| * m < 2^62 for |x| <= x_bound."""
    ratio = np.asarray(ratio, dtype=np.float64)
    m_cap = min(2.0**m_bits, 2.0**62 / max(x_bound, 1.0))
    s = math.floor(math.log2(m_cap / np.abs(ratio).max()))
    m = np.rint(ratio * 2.0**s).astype(np.int64)
    if np.abs(m).max() >= m_cap:
        s -= 1
        m = np.rint(ratio * 2.0**s).astype(np.int64)
    assert np.abs(m).max() < m_cap and 0 <= s <= 62, (m.max(), s)
    return m, s


def q8_per_out(w):
    """w: [out, in] float -> int8 [in, out] and per-out scale."""
    sc = w.abs().amax(1).clamp_min(1e-12) / 127.0
    q = torch.round(w / sc[:, None]).clamp(-127, 127).to(torch.int8)
    return q.T.contiguous().numpy(), sc.double().numpy()


def dround(x):
    return int(x.to_integral_value(rounding=ROUND_HALF_EVEN))


def exp_table():
    with localcontext() as c:
        c.prec = 60
        return np.array(
            [dround(EXP_ONE * (Decimal(-d) / EXP_STEPS).exp()) for d in range(EXP_LEN)],
            dtype=np.int64,
        )


def silu_table(s_in, s_out):
    """int16 -> int16: q -> round(silu(q * s_in) / s_out), saturated."""
    with localcontext() as c:
        c.prec = 40
        si, so = Decimal(repr(s_in)), Decimal(repr(s_out))
        out = np.empty(65536, dtype=np.int64)
        for i, q in enumerate(range(-32768, 32768)):
            x = q * si
            out[i] = max(-32768, min(32767, dround(x / (1 + (-x).exp()) / so)))
        return out


def rope_q14(cfg, n):
    cos, sin = Q.rope_tables(cfg, n)
    return np.rint(cos.double().numpy() * 16384).astype(np.int64), np.rint(
        sin.double().numpy() * 16384
    ).astype(np.int64)


def calibrate(cfg, w, n_windows=16, t=256):
    tok = AutoTokenizer.from_pretrained(Q.MODEL)
    text = "".join(pq.read_table(WORK / "data/wt2-train.parquet").column("text").to_pylist())
    ids = tok(text[:400_000])["input_ids"]
    mx = collections.defaultdict(float)

    def hook(name, x):
        x = x.detach()
        if name.endswith("scores"):
            return
        mx[name] = max(mx[name], x.abs().max().item())
        if name.endswith(".gate"):
            mx[name.replace("gate", "silu")] = max(
                mx[name.replace("gate", "silu")], torch.nn.functional.silu(x).abs().max().item()
            )

    with torch.no_grad():
        for j in range(n_windows):
            Q.forward(cfg, w, ids[j * t : (j + 1) * t], hook)
    return dict(mx)


def convert(max_pos=1024):
    cfg, w = Q.load()
    if CALIB.exists() and "--recalibrate" not in sys.argv:
        # pinned maxima (exact float64, hex): every machine builds the same integers
        mx = {k: float.fromhex(v) for k, v in json.loads(CALIB.read_text()).items()}
    else:
        mx = calibrate(cfg, w)
        CALIB.write_text(json.dumps({k: float(v).hex() for k, v in sorted(mx.items())}, indent=0))

    def act(name):
        return MARGIN * mx[name] / 32767.0

    nh, d = cfg["num_attention_heads"], cfg["hidden_size"]
    hd = d // nh
    P = {"cfg": cfg, "layers": [], "calib_max": mx}

    emb_q, emb_s = q8_per_out(w["model.embed_tokens.weight"])  # [D, V], scale per vocab row
    P["head_w"] = emb_q  # tied: head weight [D, V]
    P["embed"] = emb_q.T.copy()  # [V, D] int8
    P["embed_m"], P["embed_s"] = mult_shift(emb_s * Q12, 127)  # row -> Q12

    def norm_params(gamma, s_out):
        # u ~ x/rms * 2^14; y = u * g >> s2 ~ x/rms * gamma / s_out
        ratio = gamma.double().numpy() / s_out / 2**14
        g, s2 = mult_shift(ratio, 2**21, m_bits=31)
        return g, s2

    eps_q24 = round(cfg["rms_norm_eps"] * 2**24)
    for i in range(cfg["num_hidden_layers"]):
        p = f"model.layers.{i}."
        L = {}
        s_qkv_in, s_mlp_in = act(f"L{i}.qkv_in"), act(f"L{i}.mlp_in")
        L["norm1_g"], L["norm1_s"] = norm_params(w[p + "input_layernorm.weight"], s_qkv_in)
        L["norm2_g"], L["norm2_s"] = norm_params(w[p + "post_attention_layernorm.weight"], s_mlp_in)

        def linear(key, s_in, s_out, k_in, bias=True, out_q12=False, p=p):
            wq, ws = q8_per_out(w[p + key + ".weight"])
            acc_bound = 32767.0 * 127 * k_in
            unit = s_in * ws  # real value of one accumulator step
            ratio = unit * Q12 if out_q12 else unit / s_out
            m, s = mult_shift(ratio, acc_bound * 1.5)
            b = None
            if bias:
                b = np.rint(w[p + key + ".bias"].double().numpy() / unit).astype(np.int64)
            return {"w": wq, "m": m, "s": s, "b": b}

        s_q, s_k, s_v = act(f"L{i}.q"), act(f"L{i}.k"), act(f"L{i}.v")
        s_o_in, s_gate, s_up, s_down_in = (
            act(f"L{i}.o_in"),
            act(f"L{i}.gate"),
            act(f"L{i}.up"),
            act(f"L{i}.down_in"),
        )
        L["q"] = linear("self_attn.q_proj", s_qkv_in, s_q, d)
        L["k"] = linear("self_attn.k_proj", s_qkv_in, s_k, d)
        L["v"] = linear("self_attn.v_proj", s_qkv_in, s_v, d)
        L["o"] = linear("self_attn.o_proj", s_o_in, None, d, bias=False, out_q12=True)
        L["gate"] = linear("mlp.gate_proj", s_mlp_in, s_gate, d, bias=False)
        L["up"] = linear("mlp.up_proj", s_mlp_in, s_up, d, bias=False)
        L["down"] = linear(
            "mlp.down_proj", s_down_in, None, cfg["intermediate_size"], bias=False, out_q12=True
        )

        # attention: scores in units of s_q * s_k; table steps of 1/EXP_STEPS nat
        L["sm_m"], L["sm_s"] = mult_shift(s_q * s_k / math.sqrt(hd) * EXP_STEPS, 2.0**37, m_bits=20)
        # P (Q15) @ v (int16) -> o_in int16
        L["pv_m"], L["pv_s"] = mult_shift(2.0**-15 * s_v / s_o_in, 32767.0 * 32767 * max_pos)
        # SwiGLU: silu table, then silu16 * up16 -> down_in int16
        s_silu = MARGIN * mx[f"L{i}.silu"] / 32767.0
        L["silu"] = silu_table(s_gate, s_silu)
        L["h_m"], L["h_s"] = mult_shift(s_silu * s_up / s_down_in, 2.0**30)
        P["layers"].append(L)
        print(f"layer {i} done", file=sys.stderr)

    s_head_in = act("head_in")
    P["normf_g"], P["normf_s"] = norm_params(w["model.norm.weight"], s_head_in)
    # logits: acc_v * s_head_in * emb_s[v]; bring all vocab rows to one unit
    P["head_m"], P["head_s"] = mult_shift(emb_s / emb_s.max(), 32767.0 * 127 * d)
    P["logit_unit"] = float(s_head_in * emb_s.max())
    P["eps_q24"] = eps_q24
    P["exp_lut"] = exp_table()
    P["cos"], P["sin"] = rope_q14(cfg, max_pos)
    return P


def fingerprint(P):
    """SHA-256 over every integer array of a converted model, in a fixed order."""
    h = hashlib.sha256()

    def walk(v):
        if isinstance(v, dict):
            for k in sorted(v):
                if k not in ("cfg", "calib_max", "logit_unit"):
                    h.update(k.encode())
                    walk(v[k])
        elif isinstance(v, list):
            for x in v:
                walk(x)
        elif v is not None:
            h.update(np.asarray(v, dtype="<i8").tobytes())

    walk(P)
    return h.hexdigest()[:16]


if __name__ == "__main__":
    P = convert()
    print("model fingerprint", fingerprint(P))
    with open(OUT, "wb") as f:
        pickle.dump(P, f, protocol=4)
    print("saved")
