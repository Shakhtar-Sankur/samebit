"""Qwen2.5-0.5B forward pass in samebit v0.2 integers. Every arithmetic step is
an op from samebit.v2.ops (or ops_torch); the runner only gathers, slices and concatenates."""

import numpy as np

from . import ops as O


def forward(P, tokens, ops=O, trace=None):
    cfg = P["cfg"]
    nh, nkv, d = cfg["num_attention_heads"], cfg["num_key_value_heads"], cfg["hidden_size"]
    hd, rep = d // nh, nh // nkv
    t = len(tokens)
    tok = np.asarray(tokens)
    cos, sin = P["cos"][:t], P["sin"][:t]
    valid = np.tril(np.ones((t, t), dtype=bool))
    # embedding row (int8) -> residual Q12, per-row multiplier
    x = ops.requant(P["embed"][tok].T, P["embed_m"][tok], P["embed_s"], 32).T
    for i, L in enumerate(P["layers"]):
        a = ops.rmsnorm(x, L["norm1_g"], P["eps_q24"], L["norm1_s"])

        def lin(inp, name, bits, L=L):
            p = L[name]
            return ops.requant(ops.matmul(inp, p["w"]), p["m"], p["s"], bits, p["b"])

        q = ops.rope(lin(a, "q", 16).reshape(t, nh, hd), cos, sin)
        k = ops.rope(lin(a, "k", 16).reshape(t, nkv, hd), cos, sin)
        v = lin(a, "v", 16).reshape(t, nkv, hd)
        heads = []
        for h in range(nh):
            g = h // rep
            sc = ops.matmul(q[:, h, :], k[:, g, :].T)
            pr = ops.softmax(sc, valid, L["sm_m"], L["sm_s"], P["exp_lut"])
            heads.append(ops.requant(ops.matmul(pr, v[:, g, :]), L["pv_m"], L["pv_s"], 16))
        o = np.concatenate(heads, axis=-1)
        x = ops.add_sat(x, lin(o, "o", 32), 32)
        b = ops.rmsnorm(x, L["norm2_g"], P["eps_q24"], L["norm2_s"])
        gate = ops.lut16(lin(b, "gate", 16), L["silu"])
        hmid = ops.mul_requant(gate, lin(b, "up", 16), L["h_m"], L["h_s"], 16)
        x = ops.add_sat(x, lin(hmid, "down", 32), 32)
        if trace is not None:
            trace.append(x.copy())
    hf = ops.rmsnorm(x, P["normf_g"], P["eps_q24"], P["normf_s"])
    return ops.requant(ops.matmul(hf, P["head_w"]), P["head_m"], P["head_s"], 63)
