"""Float32 reference for Qwen2.5-0.5B, written out op by op so the integer
version can mirror it exactly. Checked against Hugging Face transformers."""

import json
import os
from pathlib import Path

import torch
from safetensors.torch import load_file

# QWEN_WORK holds model/ (Qwen2.5-0.5B), data/ (WikiText-2) and the converted .pkl
MODEL = Path(os.environ.get("QWEN_WORK", Path(__file__).parent)) / "model"


def load(model_dir=MODEL):
    cfg = json.loads((Path(model_dir) / "config.json").read_text())
    w = {k: v.float() for k, v in load_file(str(Path(model_dir) / "model.safetensors")).items()}
    return cfg, w


def rmsnorm(x, g, eps):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * g


def rope_tables(cfg, n):
    hd = cfg["hidden_size"] // cfg["num_attention_heads"]
    inv = 1.0 / (cfg["rope_theta"] ** (torch.arange(0, hd, 2, dtype=torch.float64) / hd))
    ang = torch.arange(n, dtype=torch.float64)[:, None] * inv[None, :]
    return torch.cos(ang).float(), torch.sin(ang).float()


def rope(x, cos, sin):  # x: [T, H, D], half-split rotation as in Qwen2 / Llama
    h = x.shape[-1] // 2
    x1, x2 = x[..., :h], x[..., h:]
    c, s = cos[:, None, :], sin[:, None, :]
    return torch.cat([x1 * c - x2 * s, x1 * s + x2 * c], dim=-1)


def forward(cfg, w, tokens, hook=None):
    """Full-sequence causal forward. hook(name, tensor) sees every linear input
    and the residual stream, for calibration."""
    hook = hook or (lambda n, t: None)
    eps = cfg["rms_norm_eps"]
    nh, nkv = cfg["num_attention_heads"], cfg["num_key_value_heads"]
    d = cfg["hidden_size"]
    hd = d // nh
    t = len(tokens)
    cos, sin = rope_tables(cfg, t)
    x = w["model.embed_tokens.weight"][torch.as_tensor(tokens)]
    mask = torch.full((t, t), float("-inf")).triu(1)
    for i in range(cfg["num_hidden_layers"]):
        p = f"model.layers.{i}."
        hook(f"L{i}.resid_in", x)
        a = rmsnorm(x, w[p + "input_layernorm.weight"], eps)
        hook(f"L{i}.qkv_in", a)
        q = a @ w[p + "self_attn.q_proj.weight"].T + w[p + "self_attn.q_proj.bias"]
        k = a @ w[p + "self_attn.k_proj.weight"].T + w[p + "self_attn.k_proj.bias"]
        v = a @ w[p + "self_attn.v_proj.weight"].T + w[p + "self_attn.v_proj.bias"]
        hook(f"L{i}.q", q)
        hook(f"L{i}.k", k)
        hook(f"L{i}.v", v)
        q = rope(q.view(t, nh, hd), cos, sin)
        k = rope(k.view(t, nkv, hd), cos, sin)
        v = v.view(t, nkv, hd)
        rep = nh // nkv
        k, v = k.repeat_interleave(rep, dim=1), v.repeat_interleave(rep, dim=1)
        att = torch.einsum("thd,shd->hts", q, k) / hd**0.5 + mask
        hook(f"L{i}.scores", att)
        att = att.softmax(-1)
        o = torch.einsum("hts,shd->thd", att, v).reshape(t, d)
        hook(f"L{i}.o_in", o)
        x = x + o @ w[p + "self_attn.o_proj.weight"].T
        hook(f"L{i}.resid_mid", x)
        b = rmsnorm(x, w[p + "post_attention_layernorm.weight"], eps)
        hook(f"L{i}.mlp_in", b)
        gate = b @ w[p + "mlp.gate_proj.weight"].T
        up = b @ w[p + "mlp.up_proj.weight"].T
        hook(f"L{i}.gate", gate)
        hook(f"L{i}.up", up)
        h = torch.nn.functional.silu(gate) * up
        hook(f"L{i}.down_in", h)
        x = x + h @ w[p + "mlp.down_proj.weight"].T
    hook("final_resid", x)
    x = rmsnorm(x, w["model.norm.weight"], eps)
    hook("head_in", x)
    return x @ w["model.embed_tokens.weight"].T


if __name__ == "__main__":
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL)
    ids = tok("The same model, the same prompt, the same bits on every chip.")["input_ids"]
    cfg, w = load()
    with torch.no_grad():
        mine = forward(cfg, w, ids)
        hf = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32)
        ref = hf(torch.tensor([ids])).logits[0]
    diff = (mine - ref).abs().max().item()
    same = bool((mine.argmax(-1) == ref.argmax(-1)).all())
    print(f"tokens={len(ids)} max|logit diff|={diff:.3e} argmax equal={same}")
