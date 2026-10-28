"""Lookup tables, computed with exact decimal arithmetic.

The tables are part of the model, not of the kernels: every backend only
indexes them. They are built with the decimal module at 60 digits so that two
machines with different libm implementations still produce the same table.
"""

from decimal import ROUND_HALF_EVEN, Decimal, localcontext

PREC = 60
PI = Decimal("3.14159265358979323846264338327950288419716939937510582097494")


def _round(d):
    return int(d.to_integral_value(rounding=ROUND_HALF_EVEN))


def _cos_sin(x):
    """cos and sin of a Decimal angle, by range reduction and Taylor series."""
    two_pi = 2 * PI
    x = x - two_pi * int(x / two_pi)
    c, s, term, n = Decimal(0), Decimal(0), Decimal(1), 0
    while True:
        if n % 2 == 0:
            c += term if n % 4 == 0 else -term
        else:
            s += term if n % 4 == 1 else -term
        n += 1
        term = term * x / n
        if abs(term) < Decimal(10) ** (-PREC + 5):
            return c, s


def exp_lut(size=256, step_den=16, one=65535):
    """lut[d] = round(one * exp(-d / step_den))."""
    with localcontext() as ctx:
        ctx.prec = PREC
        return [_round(one * (Decimal(-d) / step_den).exp()) for d in range(size)]


def silu_lut(in_den=16, out_den=16):
    """int8 -> int8 table: q -> round(silu(q / in_den) * out_den), saturated."""
    with localcontext() as ctx:
        ctx.prec = PREC
        out = []
        for q in range(-128, 128):
            x = Decimal(q) / in_den
            y = _round(x / (1 + (-x).exp()) * out_den)
            out.append(max(-128, min(127, y)))
        return out


def rope_tables(max_pos, head_dim, base=10000, one=1 << 14):
    """Q14 cos and sin tables of shape [max_pos][head_dim // 2]."""
    half = head_dim // 2
    with localcontext() as ctx:
        ctx.prec = PREC
        ln_base = Decimal(base).ln()
        freqs = [(-Decimal(2 * i) / head_dim * ln_base).exp() for i in range(half)]
        cos, sin = [], []
        for p in range(max_pos):
            cs = [_cos_sin(p * f) for f in freqs]
            cos.append([_round(c * one) for c, _ in cs])
            sin.append([_round(s * one) for _, s in cs])
        return cos, sin
