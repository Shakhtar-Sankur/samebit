# samebit specification, version 0.2 (draft)

Version 0.1 (`SPEC.md`) fixes int8 activations and int32 accumulators, which
suits small models. Version 0.2 widens the formats so a real pretrained model
keeps its accuracy. The rules that make results identical everywhere do not
change: integer arithmetic only, `rshift_round` (round half up), saturation,
exact integer square root and lookup tables built with decimal arithmetic.

`src/samebit/v2/ops.py` is the executable form. `src/samebit/v2/ops_torch.py`
is a second, separately written implementation, and `tests/test_v2_ops.py`
checks that the two agree exactly.

## Formats

| Tensor | Format |
|---|---|
| Weights | int8, one scale per output channel |
| Activations entering a linear layer, q, k, v | int16, one static scale per tensor (2 × calibrated max / 32767) |
| Residual stream | int32, value × 2^12 (Q12) |
| Accumulators, requant products | int64 |
| Rotary tables | Q14 |
| Attention probabilities | Q15 |
| Requant multipliers | signed, below 2^24 in magnitude, shift 0..62 |

## Ops

- **matmul:** exact integer dot products. An implementation may use float64
  BLAS when every partial sum is an integer below 2^53. For the shapes below
  the largest bound is 32767 × 32767 × 1024 (probabilities times values),
  under 2^41, so this always holds.
- **requant(x, m, s, bits, bias):** `sat_bits(rshift_round((x + bias) × m, s))`.
  The converter picks m and s so that `|x| × |m| < 2^62` for the proved
  accumulator bound.
- **rmsnorm(x, g, eps, s2):** `n = floor(sum(x²) / d) + eps` (eps in Q24),
  `r = isqrt64(n)` (31 fixed steps), `inv = floor(2^40 / r)`,
  `u = rshift_round(x × inv, 26)` (Q14), `y = sat16(rshift_round(u × g, s2))`.
  The precondition is `|x| < 2^26`, `d ≤ 2048`, and the implementation checks it.
  For Qwen2.5-0.5B the largest calibrated residual is 1756, which is 7.2 × 10^6
  in Q12, below 2^23.
- **rope:** half-split rotation with Q14 tables, `rshift_round(·, 14)`, sat16.
- **softmax:** `d = min(rshift_round((max − score) × m, s), 4095)` indexes an
  exp table in Q24 with 128 steps per nat. The result is
  `p = min(floor((e × 2^16 + total) / (2 × total)), 32767)` (Q15). Masked
  positions give 0.
- **SiLU:** a 65536-entry int16 → int16 table per layer.
- **SwiGLU product:** `sat16(rshift_round(silu × up × m, s))`.
- **Residual add:** `sat32(a + b)`.
- **Logits:** int64. Each vocabulary row is requantized to one common unit, so
  argmax and cross-entropy compare like with like.

## Conversion

`tools/qwen2/convert.py` chooses every constant offline from 16 windows of
256 tokens of WikiText-2 *train*. Floats appear only there. The runtime
receives integer tensors and integer (multiplier, shift) pairs.
