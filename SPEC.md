# samebit specification, version 0.1

This document defines every samebit operation exactly. An implementation
conforms when it returns the same integers as the definitions below for every
input that satisfies the stated preconditions. `src/samebit/reference.py` is
the executable form of this document; when the two disagree, that is a bug in
one of them and should be reported.

No operation uses floating point. Floating-point numbers appear only in the
offline construction of lookup tables (section 4), and even there they are
computed with 60-digit decimal arithmetic, not with a platform's `libm`.

## 1. Notation

All values are mathematical integers. Implementations may use any integer
width that holds every intermediate value; section 5 gives the widths that
are always sufficient.

| Symbol | Meaning |
|---|---|
| `floor(a / b)` | Division rounded toward negative infinity. In this spec it is only applied with `a >= 0, b > 0`, where every common language agrees. |
| `sat(v)` | `min(max(v, -128), 127)` |
| `rshift_round(v, s)` | `v` if `s = 0`, else `floor((v + 2^(s-1)) / 2^s)`: rounds half up, toward positive infinity, for either sign of `v` |
| `isqrt(n)` | `floor(sqrt(n))` for `0 <= n < 2^32` |

`rshift_round` is defined on signed values. C's `>>` on a negative signed
value is implementation-defined, and C's `/` truncates toward zero, so a C
implementation must spell out the floor (see `src/samebit/csrc/samebit.c`).

Examples: `rshift_round(-3, 1) = -1`, `rshift_round(-6, 2) = -1`,
`rshift_round(-7, 2) = -2`, `rshift_round(3, 1) = 2`.

## 2. Tensors

Activations and weights are int8. Accumulators are int32. Per-channel
parameters apply along the last axis; a scalar parameter applies to every
channel. Probabilities are integers in `[0, 255]` with an implied scale of
1/256.

## 3. Operations

### 3.1 `matmul(A, B)`

`A` is `[M, K]`, `B` is `[K, N]`. `C[i, j] = sum over t of A[i, t] * B[t, j]`.

Precondition: `K * max|A| * max|B| <= 2^31 - 1`. For int8 inputs this is
`K <= 131071`; for probabilities times int8 it is `K <= 65793`.

Under the precondition every partial sum, in any order and with any blocking,
fits in int32 (section 5.1). Implementations may therefore reorder, tile,
split across threads or use int8 dot-product instructions freely.

### 3.2 `requantize(x, m, s)`

`y = sat(rshift_round(x * m, s))`, with `|x| < 2^31`, `0 <= m < 2^31` and
`0 <= s <= 62`. `m` and `s` may be per channel.

### 3.3 `rmsnorm(x, g, m, s)`

For each row `x[0..D-1]`:

1. `q = sum of x_i^2`
2. `n = floor(q * 2^16 / D)`
3. `r = max(isqrt(n), 1)`, so `r` is about `rms(x) * 2^8`
4. `inv = floor(2^30 / r)`
5. `u_i = rshift_round(x_i * inv, 14)`, so `u_i` is about `x_i / rms(x) * 2^8`
6. `y_i = sat(rshift_round(u_i * g_i * m, s))`

`isqrt` must be exact. The reference uses the 16-step digit-by-digit method,
which needs only comparisons, subtraction and shifts and so runs unchanged on
any backend.

### 3.4 `rope(x, cos, sin)`

`x` is `[T, H, D]`, `cos` and `sin` are `[T, D/2]` tables in Q14 (section
4.3). For `j < D/2`, with `x1 = x[..., j]` and `x2 = x[..., j + D/2]`:

- `y[..., j] = sat(rshift_round(x1 * cos[t, j] - x2 * sin[t, j], 14))`
- `y[..., j + D/2] = sat(rshift_round(x1 * sin[t, j] + x2 * cos[t, j], 14))`

### 3.5 `softmax(scores, valid, shift, lut)`

For each row, over the positions where `valid` is true (at least one per
row):

1. `mx = max of scores_j`
2. `d_j = min(rshift_round(mx - scores_j, shift), len(lut) - 1)`
3. `e_j = lut[d_j]`, and `e_j = 0` where `valid` is false
4. `S = sum of e_j`. `S >= lut[0] > 0` because the maximum has `d = 0`.
5. `p_j = min(floor((e_j * 512 + S) / (2 * S)), 255)`, which is
   `e_j * 256 / S` rounded half up and clamped

### 3.6 `lut(x, table)`

`y = table[x + 128]` for int8 `x` and a 256-entry int8 table.

### 3.7 `mul_requant(a, b, m, s)` and `add_requant(a, b, ma, mb, s)`

- `mul_requant`: `y = sat(rshift_round(a * b * m, s))`
- `add_requant`: `y = sat(rshift_round(a * ma + b * mb, s))`

### 3.8 `argmax(x)`

The index of the largest value in each row. Ties go to the lowest index.
(Several libraries do not promise which tied index they return; a conforming
backend must.)

## 4. Lookup tables

Tables are model data. Backends only index them, so the kernels never depend
on how a table was built. The construction is still pinned, so that two
parties who build a table from this spec get the same bytes.

All tables are computed with Python's `decimal` module at 60 significant
digits and rounded half to even (`src/samebit/tables.py`).

### 4.1 Exponential: `lut[d] = round(65535 * exp(-d / 16))`, `d = 0..255`.

### 4.2 SiLU: `lut[q + 128] = sat(round(silu(q / 16) * 16))`, `q = -128..127`.

### 4.3 Rotary tables

`cos[p, i] = round(2^14 * cos(p * f_i))` and likewise for `sin`, with
`f_i = 10000^(-2i / D)`. Angles are reduced modulo 2π with a 60-digit π and
evaluated by Taylor series.

## 5. Bounds

`docs/BOUNDS.md` proves that the widths below are enough.

| Quantity | Width |
|---|---|
| matmul accumulator under the 3.1 precondition | int32 |
| `x * m` in `requantize` | int64 |
| `q * 2^16` and `n` in `rmsnorm` | int64; `n < 2^31` |
| softmax `e_j * 512 + S`, `2 * S` | int64 |

## 6. Conformance

`python -m samebit` runs the reference decoder (`src/samebit/model.py`) on
every installed backend and compares a SHA-256 digest of all 153 intermediate
tensors with `src/samebit/golden.json`. A backend conforms to version 0.1 when
its digest matches and it passes `tests/`.
