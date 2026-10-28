# Overflow bounds

samebit's determinism argument rests on two facts. Integer addition is
associative, and no intermediate value overflows its width. The first is
mathematics. This file proves the second for each operation in SPEC.md
version 0.1. Machine-checked versions of these proofs are on the roadmap.

## 1. matmul: any summation order fits in int32

**Claim.** Let `p_t = A[i, t] * B[t, j]` for `t = 0..K-1`, and suppose
`K * max|A| * max|B| <= 2^31 - 1`. Then every partial sum over any subset of
the `p_t`, added in any order, lies in `[-(2^31 - 1), 2^31 - 1]`.

**Proof.** For any subset `T`, `|sum over T of p_t| <= sum over T of |p_t|
<= K * max|A| * max|B| <= 2^31 - 1`. Every intermediate value of any
evaluation order, including tree reductions split across threads, warps or
cores, is such a partial sum. ∎

So no evaluation order overflows, and because integer addition is
associative and commutative, every order gives the same result.

**int8 case.** `max|a * b| = 128 * 128 = 16384`, attained at
`(-128) * (-128)`. The condition is `16384 * K <= 2^31 - 1`, so
`K <= 131071`. At `K = 131072` the all-`-128` input reaches exactly `2^31`
and overflows. `tests/test_ops.py::test_matmul_worst_case_bound` runs
`K = 131071` on every backend.

**Attention case.** Probabilities are at most 255, so `max|p * v| = 255 * 128
= 32640` and `K <= 65793` key positions.

## 2. requantize

`|x| <= 2^31` and `0 <= m < 2^31` give `|x * m| < 2^62`. With `s <= 62`, the
rounding term is at most `2^61`, and `|x * m + 2^(s-1)| < 2^62 + 2^61 < 2^63`.
int64 is enough.

## 3. rmsnorm

- `q = sum of x_i^2 <= D * 2^14`, so `n = floor(q * 2^16 / D) <= 2^30`. This
  is inside the domain of the 32-bit `isqrt`, and `q * 2^16 <= D * 2^30`
  fits in int64 for any `D < 2^33`.
- `r >= 1` and `inv <= 2^30`, so `|x_i * inv| <= 2^37`.
- `|u_i| <= 2^37 / 2^14 + 1 <= 2^23 + 1`, and `|u_i * g_i| <= 2^30 + 128`,
  which is below the `2^31` bound required by requantize.
- `isqrt` keeps `res + bit < 2^32` at every step.

## 4. rope

`|x1 * c| + |x2 * s| <= 2 * 128 * 2^14 = 2^22`. int32 is enough.

## 5. softmax

`e_j <= 65535 < 2^16`, so `e_j * 512 < 2^25`. `S <= 2^16 * cols`, so
`2 * S < 2^48` for any `cols < 2^31`. int64 is enough. `S > 0` always,
because the row maximum has `d = 0` and `lut[0] = 65535`.

## 6. mul_requant and add_requant

- `mul_requant`: `|a * b| <= 2^14` and `|a * b * m| < 2^45`.
- `add_requant`: for `|ma|, |mb| < 2^31`, `|a * ma + b * mb| < 2^39`.

## What these proofs do not cover

- They show that conforming backends agree with each other. They say nothing
  about how close the integer model is to a floating-point original. That is
  an accuracy question, and stage 2 measures it on a real model.
- They assume the implementation computes the spec. The conformance suite
  checks that empirically on every backend. A machine-checked proof that a
  kernel matches the spec is future work.
