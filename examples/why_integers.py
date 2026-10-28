"""Why integers: the same dot product, summed in different orders.

Floating-point addition is not associative, so a GPU that splits a reduction
differently (another batch size, another chip, another kernel) can return
different bits. Integer addition is associative: every order agrees.
"""

import numpy as np

rng = np.random.default_rng(7)
k = 4096
a = rng.integers(-128, 128, k)
b = rng.integers(-128, 128, k)
fa, fb = (a * 0.01).astype(np.float32), (b * 0.01).astype(np.float32)


def float_sum(order, block):
    acc = np.float32(0)
    for start in range(0, k, block)[::order]:
        part = np.float32(0)
        for t in range(start, min(start + block, k)):
            part = np.float32(part + fa[t] * fb[t])
        acc = np.float32(acc + part)
    return acc


def int_sum(order, block):
    acc = 0
    for start in range(0, k, block)[::order]:
        acc += sum(int(a[t]) * int(b[t]) for t in range(start, min(start + block, k)))
    return acc


print(f"{'order':>24} {'float32 result':>22} {'int32 result':>14}")
for order, block in [(1, k), (1, 64), (-1, 64), (1, 7), (-1, 1)]:
    f = float_sum(order, block)
    print(
        f"{('forward' if order == 1 else 'backward') + f', blocks of {block}':>24} "
        f"{float(f):>11.7f} ({f.view(np.uint32):08x}) {int_sum(order, block):>14}"
    )
