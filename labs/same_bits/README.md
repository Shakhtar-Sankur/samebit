# The Same-Bits Contract

**An interactive lab: why temperature 0 does not give the same answer twice, and what it costs
to make it.**

You send the same prompt to a language model at temperature 0, twice, and get two different
answers. Nothing was sampled. The server batched your request with different neighbours, the
kernels picked a reduction order for that batch size, and floating-point addition rounded
differently. This lab reproduces that chain, one step per part. It uses real IEEE 754 float16 and
float32 arithmetic, rounded after every operation, with every reduction order written out
explicitly.

| Part | Prediction | What students measure |
|---|---|---|
| A · One sum, five orders | How many results from 5 orders? | 4 distinct float32 sums of the same 4,096 numbers |
| B · The batch picks the kernel | Alone vs batched with 31 others? | 113 of 128 outputs change (float16) |
| C · From bits to words | How many of 64 users get different text? | 7 of 64, at batch 16 in float16 |
| D · Batch-invariant kernels | And with a fixed split? | 0 of 64, and the extra reduction work it costs |
| E · Integer arithmetic | int8 × int8, exact int32 sums? | 0 of 64 at any batch, and 94.5% token agreement with float |

Each part locks a structured prediction before the instruments appear (Predict, Discover,
Explain), and a synthesis asks which contract to choose for an RL trainer and for a regulator.

## Run it

```bash
pip install marimo numpy
marimo run lab_same_bits.py        # or: marimo edit lab_same_bits.py
python -m pytest -q tests          # 8 checks on every number the lab shows
```

The lab depends only on numpy, so it can be exported to WebAssembly and run in a browser
(`marimo export html-wasm`).

## Where the numbers come from

- `engine.py` holds every experiment. `matmul_ordered` accumulates a split-K matrix multiply
  step by step in the chosen format; `splits_for_batch` picks the split the way tuned GPU
  libraries do (split K when the batch is too small to fill the chip). A seeded toy recurrent
  language model (64 wide, 76 tokens) decodes greedily through that kernel.
- The real-scale figures quoted in the lab are from Horace He and Thinking Machines Lab,
  [*Defeating Nondeterminism in LLM Inference*](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/)
  (10 September 2025). There, 1,000 temperature-0 completions of Qwen3-235B gave 80 distinct
  texts, and all 1,000 were identical with batch-invariant kernels (Qwen3-8B: 26 s → 42 s).
- The integer result at real scale is from samebit, in this repository: Qwen2.5-0.5B in integers
  within 0.75% of float32 perplexity, with identical bits on an NVIDIA T4 GPU and an x86 CPU.

Written in the format of [MLSysBook](https://mlsysbook.ai) Co-Labs, the labs of Harvard's open *Machine Learning Systems*
textbook.
