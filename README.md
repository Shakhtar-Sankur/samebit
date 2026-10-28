# samebit

**The same model, the same prompt, the same bits, on every chip.**

samebit is an integer-only inference specification with conforming kernels
for several backends. A model that follows the spec produces bit-identical
intermediate tensors, logits and tokens on every conforming backend, with any
batch size, thread count or summation order.

```
$ python -m samebit
golden e5aa829b49bc2cef  tokens [195, 18, 29, 236, 112, 255, 129, 248, 13, 41, 129, 128, 50, 62, 114, 16]
reference  e5aa829b49bc2cef  PASS  (0.29s)
numpy      e5aa829b49bc2cef  PASS  (0.02s)
c          e5aa829b49bc2cef  PASS  (0.08s)
torch      e5aa829b49bc2cef  PASS  (0.06s)
mlx        e5aa829b49bc2cef  PASS  (0.46s)
```

Each line is a SHA-256 over all 153 intermediate tensors of a 16-token greedy
generation. All five implementations are written separately and share no
arithmetic code.

Continuous integration reproduces the same digest on x86-64 Linux and on ARM64
Apple-silicon macOS, with Python 3.9 and 3.12: 4 machines × 5 backends, one
digest ([run](https://github.com/Shakhtar-Sankur/samebit/actions/runs/40629298323)).
On an NVIDIA Tesla T4 (Kaggle, 28 October 2026), the CUDA backend reproduces the same
digest, with its matmul split across thread blocks and combined by `atomicAdd`, and
all 60 tests pass:

```
cuda: Tesla T4
cuda       e5aa829b49bc2cef  PASS  (0.27s)
```

## A real model: Qwen2.5-0.5B

Spec v0.2 ([docs/SPEC-v0.2.md](docs/SPEC-v0.2.md)) carries a real pretrained model, Qwen2.5-0.5B
(494M parameters, 24 layers, grouped-query attention), entirely in integers: int8 weights, int16
activations, an int32 residual stream and int64 accumulators.

Accuracy on WikiText-2 test, against the original float32 model:

| | Tokens | Float32 perplexity | samebit integer perplexity | Same top-1 token |
|---|---|---|---|---|
| Qwen2.5-0.5B | 10,240 (40 × 256) | 22.99 | 23.16 (+0.75%) | 93.25% |

Calibration uses only WikiText-2 *train*; the test text is never seen during conversion.

Bit-identity across three separately written implementations, checked on all 24 residual
streams and the logits (`tools/qwen2/determinism.py`):

```
numpy-blas   35650ae52353be4c  (12.1s)
numpy-int64  35650ae52353be4c  (111.9s)
torch        35650ae52353be4c  (6.5s)
32 tokens, 24 residual streams + logits: IDENTICAL
```

The NumPy BLAS path does its integer matmuls in float64. That is exact because the spec bounds
every partial sum below 2^53. A floating-point unit therefore gives the same bits as an integer
unit. Steps to reproduce are in [tools/qwen2](tools/qwen2/README.md).

## The problem

Ask a model the same question twice at temperature 0 and you can get two
different answers. Floating-point addition is not associative, so the result
of a reduction depends on how a kernel splits it. That split changes with the
batch size, the GPU model, the vendor, the kernel library and the version.

`examples/why_integers.py` sums one 4096-element dot product in five orders:

```
                   order         float32 result   int32 result
 forward, blocks of 4096 -31.6259975 (c1fd020b)        -316260
   forward, blocks of 64 -31.6260014 (c1fd020d)        -316260
  backward, blocks of 64 -31.6259975 (c1fd020b)        -316260
    forward, blocks of 7 -31.6260242 (c1fd0219)        -316260
   backward, blocks of 1 -31.6259823 (c1fd0203)        -316260
```

Four different float32 results, one integer result.

Reproducible inference matters because:

- **Reinforcement learning:** a sampler and a trainer that compute
  log-probabilities differently is a known source of instability.
- **Audits:** evaluation, safety and regulatory work need results anyone can
  reproduce.
- **Verification:** if an inference provider's output is fully determined,
  anyone can check that they ran the model they claim.
- **Hardware portability:** a model can give the same answer on an NVIDIA
  GPU, an AMD GPU, an Apple laptop and a phone.

Recent work makes floating-point inference batch-invariant or invariant
across tensor-parallel sizes within one software stack. samebit takes a
different route: it removes floating point from the runtime, so agreement
follows from integer arithmetic and does not depend on any vendor's kernels.

## How it works

- **Integer arithmetic only.** int8 weights and activations, exact int32
  accumulation and fixed-point requantization (`rshift_round`, round half up,
  defined for negative values too).
- **Proved bounds.** `docs/BOUNDS.md` shows that no intermediate overflows in
  any summation order. Up to K = 131071, an int8 matmul is exact however it
  is tiled, threaded or vectorised.
- **Exact nonlinearities.** RMSNorm uses an exact 16-step integer square root.
  Softmax, SiLU and rotary embeddings use lookup tables built with 60-digit
  decimal arithmetic, so table bytes do not depend on a platform's `libm`.
- **Pinned tie-breaking.** argmax returns the lowest tied index on every
  backend.

The spec is [SPEC.md](SPEC.md). Its executable form is
`src/samebit/reference.py`, written in plain Python integers with explicit
bound checks.

## Backends

| Backend | Where it runs | Status |
|---|---|---|
| `reference` | Plain Python integers | Conforms; the definition |
| `numpy` | Any CPU | Conforms |
| `c` | Portable C99 with optional OpenMP; three accumulation orders | Conforms; all orders agree |
| `torch` | PyTorch CPU | Conforms |
| `mlx` | Apple MLX (Metal on Apple silicon, CPU elsewhere) | Conforms on CPU; Metal not yet run |
| `cuda` | NVIDIA GPUs (CuPy, kernels compiled at run time) | Conforms on a Tesla T4 (Kaggle, 28 Oct 2026); also checked under CPU emulation on every push |
| ROCm / HIP | AMD GPUs | Planned |

## Quick start

```bash
git clone https://github.com/Shakhtar-Sankur/samebit
cd samebit
pip install -e ".[test]"         # numpy and pytest
pip install torch mlx            # optional extra backends
python -m samebit                # conformance check against the golden digest
pytest                           # op-level and end-to-end tests
python examples/why_integers.py  # floats vs integers
```

The C backend compiles itself on first use with the system C compiler.

### On an NVIDIA GPU

Open [`examples/kaggle_gpu.ipynb`](examples/kaggle_gpu.ipynb) in Kaggle, set the accelerator to a
T4 or P100 and run all cells, or locally:

```bash
pip install -e ".[test,cuda]"
python -m samebit cuda
```

The CUDA matmul splits every dot product across `SAMEBIT_CUDA_SPLITK` thread blocks (default 8)
and combines them with `atomicAdd`, so the order of the final additions changes on every run. The
digest must not.

Without a GPU, `tests/emulated_cupy` stands in for CuPy: it compiles the same CUDA source as C++,
runs every block and thread on the CPU, and shuffles the split-K blocks. CI runs the CUDA kernels
this way on every push.

## Tests

- **Op conformance** (`tests/test_ops.py`): every op on every backend must
  match the reference exactly. The tests cover the int8 minimum, saturation,
  rounding of negative halves, all-tied rows, masked softmax and the
  largest matmul depth allowed by the int32 bound.
- **End to end** (`tests/test_model.py`): a 2-layer, Llama-shaped decoder
  with RMSNorm, rotary attention with a KV cache and SwiGLU must reproduce
  the golden digest on every backend. Processing the prompt in one batch or
  one token at a time must give identical logits.
- **Order independence:** the C backend runs ascending, descending and
  blocked back-to-front accumulation and must agree.

## Roadmap

1. **Stage 1, done:** spec v0.1, proved bounds, five conforming backends, and
   a conformance suite with a golden digest.
2. **Stage 2:** CUDA kernels (done: conform on a Tesla T4) and Metal.
   Convert a real open model to the spec (done: Qwen2.5-0.5B, spec v0.2, three implementations
   bit-identical). Next: v0.2 kernels for C, CUDA and Metal, and the speed cost of determinism.
3. **Stage 3:** AMD ROCm. Machine-checked proofs that the kernels meet the
   spec and the bounds. Preprint.
4. **Stage 4:** a public auditor that checks, from outputs alone, whether a
   hosted model matches its published weights.

## Prior art

Integer-only inference is well established: gemmlowp and TensorFlow Lite
fixed-point requantization, and integer-only transformers such as I-BERT
(Kim et al., 2021). samebit applies these ideas to a different goal. It turns
integer inference into an exact, cross-vendor conformance target with a
written spec, proved bounds and a golden digest.

## Citation

See [CITATION.cff](CITATION.cff).

## Licence

Apache-2.0. Copyright 2026 Sankur Kundu.
