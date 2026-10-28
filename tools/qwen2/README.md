# Qwen2.5-0.5B in samebit v0.2

Converts Qwen2.5-0.5B (24 layers, hidden 896, GQA 14/2 heads, q/k/v biases,
RoPE θ = 10^6, tied embeddings, vocabulary 151,936) to samebit spec v0.2
integers (`docs/SPEC-v0.2.md`). Then measures accuracy against the float
model and checks bit-identity across implementations.

## Setup

```bash
pip install -e ".[test]" torch transformers safetensors pyarrow
cd tools/qwen2
huggingface-cli download Qwen/Qwen2.5-0.5B --local-dir model
mkdir -p data
huggingface-cli download Salesforce/wikitext wikitext-2-raw-v1/test-00000-of-00001.parquet \
  wikitext-2-raw-v1/train-00000-of-00001.parquet --repo-type dataset --local-dir data
mv data/wikitext-2-raw-v1/test-*.parquet data/wt2-test.parquet
mv data/wikitext-2-raw-v1/train-*.parquet data/wt2-train.parquet
```

Set `QWEN_WORK` to use another directory holding `model/` and `data/`.

## On an NVIDIA GPU

Open [`examples/kaggle_qwen.ipynb`](../../examples/kaggle_qwen.ipynb) in Kaggle with a T4 and Internet on, and run all cells.
Or locally: `pip install cupy-cuda12x`, then `python gpu_check.py`.

## Run

```bash
python reference.py      # float32 reference vs Hugging Face transformers
python convert.py        # integer model -> qwen2.5-0.5b.samebit.pkl (~1 min)
python determinism.py    # three implementations, bit for bit
python evaluate.py 40    # perplexity on 40 × 256 tokens of WikiText-2 test
```

| File | Purpose |
|---|---|
| `reference.py` | Op-by-op float32 forward pass, matches transformers to 5.4e-5 in the logits |
| `activation_stats.py` | Per-tensor and per-channel activation ranges |
| `convert.py` | Calibration and conversion to integers |
| `determinism.py` | NumPy (float64 BLAS), NumPy (int64) and PyTorch (int64) runs, compared across all 24 residual streams and the logits |
| `evaluate.py` | Float vs integer perplexity and top-1 agreement |
| `gpu_check.py` | The integer model on an NVIDIA GPU (`samebit.v2.ops_cuda`): bits vs CPU, any summation order, speed vs float16 / float32 |
| `calibration.json` | Pinned activation maxima, so every machine builds the same integer model (fingerprint `e1994bc2e4f9bff7`); `convert.py --recalibrate` recomputes them |
| `../../src/samebit/v2/qwen2.py` | The integer forward pass |

## Results (28 October 2026, x86-64 CPU)

| | Float32 | samebit v0.2 integers |
|---|---|---|
| WikiText-2 test perplexity, 40 × 256 tokens | 22.99 | 23.16 (+0.75%) |
| Same top-1 token as float32 | | 93.25% |
| Bit-identical across NumPy BLAS, NumPy int64, PyTorch | | yes: all 24 residual streams and the logits |
