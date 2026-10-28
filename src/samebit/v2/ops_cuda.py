"""samebit spec v0.2 ops as CUDA kernels, compiled at run time by CuPy (NVRTC).

Arrays stay on the GPU between ops: call prepare(P) once to upload a converted
model, then run samebit.v2.qwen2.forward(P, tokens, ops=ops_cuda).

The matmul splits every dot product into SAMEBIT_CUDA_SPLITK chunks that run
in different thread blocks and are combined with a 64-bit atomicAdd, so the
order of the final additions changes from run to run. Operands are at most
16-bit by the spec, so each product fits in int32; sums are kept in int64.
"""

import os

import cupy as cp
import numpy as np

xp = cp
SPLITK = int(os.environ.get("SAMEBIT_CUDA_SPLITK", "4"))
ROWS = 8  # output rows per thread in matmul: each weight is read once per 8 rows

_SRC = r"""
typedef long long i64;
typedef unsigned long long u64;

__device__ i64 floor_shift(i64 v, int s) {
  if (s == 0) return v;
  if (v >= 0) return v >> s;
  return -(i64)((((u64)(-(v + 1))) >> s) + 1);
}
__device__ i64 rsr(i64 v, int s) { return s == 0 ? v : floor_shift(v + (1LL << (s - 1)), s); }
__device__ i64 satb(i64 v, int bits) {
  i64 hi = (i64)((1ULL << (bits - 1)) - 1), lo = -hi - 1;
  return v < lo ? lo : v > hi ? hi : v;
}
__device__ long long gid() { return (long long)blockIdx.x * blockDim.x + threadIdx.x; }

// out (zeroed) [m, n] += a [m, k] @ b [k, n]; blockIdx.y = group of 8 rows,
// blockIdx.z = k-chunk. Every operand is within int16, so a * b fits in int32.
extern "C" __global__ void mm8(const i64* a, const signed char* b, u64* out,
                               int m, int k, int n, int chunk) {
  int j = blockIdx.x * blockDim.x + threadIdx.x, i0 = blockIdx.y * 8, t0 = blockIdx.z * chunk;
  if (j >= n || t0 >= k) return;
  int t1 = min(k, t0 + chunk), rows = min(8, m - i0);
  i64 acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  for (int t = t0; t < t1; t++) {
    int bv = b[(i64)t * n + j];
#pragma unroll
    for (int r = 0; r < 8; r++)
      if (r < rows) acc[r] += (int)a[(i64)(i0 + r) * k + t] * bv;
  }
#pragma unroll
  for (int r = 0; r < 8; r++)
    if (r < rows) atomicAdd(&out[(i64)(i0 + r) * n + j], (u64)acc[r]);
}

extern "C" __global__ void mm64(const i64* a, const i64* b, u64* out,
                                int m, int k, int n, int chunk) {
  int j = blockIdx.x * blockDim.x + threadIdx.x, i0 = blockIdx.y * 8, t0 = blockIdx.z * chunk;
  if (j >= n || t0 >= k) return;
  int t1 = min(k, t0 + chunk), rows = min(8, m - i0);
  i64 acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  for (int t = t0; t < t1; t++) {
    int bv = (int)b[(i64)t * n + j];
#pragma unroll
    for (int r = 0; r < 8; r++)
      if (r < rows) acc[r] += (int)a[(i64)(i0 + r) * k + t] * bv;
  }
#pragma unroll
  for (int r = 0; r < 8; r++)
    if (r < rows) atomicAdd(&out[(i64)(i0 + r) * n + j], (u64)acc[r]);
}

extern "C" __global__ void requant(const i64* x, const i64* m, const i64* bias, i64* out,
                                   long long total, int cols, int pc, int s, int bits, int hb) {
  long long idx = gid(); if (idx >= total) return;
  int c = (int)(idx % cols);
  i64 v = x[idx];
  if (hb) v += bias[c];
  out[idx] = satb(rsr(v * m[pc ? c : 0], s), bits);
}

extern "C" __global__ void rmsnorm(const i64* x, const i64* g, i64* out,
                                   int rows, int d, long long eps, int s2) {
  int r = (int)gid(); if (r >= rows) return;
  const i64* row = x + (i64)r * d;
  i64 ss = 0;
  for (int c = 0; c < d; c++) ss += row[c] * row[c];
  i64 n = ss / d + eps, res = 0, bit = 1LL << 60;
  for (int it = 0; it < 31; it++) {
    if (n >= res + bit) { n -= res + bit; res = (res >> 1) + bit; } else { res >>= 1; }
    bit >>= 2;
  }
  if (res < 1) res = 1;
  i64 inv = (1LL << 40) / res;
  for (int c = 0; c < d; c++) out[(i64)r * d + c] = satb(rsr(rsr(row[c] * inv, 26) * g[c], s2), 16);
}

extern "C" __global__ void rope(const i64* x, const i64* cs, const i64* sn, i64* out,
                                int t, int h, int d) {
  long long idx = gid();
  int half = d / 2;
  if (idx >= (long long)t * h * half) return;
  int i = (int)(idx % half);
  long long th = idx / half;
  int tt = (int)(th / h);
  i64 c = cs[(i64)tt * half + i], s = sn[(i64)tt * half + i];
  i64 x1 = x[th * d + i], x2 = x[th * d + i + half];
  out[th * d + i] = satb(rsr(x1 * c - x2 * s, 14), 16);
  out[th * d + i + half] = satb(rsr(x1 * s + x2 * c, 14), 16);
}

extern "C" __global__ void softmax(const i64* sc, const unsigned char* ok, const i64* lut,
                                   int lut_len, long long m, int s, i64* out, int rows, int cols) {
  int r = (int)gid(); if (r >= rows) return;
  const i64* row = sc + (i64)r * cols;
  const unsigned char* v = ok + (i64)r * cols;
  i64 mx = -9223372036854775807LL - 1, tot = 0;
  for (int j = 0; j < cols; j++) if (v[j] && row[j] > mx) mx = row[j];
  for (int j = 0; j < cols; j++) {
    if (!v[j]) continue;
    i64 d = rsr((mx - row[j]) * m, s);
    tot += lut[d < lut_len - 1 ? d : lut_len - 1];
  }
  for (int j = 0; j < cols; j++) {
    i64 e = 0;
    if (v[j]) { i64 d = rsr((mx - row[j]) * m, s); e = lut[d < lut_len - 1 ? d : lut_len - 1]; }
    i64 p = (e * 65536 + tot) / (2 * tot);
    out[(i64)r * cols + j] = p < 32767 ? p : 32767;
  }
}

extern "C" __global__ void lut16(const i64* x, const i64* table, i64* out, long long n) {
  long long i = gid(); if (i < n) out[i] = table[x[i] + 32768];
}

extern "C" __global__ void mul_requant(const i64* a, const i64* b, i64* out, long long n,
                                       long long m, int s, int bits) {
  long long i = gid(); if (i < n) out[i] = satb(rsr(a[i] * b[i] * m, s), bits);
}

extern "C" __global__ void add_sat(const i64* a, const i64* b, i64* out, long long n, int bits) {
  long long i = gid(); if (i < n) out[i] = satb(a[i] + b[i], bits);
}

extern "C" __global__ void argmax(const i64* x, i64* out, int rows, int cols) {
  int r = (int)gid(); if (r >= rows) return;
  int best = 0;
  for (int j = 1; j < cols; j++) if (x[(i64)r * cols + j] > x[(i64)r * cols + best]) best = j;
  out[r] = best;
}
"""

_mod = cp.RawModule(code=_SRC, options=("--std=c++14",))
_k = {
    n: _mod.get_function(n)
    for n in ("mm8", "mm64", "requant", "rmsnorm", "rope", "softmax", "lut16")
    + ("mul_requant", "add_sat", "argmax")
}
_T = 256


def _grid(n):
    return ((int(n) + _T - 1) // _T,)


def _d(x, dtype=cp.int64):
    return cp.ascontiguousarray(cp.asarray(x, dtype=dtype))


def _i(v):
    return np.int64(int(v))


def asnumpy(x):
    return cp.asnumpy(x)


def device_name():
    return cp.cuda.runtime.getDeviceProperties(0)["name"].decode()


def sync():
    if hasattr(cp.cuda, "Device"):
        cp.cuda.Device().synchronize()


def prepare(P):
    """Upload every array of a converted model to the GPU (scalars stay on the host)."""
    if isinstance(P, dict):
        return {k: prepare(v) for k, v in P.items()}
    if isinstance(P, list):
        return [prepare(v) for v in P]
    if isinstance(P, np.ndarray) and P.ndim > 0:
        return _d(P, P.dtype)
    return P


def matmul(a, b):
    a = _d(a)
    b = _d(b, cp.int8) if b.dtype == np.int8 else _d(b)
    m, k = a.shape
    n = b.shape[1]
    out = cp.zeros((m, n), dtype=cp.uint64)
    parts = max(1, min(SPLITK, k))
    chunk = (k + parts - 1) // parts
    _k["mm8" if b.dtype == np.int8 else "mm64"](
        ((n + 127) // 128, (m + ROWS - 1) // ROWS, parts),
        (128,),
        (a, b, out, np.int32(m), np.int32(k), np.int32(n), np.int32(chunk)),
    )
    return out.view(cp.int64)


def requant(x, m, s, bits, bias=None):
    x = _d(x)
    cols = x.shape[-1]
    mv = _d(m).reshape(-1)
    hb = bias is not None
    bv = _d(bias) if hb else mv
    out = cp.empty(x.shape, dtype=cp.int64)
    _k["requant"](
        _grid(x.size),
        (_T,),
        (x, mv, bv, out, np.int64(x.size), np.int32(cols), np.int32(mv.size > 1))
        + (np.int32(int(s)), np.int32(bits), np.int32(hb)),
    )
    return out


def rmsnorm(x, g, eps_q24, s2):
    x = _d(x)
    d = x.shape[-1]
    rows = x.size // d
    out = cp.empty(x.shape, dtype=cp.int64)
    _k["rmsnorm"](
        _grid(rows),
        (_T,),
        (x, _d(g), out, np.int32(rows), np.int32(d), _i(eps_q24), np.int32(int(s2))),
    )
    return out


def rope(x, cos, sin):
    x = _d(x)
    t, h, d = x.shape
    out = cp.empty(x.shape, dtype=cp.int64)
    _k["rope"](
        _grid(t * h * (d // 2)),
        (_T,),
        (x, _d(cos), _d(sin), out, np.int32(t), np.int32(h), np.int32(d)),
    )
    return out


def softmax(scores, valid, m, s, lut):
    sc = _d(scores)
    rows, cols = sc.shape
    lut = _d(lut)
    out = cp.empty(sc.shape, dtype=cp.int64)
    _k["softmax"](
        _grid(rows),
        (_T,),
        (sc, _d(valid, cp.uint8), lut, np.int32(lut.size), _i(m), np.int32(int(s)), out)
        + (np.int32(rows), np.int32(cols)),
    )
    return out


def lut16(x, table):
    x = _d(x)
    out = cp.empty(x.shape, dtype=cp.int64)
    _k["lut16"](_grid(x.size), (_T,), (x, _d(table), out, np.int64(x.size)))
    return out


def mul_requant(a, b, m, s, bits):
    a, b = _d(a), _d(b)
    out = cp.empty(a.shape, dtype=cp.int64)
    _k["mul_requant"](
        _grid(a.size),
        (_T,),
        (a, b, out, np.int64(a.size), _i(m), np.int32(int(s)), np.int32(bits)),
    )
    return out


def add_sat(a, b, bits):
    a, b = _d(a), _d(b)
    out = cp.empty(a.shape, dtype=cp.int64)
    _k["add_sat"](_grid(a.size), (_T,), (a, b, out, np.int64(a.size), np.int32(bits)))
    return out


def argmax(x):
    x = _d(x)
    rows, cols = x.shape
    out = cp.empty(rows, dtype=cp.int64)
    _k["argmax"](_grid(rows), (_T,), (x, out, np.int32(rows), np.int32(cols)))
    return out
