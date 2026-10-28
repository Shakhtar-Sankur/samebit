"""CUDA backend: the integer kernels as CUDA C, compiled at run time by CuPy
(NVRTC, so no CUDA toolkit is needed beyond the driver).

The matmul kernel splits every dot product into SAMEBIT_CUDA_SPLITK chunks
that run in different thread blocks and are combined with atomicAdd, so the
order of the final additions changes from run to run. Integer addition makes
that harmless; with floats it would change the result.
"""

import os

import cupy as cp
import numpy as np

name = "cuda"
SPLITK = int(os.environ.get("SAMEBIT_CUDA_SPLITK", "8"))

if cp.cuda.runtime.getDeviceCount() < 1:  # make the backend unavailable, not broken
    raise ImportError("no CUDA device")

_SRC = r"""
typedef long long i64;

__device__ i64 floor_shift(i64 v, int s) {
  if (s == 0) return v;
  if (v >= 0) return v >> s;
  return -(i64)((((unsigned long long)(-(v + 1))) >> s) + 1);
}
__device__ i64 rsr(i64 v, int s) { return s == 0 ? v : floor_shift(v + (1LL << (s - 1)), s); }
__device__ signed char sat8(i64 v) { return (signed char)(v < -128 ? -128 : v > 127 ? 127 : v); }
__device__ i64 isqrt32(i64 n) {
  i64 res = 0, bit = 1LL << 30;
  for (int i = 0; i < 16; i++) {
    if (n >= res + bit) { n -= res + bit; res = (res >> 1) + bit; } else { res >>= 1; }
    bit >>= 2;
  }
  return res;
}
__device__ long long gid() { return (long long)blockIdx.x * blockDim.x + threadIdx.x; }

// out must be zeroed. blockIdx.y = row, blockIdx.z = k-chunk.
extern "C" __global__ void matmul(const int* a, const int* b, int* out, int m, int k, int n, int chunk) {
  int j = blockIdx.x * blockDim.x + threadIdx.x, i = blockIdx.y, t0 = blockIdx.z * chunk;
  if (j >= n || t0 >= k) return;
  int t1 = min(k, t0 + chunk), acc = 0;
  for (int t = t1 - 1; t >= t0; --t) acc += a[(i64)i * k + t] * b[(i64)t * n + j];
  atomicAdd(&out[(i64)i * n + j], acc);
}

extern "C" __global__ void requant(const i64* x, const i64* m, const i64* s, signed char* out,
                                   long long total, int cols, int pc) {
  long long idx = gid(); if (idx >= total) return;
  int c = pc ? (int)(idx % cols) : 0;
  out[idx] = sat8(rsr(x[idx] * m[c], (int)s[c]));
}

extern "C" __global__ void rmsnorm(const i64* x, const i64* g, const i64* m, const i64* s,
                                   signed char* out, int rows, int d, int pc) {
  int r = (int)gid(); if (r >= rows) return;
  const i64* row = x + (i64)r * d;
  i64 sq = 0;
  for (int j = d - 1; j >= 0; --j) sq += row[j] * row[j];
  i64 rr = isqrt32((sq << 16) / d); if (rr < 1) rr = 1;
  i64 inv = (1LL << 30) / rr;
  for (int j = 0; j < d; j++) {
    int c = pc ? j : 0;
    i64 v = rsr(row[j] * inv, 14) * g[c];
    out[(i64)r * d + j] = sat8(rsr(v * m[c], (int)s[c]));
  }
}

// one thread per (t, h, j < d/2)
extern "C" __global__ void rope(const i64* x, const i64* cs, const i64* sn, signed char* out,
                                int t, int h, int d) {
  int half = d / 2; long long idx = gid(); if (idx >= (long long)t * h * half) return;
  int j = (int)(idx % half), th = (int)(idx / half), ti = th / h;
  const i64* v = x + (i64)th * d; signed char* o = out + (i64)th * d;
  i64 c = cs[(i64)ti * half + j], s = sn[(i64)ti * half + j];
  o[j] = sat8(rsr(v[j] * c - v[j + half] * s, 14));
  o[j + half] = sat8(rsr(v[j] * s + v[j + half] * c, 14));
}

extern "C" __global__ void softmax(const i64* sc, const unsigned char* ok, const i64* lut,
                                   int lut_len, int shift, int* out, int rows, int cols) {
  int r = (int)gid(); if (r >= rows) return;
  const i64* row = sc + (i64)r * cols; const unsigned char* v = ok + (i64)r * cols;
  i64 mx = -9223372036854775807LL - 1, total = 0;
  for (int j = 0; j < cols; j++) if (v[j] && row[j] > mx) mx = row[j];
  for (int j = 0; j < cols; j++) if (v[j]) { i64 dd = rsr(mx - row[j], shift); total += lut[dd < lut_len - 1 ? dd : lut_len - 1]; }
  for (int j = 0; j < cols; j++) {
    i64 e = 0;
    if (v[j]) { i64 dd = rsr(mx - row[j], shift); e = lut[dd < lut_len - 1 ? dd : lut_len - 1]; }
    i64 p = (e * 512 + total) / (2 * total);
    out[(i64)r * cols + j] = (int)(p < 255 ? p : 255);
  }
}

extern "C" __global__ void lut8(const i64* x, const signed char* table, signed char* out, long long n) {
  long long i = gid(); if (i < n) out[i] = table[x[i] + 128];
}

extern "C" __global__ void mul_requant(const i64* a, const i64* b, const i64* m, const i64* s,
                                       signed char* out, long long total, int cols, int pc) {
  long long i = gid(); if (i >= total) return;
  int c = pc ? (int)(i % cols) : 0;
  out[i] = sat8(rsr(a[i] * b[i] * m[c], (int)s[c]));
}

extern "C" __global__ void add_requant(const i64* a, const i64* b, i64 ma, i64 mb, int s,
                                       signed char* out, long long n) {
  long long i = gid(); if (i < n) out[i] = sat8(rsr(a[i] * ma + b[i] * mb, s));
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
    for n in (
        "matmul",
        "requant",
        "rmsnorm",
        "rope",
        "softmax",
        "lut8",
        "mul_requant",
        "add_requant",
        "argmax",
    )
}
_T = 256


def _grid(n):
    return ((int(n) + _T - 1) // _T,)


def _g(x, dtype=cp.int64):
    return cp.ascontiguousarray(cp.asarray(np.asarray(x), dtype=dtype))


def _channels(p, n):
    p = _g(p).reshape(-1)
    return p, int(p.size == n and n > 1)


def device_name():
    return cp.cuda.runtime.getDeviceProperties(0)["name"].decode()


def matmul(a, b):
    a, b = _g(a, cp.int32), _g(b, cp.int32)
    m, k = a.shape
    n = b.shape[1]
    out = cp.zeros((m, n), dtype=cp.int32)
    parts = max(1, min(SPLITK, k))
    chunk = (k + parts - 1) // parts
    _k["matmul"](
        ((n + 127) // 128, m, parts),
        (128,),
        (a, b, out, np.int32(m), np.int32(k), np.int32(n), np.int32(chunk)),
    )
    return cp.asnumpy(out)


def requantize(x, m, s):
    x = _g(x)
    cols = x.shape[-1]
    mv, pc = _channels(m, cols)
    sv, _ = _channels(s, cols)
    out = cp.empty(x.shape, dtype=cp.int8)
    _k["requant"](
        _grid(x.size), (_T,), (x, mv, sv, out, np.int64(x.size), np.int32(cols), np.int32(pc))
    )
    return cp.asnumpy(out)


def rmsnorm(x, g, m, s):
    x = _g(x)
    d = x.shape[-1]
    gv, pc = _channels(g, d)
    mv = _g(np.broadcast_to(np.asarray(m), gv.shape))
    sv = _g(np.broadcast_to(np.asarray(s), gv.shape))
    rows = x.size // d
    out = cp.empty(x.shape, dtype=cp.int8)
    _k["rmsnorm"](
        _grid(rows), (_T,), (x, gv, mv, sv, out, np.int32(rows), np.int32(d), np.int32(pc))
    )
    return cp.asnumpy(out)


def rope(x, cos, sin):
    x = _g(x)
    t, h, d = x.shape
    out = cp.empty(x.shape, dtype=cp.int8)
    _k["rope"](
        _grid(t * h * (d // 2)),
        (_T,),
        (x, _g(cos), _g(sin), out, np.int32(t), np.int32(h), np.int32(d)),
    )
    return cp.asnumpy(out)


def softmax(scores, valid, shift, exp_lut):
    sc = _g(scores)
    ok = _g(valid, cp.uint8)
    lut_v = _g(exp_lut)
    rows, cols = sc.shape
    out = cp.empty(sc.shape, dtype=cp.int32)
    _k["softmax"](
        _grid(rows),
        (_T,),
        (sc, ok, lut_v, np.int32(lut_v.size), np.int32(shift), out, np.int32(rows), np.int32(cols)),
    )
    return cp.asnumpy(out)


def lut(x, table):
    x = _g(x)
    out = cp.empty(x.shape, dtype=cp.int8)
    _k["lut8"](_grid(x.size), (_T,), (x, _g(table, cp.int8), out, np.int64(x.size)))
    return cp.asnumpy(out)


def mul_requant(a, b, m, s):
    a, b = _g(a), _g(b)
    cols = a.shape[-1]
    mv, pc = _channels(m, cols)
    sv, _ = _channels(s, cols)
    out = cp.empty(a.shape, dtype=cp.int8)
    _k["mul_requant"](
        _grid(a.size), (_T,), (a, b, mv, sv, out, np.int64(a.size), np.int32(cols), np.int32(pc))
    )
    return cp.asnumpy(out)


def add_requant(a, b, ma, mb, s):
    a, b = _g(a), _g(b)
    out = cp.empty(a.shape, dtype=cp.int8)
    _k["add_requant"](
        _grid(a.size), (_T,), (a, b, np.int64(ma), np.int64(mb), np.int32(s), out, np.int64(a.size))
    )
    return cp.asnumpy(out)


def argmax(x):
    x = _g(x)
    rows, cols = x.shape
    out = cp.empty(rows, dtype=cp.int64)
    _k["argmax"](_grid(rows), (_T,), (x, out, np.int32(rows), np.int32(cols)))
    return cp.asnumpy(out)
