"""C backend: portable C99 kernels built on first use and loaded with ctypes.

SAMEBIT_C_ORDER picks the matmul accumulation order (0 ascending, 1 descending,
2 blocked back to front). Every order must give identical results.
"""

import ctypes
import hashlib
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np

name = "c"
ORDER = int(os.environ.get("SAMEBIT_C_ORDER", "0"))
_SRC = Path(__file__).resolve().parent.parent / "csrc" / "samebit.c"


def _build():
    src = _SRC.read_bytes()
    tag = hashlib.sha256(src).hexdigest()[:16]
    cache = Path(os.environ.get("SAMEBIT_CACHE", Path(tempfile.gettempdir()) / "samebit"))
    cache.mkdir(parents=True, exist_ok=True)
    lib = cache / f"libsamebit-{tag}.so"
    if not lib.exists():
        cc = os.environ.get("CC", "cc")
        base = [cc, "-O2", "-std=c99", "-fPIC", "-shared", str(_SRC), "-o", str(lib)]
        # OpenMP is optional; fall back to a single-threaded build.
        if subprocess.run(base + ["-fopenmp"], check=False, capture_output=True).returncode != 0:
            subprocess.run(base, check=True, capture_output=True)
    return ctypes.CDLL(str(lib))


_lib = _build()


def _p(arr):
    return arr.ctypes.data_as(ctypes.c_void_p)


def _i64(x):
    return np.ascontiguousarray(np.asarray(x), dtype=np.int64)


def _channels(p, n):
    p = _i64(p).reshape(-1)
    return p, int(p.size == n and n > 1)


def matmul(a, b):
    a = np.ascontiguousarray(a, dtype=np.int32)
    b = np.ascontiguousarray(b, dtype=np.int32)
    m, k = a.shape
    n = b.shape[1]
    out = np.empty((m, n), dtype=np.int32)
    _lib.sb_matmul(_p(a), _p(b), _p(out), m, k, n, ORDER)
    return out


def requantize(x, m, s):
    x = _i64(x)
    cols = x.shape[-1]
    mv, pc = _channels(m, cols)
    sv, _ = _channels(s, cols)
    out = np.empty(x.shape, dtype=np.int8)
    _lib.sb_requantize(_p(x), _p(mv), _p(sv), _p(out), x.size // cols, cols, pc)
    return out


def rmsnorm(x, g, m, s):
    x = _i64(x)
    d = x.shape[-1]
    gv, pc = _channels(g, d)
    mv, sv = _i64(np.broadcast_to(m, gv.shape)), _i64(np.broadcast_to(s, gv.shape))
    out = np.empty(x.shape, dtype=np.int8)
    _lib.sb_rmsnorm(_p(x), _p(gv), _p(mv), _p(sv), _p(out), x.size // d, d, pc)
    return out


def rope(x, cos, sin):
    x = _i64(x)
    t, h, d = x.shape
    c, s = _i64(cos), _i64(sin)
    out = np.empty(x.shape, dtype=np.int8)
    _lib.sb_rope(_p(x), _p(c), _p(s), _p(out), t, h, d)
    return out


def softmax(scores, valid, shift, exp_lut):
    sc = _i64(scores)
    ok = np.ascontiguousarray(valid, dtype=np.uint8)
    lut_v = _i64(exp_lut)
    rows, cols = sc.shape
    out = np.empty(sc.shape, dtype=np.int32)
    _lib.sb_softmax(_p(sc), _p(ok), _p(lut_v), lut_v.size, int(shift), _p(out), rows, cols)
    return out


def lut(x, table):
    x = _i64(x)
    t = np.ascontiguousarray(table, dtype=np.int8)
    out = np.empty(x.shape, dtype=np.int8)
    _lib.sb_lut(_p(x), _p(t), _p(out), ctypes.c_int64(x.size))
    return out


def mul_requant(a, b, m, s):
    a, b = _i64(a), _i64(b)
    cols = a.shape[-1]
    mv, pc = _channels(m, cols)
    sv, _ = _channels(s, cols)
    out = np.empty(a.shape, dtype=np.int8)
    _lib.sb_mul_requant(_p(a), _p(b), _p(mv), _p(sv), _p(out), a.size // cols, cols, pc)
    return out


def add_requant(a, b, ma, mb, s):
    a, b = _i64(a), _i64(b)
    out = np.empty(a.shape, dtype=np.int8)
    _lib.sb_add_requant(
        _p(a),
        _p(b),
        ctypes.c_int64(ma),
        ctypes.c_int64(mb),
        int(s),
        _p(out),
        ctypes.c_int64(a.size),
    )
    return out


def argmax(x):
    x = _i64(x)
    out = np.empty(x.shape[0], dtype=np.int64)
    _lib.sb_argmax(_p(x), _p(out), x.shape[0], x.shape[1])
    return out
