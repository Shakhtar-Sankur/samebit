"""A fake CuPy for testing samebit's CUDA backend without a GPU: arrays are
NumPy arrays, and RawModule compiles the CUDA C source as C++ with a prelude
that emulates blockIdx/threadIdx, then runs every block and thread in order
(and, for matmul split-K, in shuffled block order)."""

import ctypes
import hashlib
import os
import random
import re
import subprocess
import tempfile

import numpy as np

int8, int32, int64, uint8 = np.int8, np.int32, np.int64, np.uint8


def asarray(x, dtype=None):
    return np.asarray(x, dtype=dtype)


def ascontiguousarray(x, dtype=None):
    return np.ascontiguousarray(x, dtype=dtype)


def zeros(shape, dtype=None):
    return np.zeros(shape, dtype=dtype)


def empty(shape, dtype=None):
    return np.empty(shape, dtype=dtype)


def asnumpy(x):
    return np.asarray(x)


class _RT:
    @staticmethod
    def getDeviceCount():
        return 1

    @staticmethod
    def getDeviceProperties(i):
        return {"name": b"CPU emulation of CUDA"}


class cuda:
    runtime = _RT


PRELUDE = r"""
#include <algorithm>
#include <cstdint>
#include <vector>
#include <cstdlib>
struct D3 { unsigned x, y, z; };
static D3 blockIdx, threadIdx, blockDim;
#define __device__ static inline
using std::min;
static inline int atomicAdd(int* p, int v) { int o = *p; *p = o + v; return o; }
"""


def _ctype(decl):
    d = decl.strip()
    if "*" in d:
        return ctypes.c_void_p
    if d.startswith("long long") or d.startswith("i64"):
        return ctypes.c_longlong
    return ctypes.c_int


class RawModule:
    def __init__(self, code, options=()):
        sigs = re.findall(r'extern "C" __global__ void (\w+)\(([^)]*)\)', code)
        body = code
        wrappers = []
        for name, params in sigs:
            body = body.replace(f'extern "C" __global__ void {name}(', f"static void k_{name}(", 1)
            names = [p.strip().split()[-1].lstrip("*") for p in params.split(",")]
            wrappers.append(f"""
extern "C" void emu_{name}(int gx, int gy, int gz, int bx, unsigned seed, {params}) {{
  blockDim = {{(unsigned)bx, 1, 1}};
  std::vector<unsigned> zs(gz); for (int emu_i = 0; emu_i < gz; emu_i++) zs[emu_i] = emu_i;
  srand(seed); for (int emu_i = gz - 1; emu_i > 0; emu_i--) std::swap(zs[emu_i], zs[rand() % (emu_i + 1)]);
  for (int emu_zi = 0; emu_zi < gz; emu_zi++) for (int emu_y = 0; emu_y < gy; emu_y++) for (int emu_x = 0; emu_x < gx; emu_x++)
    for (int emu_t = 0; emu_t < bx; emu_t++) {{
      blockIdx = {{(unsigned)emu_x, (unsigned)emu_y, zs[emu_zi]}}; threadIdx = {{(unsigned)emu_t, 0, 0}};
      k_{name}({", ".join(names)});
    }}
}}""")
            setattr(self, "_sig_" + name, [_ctype(p) for p in params.split(",")])
        src = PRELUDE + body + "\n".join(wrappers)
        tag = hashlib.sha256(src.encode()).hexdigest()[:12]
        d = tempfile.mkdtemp()
        cpp, so = os.path.join(d, "k.cpp"), os.path.join(d, f"k{tag}.so")
        open(cpp, "w").write(src)
        subprocess.run(
            [
                "g++",
                "-O1",
                "-std=c++14",
                "-fPIC",
                "-shared",
                "-Wall",
                "-Wno-unused-function",
                cpp,
                "-o",
                so,
            ],
            check=True,
        )
        self.lib = ctypes.CDLL(so)

    def get_function(self, name):
        fn = getattr(self.lib, "emu_" + name)
        sig = getattr(self, "_sig_" + name)

        def launch(grid, block, args):
            g = tuple(grid) + (1,) * (3 - len(grid))
            cargs = []
            for a, t in zip(args, sig):
                if t is ctypes.c_void_p:
                    assert a.flags["C_CONTIGUOUS"]
                    cargs.append(a.ctypes.data_as(ctypes.c_void_p))
                else:
                    cargs.append(t(int(a)))
            fn(
                int(g[0]),
                int(g[1]),
                int(g[2]),
                int(block[0]),
                ctypes.c_uint(random.randrange(1 << 30)),
                *cargs,
            )

        return launch
