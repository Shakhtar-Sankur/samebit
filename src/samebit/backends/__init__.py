"""Backends. Each module exposes the same integer ops as samebit.reference."""

import importlib

NAMES = ("reference", "numpy", "c", "torch", "mlx")


def load(name):
    if name == "reference":
        from .. import reference

        reference.name = "reference"
        return reference
    return importlib.import_module(f".{name}_backend", __name__)


def available():
    out = []
    for n in NAMES:
        try:
            load(n)
            out.append(n)
        except Exception:  # missing package or no C compiler
            pass
    return out
