#!/usr/bin/env python3
"""Native-accelerated 64-bit FNV-1a with a pure-Python fallback.

HARP-Cache hashes every 32-token block on both the store and resolve paths
(fnv1a_64 accounts for ~93% of resolve_prefix cumulative time).  The byte loop
is strictly sequential, so SIMD cannot help, but a single C pass is ~16x
faster than the Python loop.  This module compiles a tiny shared object once
per process and falls back to Python whenever a compiler is unavailable.

The hash is bit-identical to the reference implementation in harp_cache.py:
    h = seed; for b in data: h = ((h ^ b) * 0x100000001b3) & 0xffffffffffffffff
"""

import ctypes
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

FNV_OFFSET_BASIS = 0xCBF29CE484222325
FNV_PRIME = 0x100000001B3
MASK_64 = 0xFFFFFFFFFFFFFFFF

_C_SOURCE = """
#include <stdint.h>
#include <stddef.h>
uint64_t modchip_fnv1a_64(const uint8_t *data, size_t n, uint64_t seed) {
    uint64_t h = seed;
    for (size_t i = 0; i < n; i++) h = (h ^ data[i]) * UINT64_C(1099511628211);
    return h;
}
"""

# Cached build artifact, next to this file. *.so is gitignored.
_LIB_PATH = Path(__file__).resolve().parent / ".fnv_accel.so"


def _fnv1a_python(data: bytes, seed: int = FNV_OFFSET_BASIS) -> int:
    h = seed
    for b in data:
        h = ((h ^ b) * FNV_PRIME) & MASK_64
    return h


def _bind(lib):
    fn = lib.modchip_fnv1a_64
    fn.argtypes = [ctypes.c_char_p, ctypes.c_size_t, ctypes.c_uint64]
    fn.restype = ctypes.c_uint64
    return fn


def _load_cached():
    """Load a previously compiled shared object, if present."""
    try:
        if _LIB_PATH.exists():
            return _bind(ctypes.CDLL(str(_LIB_PATH)))
    except OSError:
        return None
    return None


def _compile_and_load():
    """Compile the C hash into the cache directory and load it."""
    compiler = shutil.which("cc") or shutil.which("gcc") or shutil.which("clang")
    if not compiler:
        return None
    try:
        with tempfile.TemporaryDirectory(prefix="modchip-fnv-", dir="/tmp") as d:
            src = Path(d) / "fnv.c"
            so = Path(d) / "fnv.so"
            src.write_text(_C_SOURCE, encoding="ascii")
            res = subprocess.run(
                [compiler, "-O3", "-shared", "-fPIC", "-o", str(so), str(src)],
                capture_output=True, text=True, timeout=30,
            )
            if res.returncode != 0 or not so.exists():
                return None
            fn = _bind(ctypes.CDLL(str(so)))
            # Persist the artifact so the next process starts without a compiler.
            try:
                shutil.copyfile(str(so), str(_LIB_PATH))
            except OSError:
                pass
            return fn
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def _load_native():
    """Load the C hash (cached build preferred), or return None if unavailable."""
    fn = _load_cached()
    if fn is None:
        fn = _compile_and_load()
    if fn is None:
        return None
    # Verify bit-identity against the reference implementation before use.
    probe = bytes(range(256))
    try:
        if fn(probe, len(probe), FNV_OFFSET_BASIS) != _fnv1a_python(probe):
            return None
    except (OSError, ValueError):
        return None
    return fn


_native = _load_native()


def fnv1a_64(data: bytes, seed: int = FNV_OFFSET_BASIS) -> int:
    """Bit-identical 64-bit FNV-1a, native when available, Python otherwise."""
    if _native is not None:
        return _native(data, len(data), seed)
    return _fnv1a_python(data, seed)
