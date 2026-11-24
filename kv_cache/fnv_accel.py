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
import hashlib
import os
import shutil
import struct
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

FNV_OFFSET_BASIS = 0xCBF29CE484222325
FNV_PRIME = 0x100000001B3
MASK_64 = 0xFFFFFFFFFFFFFFFF

# Fixed-width 64-bit type.  ctypes.c_ulong is only 32-bit on Windows and
# would silently truncate the 64-bit seed there; ctypes.c_uint64 is always 8 bytes.
_U64 = ctypes.c_uint64

_C_SOURCE = """
#include <stdint.h>
#include <stddef.h>

uint64_t modchip_fnv1a_64(const uint8_t *data, size_t n, uint64_t seed) {
    uint64_t h = seed;
    for (size_t i = 0; i < n; i++) h = (h ^ data[i]) * UINT64_C(1099511628211);
    return h;
}

/* Chained FNV-1a over fixed-size blocks of little-endian int32 tokens packed
 * into a byte buffer.  out[k] = FNV64(seed || block_0 || ... || block_k);
 * returns the block count.  Equivalent to calling modchip_fnv1a_64 on each
 * block's 4*block bytes with the previous block's hash as seed. */
size_t modchip_fnv_chain(const uint8_t *data, size_t nbytes,
                          size_t block, uint64_t seed, uint64_t *out) {
    size_t nblocks = nbytes / (block * 4), k = 0;
    uint64_t h = seed;
    for (size_t b = 0; b < nblocks; b++) {
        const uint8_t *p = data + b * block * 4;
        for (size_t i = 0; i < block * 4; i++) {
            h = (h ^ p[i]) * UINT64_C(1099511628211);
        }
        out[k++] = h;
    }
    return k;
}

/* Build tag.  The loader rejects a cached artifact whose tag does not match
 * the current source, so editing the C source always forces a rebuild. */
uint32_t modchip_fnv_version(void) { return MODCHIP_FNV_VERSION; }
"""

# Source-derived build tag: changing _C_SOURCE changes this digest, which
# changes the artifact filename, so a stale .so can never be picked up.
_BUILD_TAG = hashlib.sha256(_C_SOURCE.encode("ascii")).hexdigest()[:12]

# Cached build artifact, next to this file.  *.so is gitignored.
_LIB_NAME = f".fnv_accel_{_BUILD_TAG}.so"
_LIB_PATH = Path(__file__).resolve().parent / _LIB_NAME


def _fallback_dir() -> Path:
    """Per-user cache directory used when the source directory is read-only."""
    try:
        uid = os.getuid()  # not available on Windows
    except AttributeError:
        uid = "default"
    return Path(tempfile.gettempdir()) / f"modchip-fnv-{uid}"


def _candidate_paths() -> list:
    """Every location the cached artifact may live in, most preferred first."""
    return [_LIB_PATH, _fallback_dir() / _LIB_NAME]


def _fnv1a_python(data: bytes, seed: int = FNV_OFFSET_BASIS) -> int:
    h = seed
    for b in data:
        h = ((h ^ b) * FNV_PRIME) & MASK_64
    return h


def _bind(lib) -> tuple:
    fn = lib.modchip_fnv1a_64
    fn.argtypes = [ctypes.c_char_p, ctypes.c_size_t, _U64]
    fn.restype = _U64
    chain = lib.modchip_fnv_chain
    chain.argtypes = [ctypes.c_char_p, ctypes.c_size_t,
                      ctypes.c_size_t, _U64,
                      ctypes.POINTER(_U64)]
    chain.restype = ctypes.c_size_t
    ver = lib.modchip_fnv_version
    ver.argtypes = []
    ver.restype = ctypes.c_uint32
    return fn, chain, ver


def _probe(fn) -> bool:
    """Verify bit-identity against the reference implementation."""
    probe = bytes(range(256))
    try:
        return fn(probe, len(probe), FNV_OFFSET_BASIS) == _fnv1a_python(probe)
    except (OSError, ValueError, ctypes.ArgumentError):
        return False


def _load_cached() -> Optional[tuple]:
    """Load a previously compiled shared object, if present and valid.

    A cached artifact that fails the version check or the bit-identity probe
    is removed so that the caller rebuilds and replaces it instead of
    silently falling back to the slow Python path.
    """
    for path in _candidate_paths():
        try:
            if not path.exists():
                continue
            lib = ctypes.CDLL(str(path))
            fn = _bind(lib)
        except (OSError, ValueError, AttributeError, ctypes.ArgumentError):
            _discard_cached(path)
            continue
        try:
            if fn[2]() != _version_number():
                raise ValueError("cached artifact has a stale build tag")
        except (OSError, ValueError):
            _discard_cached(path)
            continue
        if not _probe(fn[0]):
            _discard_cached(path)
            continue
        return fn
    return None


def _discard_cached(path: Optional[Path] = None) -> None:
    """Best-effort removal of a bad cached artifact."""
    try:
        (path or _LIB_PATH).unlink()
    except OSError:
        pass


def _version_number() -> int:
    """Numeric build tag embedded in the shared object."""
    return int(_BUILD_TAG[:8], 16)


def _install(so: Path) -> Optional[Path]:
    """Install the compiled artifact at a persistent path, atomically.

    The file is written to a unique temporary name in the target directory and
    then moved into place with os.replace().  That is atomic within a
    filesystem, so a concurrent importer sees either the old or the new
    library but never a half-written one, and a library that is already
    mmap'd by another process is never overwritten in place.

    The preferred location is next to this source file.  If that directory is
    read-only (e.g. a system-wide install), a per-user directory under the
    system temp dir is used instead, so the artifact is still cached across
    processes.

    Returns the persistent path to dlopen, or None if no location is
    writable.
    """
    for target_dir in (_LIB_PATH.parent, _fallback_dir()):
        try:
            target_dir.mkdir(parents=True, exist_ok=True)
            fd, tmp_name = tempfile.mkstemp(prefix=".fnv_accel-",
                                            suffix=".so",
                                            dir=str(target_dir))
            os.close(fd)
            tmp = Path(tmp_name)
            try:
                shutil.copyfile(str(so), str(tmp))
                os.chmod(str(tmp), 0o755)
                final = target_dir / _LIB_NAME
                os.replace(str(tmp), str(final))
                return final
            finally:
                if tmp.exists():
                    try:
                        tmp.unlink()
                    except OSError:
                        pass
        except OSError:
            continue
    return None


def _compile_and_load() -> Optional[tuple]:
    """Compile the C hash and load it.

    The artifact is installed at a persistent path *before* it is dlopen'd:
    on Windows a loaded DLL cannot be deleted, so loading it from inside the
    temporary build directory would make the cleanup fail and silently
    disable the native path.
    """
    compiler = shutil.which("cc") or shutil.which("gcc") or shutil.which("clang")
    if not compiler:
        return None
    try:
        with tempfile.TemporaryDirectory(prefix="modchip-fnv-") as d:
            src = Path(d) / "fnv.c"
            so = Path(d) / "fnv.so"
            src.write_text(
                _C_SOURCE.replace("MODCHIP_FNV_VERSION",
                                  f"UINT32_C({_version_number()})"),
                encoding="ascii")
            res = subprocess.run(
                [compiler, "-O3", "-shared", "-fPIC", "-o", str(so), str(src)],
                capture_output=True, text=True, timeout=30,
            )
            if res.returncode != 0 or not so.exists():
                return None
            persistent = _install(so)
            if persistent is None:
                return None
            try:
                fn = _bind(ctypes.CDLL(str(persistent)))
            except (OSError, ValueError, AttributeError, ctypes.ArgumentError):
                return None
            if not _probe(fn[0]):
                return None
            return fn
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def _load_native() -> Optional[tuple]:
    """Load the C hash (cached build preferred), or return None if unavailable."""
    fn = _load_cached()
    if fn is None:
        fn = _compile_and_load()
    if fn is None:
        return None
    # Final bit-identity check before the fast path is trusted.
    if not _probe(fn[0]):
        return None
    return fn


_native = _load_native()


def fnv1a_64(data: bytes, seed: int = FNV_OFFSET_BASIS) -> int:
    """Bit-identical 64-bit FNV-1a, native when available, Python otherwise."""
    if _native is not None:
        if not isinstance(data, bytes):
            data = bytes(data)
        return _native[0](data, len(data), seed)
    return _fnv1a_python(data, seed)


def fnv_chain_blocks(tokens, block: int, seed: int = FNV_OFFSET_BASIS):
    """Chained FNV-1a over `block`-sized chunks of `tokens`.

    Returns the list of chained hashes, one per complete block.  The result is
    identical to calling `fnv1a_64` on each block's packed little-endian bytes
    with the previous block's hash as seed.  Returns None when the native
    implementation is unavailable, so callers can fall back to the per-block
    path.
    """
    if _native is None:
        return None
    n_blocks = len(tokens) // block
    if n_blocks == 0:
        return []
    try:
        raw = struct.pack(f"<{n_blocks * block}i", *tokens[:n_blocks * block])
        out = (_U64 * n_blocks)()
        _native[1](raw, len(raw), block, seed, out)
        return [out[i] for i in range(n_blocks)]
    except (ValueError, TypeError, OverflowError, MemoryError, struct.error):
        return None
