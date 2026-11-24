"""Tests for the native-accelerated FNV-1a hash used by HARP-Cache.

The hash is on the hot path of every store/resolve (93% of resolve_prefix
cumulative time).  The native implementation must be bit-identical to the
reference Python loop for every input, and the fallback must work when no
compiler is available.
"""

import random
import struct
import unittest

from kv_cache import fnv_accel
from kv_cache.harp_cache import FNV_OFFSET_BASIS, fnv1a_64, hash_token_block


class NativeFNVTest(unittest.TestCase):
    def test_native_matches_python_reference_on_random_buffers(self):
        """The C implementation must be bit-identical for arbitrary bytes+seeds."""
        if fnv_accel._native is None:
            self.skipTest("no C compiler available on this host")
        rng = random.Random(0xC0FFEE)
        for _ in range(500):
            data = bytes(rng.getrandbits(8) for _ in range(rng.randint(0, 512)))
            seed = rng.getrandbits(64)
            self.assertEqual(
                fnv_accel._native[0](data, len(data), seed),
                fnv_accel._fnv1a_python(data, seed),
            )

    def test_harp_fnv_is_bit_identical_to_reference(self):
        """harp_cache.fnv1a_64 delegates without changing hash semantics."""
        rng = random.Random(7)
        for _ in range(200):
            data = bytes(rng.getrandbits(8) for _ in range(rng.randint(0, 128)))
            seed = rng.getrandbits(64)
            self.assertEqual(fnv1a_64(data, seed), fnv_accel._fnv1a_python(data, seed))

    def test_empty_buffer_and_known_vector(self):
        self.assertEqual(fnv1a_64(b""), FNV_OFFSET_BASIS)
        # FNV-1a 64-bit of ASCII "a" is a published test vector.
        self.assertEqual(fnv1a_64(b"a"), 0xAF63DC4C8601EC8C)

    def test_hash_token_block_chain_unchanged(self):
        """Causal chaining via prev_hash must produce identical results."""
        block = [1000 + i for i in range(32)]
        h0 = hash_token_block(block)
        h1 = hash_token_block(block, prev_hash=h0)
        self.assertNotEqual(h0, h1)
        self.assertEqual(h0, fnv1a_64(__import__("struct").pack("<32i", *block)))

    def test_batched_chain_bit_identical_to_per_block(self):
        """The batched native chain must equal sequential hash_token_block calls."""
        if fnv_accel._native is None:
            self.skipTest("no C compiler available on this host")
        rng = random.Random(0x5EED)
        for _ in range(100):
            nblocks = rng.randint(1, 40)
            tokens = [rng.randint(-(2**31), 2**31 - 1) for _ in range(32 * nblocks)]
            seed = rng.getrandbits(64)
            expected = []
            h = seed
            for b in range(nblocks):
                h = fnv1a_64(struct.pack("<32i", *tokens[b*32:(b+1)*32]), h)
                expected.append(h)
            got = fnv_accel.fnv_chain_blocks(tokens, 32, seed)
            self.assertEqual(got, expected)

    def test_batched_chain_handles_empty_and_partial_tail(self):
        if fnv_accel._native is None:
            self.skipTest("no C compiler available on this host")
        self.assertEqual(fnv_accel.fnv_chain_blocks([], 32), [])
        self.assertEqual(len(fnv_accel.fnv_chain_blocks([1, 2, 3] * 20, 32)), 1)

    def test_batched_chain_disabled_when_hash_token_block_is_patched(self):
        """HARP collision tests monkeypatch hash_token_block; the batched path
        must fall back to per-block hashing in that case, or the patch is ignored."""
        import kv_cache.harp_cache as harp_mod

        if harp_mod._ORIGINAL_HASH_TOKEN_BLOCK is not harp_mod.hash_token_block:
            self.skipTest("hash_token_block already patched by another test")
        original = harp_mod.hash_token_block
        try:
            self.assertIsNotNone(harp_mod._hash_chain([1] * 64, 32, FNV_OFFSET_BASIS))
            harp_mod.hash_token_block = lambda tokens, prev_hash=FNV_OFFSET_BASIS: 999
            self.assertIsNone(harp_mod._hash_chain([1] * 64, 32, FNV_OFFSET_BASIS))
        finally:
            harp_mod.hash_token_block = original

    def test_native_faster_than_python(self):
        """Guard against accidentally falling back to the slow Python loop."""
        if fnv_accel._native is None:
            self.skipTest("no C compiler available on this host")
        import time

        data = bytes(range(256)) * 4
        iterations = 2000
        t0 = time.perf_counter_ns()
        for _ in range(iterations):
            fnv_accel._fnv1a_python(data)
        t_py = time.perf_counter_ns() - t0
        t0 = time.perf_counter_ns()
        for _ in range(iterations):
            fnv1a_64(data)
        t_native = time.perf_counter_ns() - t0
        self.assertLess(t_native, t_py / 3,
                        f"native too slow: {t_native} vs Python {t_py}")

    def test_embedded_null_bytes_hash_full_buffer(self):
        """c_char_p must not truncate the buffer at an embedded NUL."""
        rng = random.Random(0xFEED)
        for _ in range(50):
            data = bytes(rng.getrandbits(8) for _ in range(rng.randint(1, 256)))
            data = data[:len(data)//2] + b"\x00" + data[len(data)//2:]
            seed = rng.getrandbits(64)
            self.assertEqual(fnv1a_64(data, seed), fnv_accel._fnv1a_python(data, seed))

    def test_bytes_like_input(self):
        """bytearray/memoryview should work just like bytes."""
        data = b"a" * 100
        seed = 0xDEADBEEF
        expected = fnv1a_64(data, seed)
        self.assertEqual(fnv1a_64(bytearray(data), seed), expected)
        self.assertEqual(fnv1a_64(memoryview(data), seed), expected)

    def test_no_compiler_falls_back_to_python(self):
        """When no compiler is found, fnv1a_64 must still be bit-identical."""
        from unittest import mock

        with mock.patch.object(fnv_accel.shutil, "which", return_value=None):
            with mock.patch.object(fnv_accel, "_native", None):
                self.assertEqual(fnv1a_64(b"fallback", 12345),
                                 fnv_accel._fnv1a_python(b"fallback", 12345))
                raw = FNV_OFFSET_BASIS.to_bytes(8, "little")
                self.assertEqual(fnv1a_64(raw), fnv_accel._fnv1a_python(raw))

    def test_corrupt_cached_artifact_is_discarded(self):
        """A cached .so that fails the probe must be removed, not trusted."""
        import tempfile
        from pathlib import Path
        from unittest import mock

        with tempfile.TemporaryDirectory() as d:
            bogus = Path(d) / "bogus.so"
            bogus.write_bytes(b"not a shared object")
            with mock.patch.object(fnv_accel, "_candidate_paths", return_value=[bogus]):
                self.assertIsNone(fnv_accel._load_cached())
            # _load_cached() removes the unusable artifact so that the next
            # import rebuilds and replaces it.
            self.assertFalse(bogus.exists())

    def test_stale_version_artifact_is_discarded(self):
        """A cached .so built from older source must not be used."""
        from unittest import mock

        if fnv_accel._native is None:
            self.skipTest("no C compiler available on this host")
        if not fnv_accel._LIB_PATH.exists():
            self.skipTest("no cached artifact on this host")

        with mock.patch.object(fnv_accel, "_version_number", return_value=0), \
             mock.patch.object(fnv_accel, "_discard_cached") as discard:
            self.assertIsNone(fnv_accel._load_cached())
            discard.assert_called_once()

    def test_cached_artifact_is_installed_atomically(self):
        """The installed artifact must be a complete, loadable shared object."""
        import ctypes
        from unittest import mock

        if fnv_accel._native is None:
            self.skipTest("no C compiler available on this host")
        if not fnv_accel._LIB_PATH.exists():
            self.skipTest("no cached artifact on this host")

        # No stray temporary files must be left behind by the install.
        leftovers = list(fnv_accel._LIB_PATH.parent.glob(".fnv_accel-*.so"))
        self.assertEqual(leftovers, [],
                         f"stray temp artifacts: {leftovers}")

        # And the artifact on disk must load and hash correctly.
        lib = ctypes.CDLL(str(fnv_accel._LIB_PATH))
        fn, chain, ver = fnv_accel._bind(lib)
        self.assertEqual(ver(), fnv_accel._version_number())
        self.assertTrue(fnv_accel._probe(fn))


if __name__ == "__main__":
    unittest.main()
