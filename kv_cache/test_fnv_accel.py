"""Tests for the native-accelerated FNV-1a hash used by HARP-Cache.

The hash is on the hot path of every store/resolve (93% of resolve_prefix
cumulative time).  The native implementation must be bit-identical to the
reference Python loop for every input, and the fallback must work when no
compiler is available.
"""

import random
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
                fnv_accel._native(data, len(data), seed),
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


if __name__ == "__main__":
    unittest.main()
