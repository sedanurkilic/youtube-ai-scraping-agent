"""
Tests for embedding serialization and the final hash.

Always run : serialization / validation tests (no model needed).
Opt-in     : RUN_MODEL_TESTS=1 python -m unittest discover -s tests -v
             loads the pinned Nomic model and checks calibration trial T1.
"""

import hashlib
import math
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import Seda_Nur_Kilic_YouTube_AI as core  # noqa: E402

TOKEN_STRING = "395,703,470,622,561,950,712,634"

# Calibration trial T1 (prefix none, native output) measured on the author's machine
# with the pinned revision. A REGRESSION value for this setup - NOT the assessment's
# reference hash (4ae6acfb...), which was not reproduced.
T1_REGRESSION_MD5 = "041ccada537fbecbde13a3936c77ecb7"


class TestSerialization(unittest.TestCase):
    def test_ten_decimals_exactly(self):
        self.assertEqual(core.serialize_embedding([0.12345678912, -0.9876, 2.0]),
                         "0.1234567891,-0.9876000000,2.0000000000")

    def test_no_spaces_and_comma_count(self):
        text = core.serialize_embedding([0.5] * 768)
        self.assertNotIn(" ", text)
        self.assertEqual(text.count(","), 767)

    def test_every_item_has_ten_decimals(self):
        for item in core.serialize_embedding([i / 7 for i in range(768)]).split(","):
            self.assertRegex(item, r"^-?\d+\.\d{10}$")

    def test_negative_tiny_value_keeps_python_formatting(self):
        # The spec mandates f"{x:.10f}"; we do not post-process it.
        self.assertEqual(core.serialize_embedding([-1e-12]), "-0.0000000000")


class TestHashFromVector(unittest.TestCase):
    def test_matches_independent_computation(self):
        vector = [(i - 384) / 1000 for i in range(768)]
        expected = hashlib.md5(",".join(f"{x:.10f}" for x in vector).encode("utf-8")).hexdigest()
        self.assertEqual(core.embedding_hash_from_vector(vector), expected)

    def test_output_format(self):
        digest = core.embedding_hash_from_vector([0.0] * 768)
        self.assertRegex(digest, r"^[0-9a-f]{32}$")

    def test_wrong_length_rejected(self):
        for n in (0, 767, 769):
            with self.subTest(n=n):
                with self.assertRaises(ValueError):
                    core.embedding_hash_from_vector([0.1] * n)

    def test_nan_and_inf_rejected(self):
        for bad in (math.nan, math.inf, -math.inf):
            with self.subTest(bad=bad):
                vector = [0.1] * 768
                vector[10] = bad
                with self.assertRaises(ValueError):
                    core.embedding_hash_from_vector(vector)

    def test_any_value_change_changes_hash(self):
        a = [0.1] * 768
        b = list(a)
        b[767] = 0.1000000001
        self.assertNotEqual(core.embedding_hash_from_vector(a), core.embedding_hash_from_vector(b))


@unittest.skipUnless(os.environ.get("RUN_MODEL_TESTS") == "1",
                     "set RUN_MODEL_TESTS=1 to load the pinned Nomic model")
class TestModelRegression(unittest.TestCase):
    def test_vector_shape_and_finite(self):
        vector = core.embed_text(TOKEN_STRING)
        self.assertEqual(len(vector), 768)
        self.assertTrue(all(math.isfinite(v) for v in vector))

    def test_repeatable_on_same_machine(self):
        self.assertEqual(core.compute_embedding_hash(TOKEN_STRING),
                         core.compute_embedding_hash(TOKEN_STRING))

    def test_matches_calibration_trial_t1(self):
        self.assertEqual(core.compute_embedding_hash(TOKEN_STRING), T1_REGRESSION_MD5)


if __name__ == "__main__":
    unittest.main()
