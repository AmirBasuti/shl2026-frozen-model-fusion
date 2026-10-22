"""Synthetic tests: validation and arithmetic, never benchmark performance."""
import copy
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from fusion import CLASS_NAMES, WEIGHTS, fuse_probabilities

ROOT = Path(__file__).resolve().parents[1]


def fixture():
    return {
        "NuTime": [[1, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0, 0]],
        "Mantis": [[0, 1, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0, 0]],
        "TiRex": [[0, 0, 1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0, 0]],
    }


class FusionTests(unittest.TestCase):
    def test_exact_published_weights_and_order(self):
        self.assertEqual(WEIGHTS, {"NuTime": 0.54, "Mantis": 0.29, "TiRex": 0.17})
        self.assertEqual(CLASS_NAMES, ["Still", "Walk", "Run", "Bike", "Car", "Bus", "Train", "Subway"])

    def test_weighted_probabilities_and_labels(self):
        inputs = fixture()
        original = copy.deepcopy(inputs)
        result = fuse_probabilities(inputs)
        self.assertEqual(result["probabilities"], [[0.54, 0.29, 0.17, 0, 0, 0, 0, 0],
                                                   [0, 0, 0, 0.54, 0.29, 0.17, 0, 0]])
        self.assertEqual(result["labels"], ["Still", "Bike"])
        self.assertEqual(result["label_indices"], [0, 3])
        self.assertEqual(inputs, original)

    def test_soft_probabilities_are_fused_without_hard_voting(self):
        inputs = {name: [[0.6, 0.4, 0, 0, 0, 0, 0, 0]] for name in WEIGHTS}
        inputs["NuTime"][0] = [0.1, 0.9, 0, 0, 0, 0, 0, 0]
        result = fuse_probabilities(inputs)
        self.assertAlmostEqual(result["probabilities"][0][0], 0.33)
        self.assertAlmostEqual(result["probabilities"][0][1], 0.67)
        self.assertEqual(result["labels"], ["Walk"])

    def test_ties_choose_first_class(self):
        result = fuse_probabilities({name: [[0.5, 0.5, 0, 0, 0, 0, 0, 0]] for name in WEIGHTS})
        self.assertEqual(result["labels"], ["Still"])

    def test_missing_or_extra_model_is_rejected(self):
        for name in WEIGHTS:
            inputs = fixture()
            del inputs[name]
            with self.assertRaises(ValueError):
                fuse_probabilities(inputs)
        inputs = fixture()
        inputs["UniShape"] = inputs["NuTime"]
        with self.assertRaises(ValueError):
            fuse_probabilities(inputs)

    def test_invalid_dimensions_and_empty_inputs_are_rejected(self):
        for value in ([], [0.125] * 8, [[1, 0]], [[1] * 9], "matrix", None):
            inputs = fixture()
            inputs["NuTime"] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                fuse_probabilities(inputs)
        with self.assertRaises(ValueError):
            fuse_probabilities([])

    def test_different_row_counts_are_rejected(self):
        inputs = fixture()
        inputs["TiRex"] = inputs["TiRex"][:1]
        with self.assertRaises(ValueError):
            fuse_probabilities(inputs)

    def test_non_numeric_non_finite_and_out_of_range_values_are_rejected(self):
        for value in (True, False, "1", None, math.nan, math.inf, -math.inf, -0.01, 1.01, 10**1000):
            inputs = fixture()
            inputs["NuTime"][0][0] = value
            with self.subTest(value=repr(value)[:30]), self.assertRaises(ValueError):
                fuse_probabilities(inputs)

    def test_non_normalized_rows_are_rejected(self):
        inputs = fixture()
        inputs["Mantis"][0] = [0.1] * 8
        with self.assertRaisesRegex(ValueError, "sum"):
            fuse_probabilities(inputs)

    def test_documented_rounding_tolerance_without_renormalization(self):
        inputs = {name: [[0.9999995, 0, 0, 0, 0, 0, 0, 0]] for name in WEIGHTS}
        self.assertAlmostEqual(fuse_probabilities(inputs)["probabilities"][0][0], 0.9999995)


class CliTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.input = self.root / "input.json"
        self.output = self.root / "fused.json"
        self.input.write_text(json.dumps(fixture()), encoding="utf-8")

    def cli(self, *extra, code=0):
        result = subprocess.run([sys.executable, "-B", str(ROOT / "fusion.py"),
                                 "--input", str(self.input), "--output", str(self.output), *extra],
                                capture_output=True, text=True, encoding="utf-8")
        self.assertEqual(result.returncode, code, result.stdout + result.stderr)
        return result

    def test_shipped_synthetic_example(self):
        self.input.write_bytes((ROOT / "examples/synthetic_probabilities.json").read_bytes())
        self.cli()
        result = json.loads(self.output.read_text())
        self.assertEqual(result["labels"], ["Still", "Bike"])
        self.assertEqual(result["class_names"], CLASS_NAMES)

    def test_overwrite_refused_without_force_and_allowed_with_force(self):
        self.output.write_text("sentinel", encoding="utf-8")
        self.cli(code=2)
        self.assertEqual(self.output.read_text(), "sentinel")
        self.cli("--force")
        self.assertEqual(json.loads(self.output.read_text())["labels"], ["Still", "Bike"])

    def test_invalid_input_never_changes_existing_output_even_with_force(self):
        self.input.write_text('{"NuTime": []}', encoding="utf-8")
        self.output.write_text("sentinel", encoding="utf-8")
        self.cli("--force", code=2)
        self.assertEqual(self.output.read_text(), "sentinel")

    def test_duplicate_json_key_is_rejected(self):
        self.input.write_text('{"NuTime": [], "NuTime": []}', encoding="utf-8")
        self.cli(code=2)
        self.assertFalse(self.output.exists())

    def test_malformed_json_does_not_create_output(self):
        self.input.write_text('{', encoding="utf-8")
        self.cli(code=2)
        self.assertFalse(self.output.exists())

    def test_input_cannot_be_overwritten(self):
        before = self.input.read_bytes()
        self.output = self.input
        self.cli("--force", code=2)
        self.assertEqual(self.input.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
