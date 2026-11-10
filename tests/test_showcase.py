"""Guard the axis convention and evidence joins behind the public visual report."""

import copy
import json
from pathlib import Path
import unittest

from scripts.build_showcase import build_cells, validate_predictions
from wafervlm.core import LABELS, metrics, read_jsonl, sha256_file


class ShowcaseTests(unittest.TestCase):
    def test_asymmetric_errors_use_true_columns_and_column_denominators(self):
        rows = [dict(id=str(i), label=t, prediction=p) for i, (t, p) in enumerate([
            ("loc", "scratch"), ("loc", "scratch"), ("loc", "loc"),
            ("scratch", "loc"), ("scratch", "scratch"),
        ])]
        cells = {(c["predicted_label"], c["true_label"]): c for c in build_cells(rows)}
        self.assertEqual(cells["scratch", "loc"]["count"], 2)
        self.assertEqual(cells["loc", "scratch"]["count"], 1)
        self.assertAlmostEqual(cells["scratch", "loc"]["fraction_of_true_class"], 2/3)
        self.assertEqual(cells["loc", "scratch"]["fraction_of_true_class"], 1/2)
        self.assertEqual(cells["scratch", "loc"]["thumbnail_id"], "0")
        self.assertIsNone(cells["center", "donut"]["thumbnail_id"])
        self.assertEqual(build_cells(rows), build_cells(list(reversed(rows))))

    def test_invalid_predictions_cannot_disappear_from_eight_by_eight_matrix(self):
        with self.assertRaisesRegex(ValueError, "invalid outputs"):
            build_cells([dict(id="a", label="loc", prediction=None)])

    def test_prediction_join_rejects_omissions_duplicates_and_identity_drift(self):
        manifest = [dict(id="a", label="loc", image_sha256="image-a", split="validation"),
                    dict(id="b", label="scratch", image_sha256="image-b", split="validation")]
        rows = [dict(**r, prediction=r["label"], response=r["label"], truncated=False) for r in manifest]
        self.assertEqual(set(validate_predictions(rows, manifest)), {"a", "b"})
        for bad in (rows[:1], [rows[0], rows[0]], [dict(rows[0], id="foreign"), rows[1]],
                    [dict(rows[0], image_sha256="wrong"), rows[1]],
                    [dict(rows[0], label="center"), rows[1]],
                    [dict(rows[0], response="unknown"), rows[1]],
                    [dict(rows[0], truncated=True), rows[1]]):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                validate_predictions(bad, manifest)
        bad_manifest = copy.deepcopy(manifest)
        bad_manifest[0]["split"] = "test"
        with self.assertRaisesRegex(ValueError, "validation"):
            validate_predictions(rows, bad_manifest)

    def test_published_report_matches_all_released_predictions_and_figure_hashes(self):
        root = Path(__file__).resolve().parents[1]/"docs/showcase/checkpoint-250"
        # Worker images intentionally exclude documentation; their integrity is
        # checked in repository CI, while the pure logic tests above run everywhere.
        if not root.exists():
            self.skipTest("Showcase artifacts are not packaged in worker images")
        evidence = json.loads((root/"evidence.json").read_text())
        rows = read_jsonl(root/"predictions.jsonl")
        self.assertEqual(len(rows), 160)
        self.assertEqual(len({r["id"] for r in rows}), 160)
        self.assertEqual(metrics(rows), evidence["metrics"])
        self.assertEqual(evidence["matrix_axes"], {"x": "true_label", "y": "predicted_label"})
        self.assertEqual(build_cells(rows), evidence["cells"])
        self.assertEqual(sha256_file(root/"predictions.jsonl"), evidence["prediction_sha256"])
        for label in LABELS:
            column = [c for c in evidence["cells"] if c["true_label"] == label]
            self.assertEqual(sum(c["count"] for c in column), 20)
            self.assertAlmostEqual(sum(c["fraction_of_true_class"] for c in column), 1)
        for name, digest in evidence["outputs"].items():
            with self.subTest(file=name):
                self.assertEqual(sha256_file(root/name), digest)
        self.assertFalse(evidence["inference_executed"])
        self.assertFalse(evidence["test_evaluated"])
        self.assertFalse(evidence["final_holdout_evaluated"])


if __name__ == "__main__":
    unittest.main()
