"""Protect identity joins and probability integrity in the CPU diagnostic."""
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from wafervlm.core import sha256_file

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("dino_paired", ROOT / "analysis/dinov3_qwen_validation.py")
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)


class PairedAnalysisTests(unittest.TestCase):
    def test_archived_evidence_and_published_output(self):
        report, cases, reference, _ = analysis.analyze()
        published = json.loads((ROOT / "docs/showcase/dinov3-qwen-paired/analysis.json").read_text())
        published.pop("image_bytes_verified")
        self.assertEqual(report, published)
        self.assertEqual(len(cases), len(reference))

    def test_rejects_corruption_missing_duplicate_and_wrong_identity(self):
        _, _, reference, models = analysis.analyze()
        original = models["linear"]
        mutations = [original[:-1], original[:-1] + [original[0]]]
        for field, value in (("label", "invalid"), ("image_sha256", "0" * 64),
                             ("probabilities", [0.125] * 8)):
            rows = [dict(r) for r in original]
            rows[0][field] = value
            mutations.append(rows)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "predictions.jsonl"
            for rows in mutations:
                path.write_text("".join(json.dumps(r) + "\n" for r in rows))
                with self.assertRaises(ValueError):
                    analysis.verified(path, sha256_file(path), reference, True)
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                analysis.verified(path, "0" * 64, reference, True)


if __name__ == "__main__":
    unittest.main()
