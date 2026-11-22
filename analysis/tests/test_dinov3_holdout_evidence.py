"""Offline regression tests for committed frozen DINO holdout evidence."""
import json
import shutil
import tempfile
import unittest
from pathlib import Path

from scripts.verify_dinov3_holdout_evidence import (
    ROOT, CHECKSUMS, FREEZE, LEDGER, RECEIPT, REPORT, RETENTION,
    verify_archives, verify_repository,
)

FILES = (FREEZE, LEDGER, RECEIPT, REPORT, CHECKSUMS, RETENTION)


class HoldoutEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tempdir.cleanup)
        self.root = Path(self.tempdir.name)
        for name in FILES:
            destination = self.root / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / name, destination)

    def change(self, name, mutate):
        path = self.root / name
        data = json.loads(path.read_text())
        mutate(data)
        path.write_text(json.dumps(data, indent=2) + "\n")

    def test_published_evidence_is_cross_consistent(self):
        result = verify_repository(ROOT)
        self.assertEqual((result["run_id"], result["models"], result["holdout_predictions"]),
                         ("44843362544", 6, 888))
        self.assertEqual(result["durable_copy_status"], "pending")
        self.assertEqual(result["independent_cleanup"], "verified_absent")

    def test_freeze_mutation_detected(self):
        self.change(FREEZE, lambda d: d["endpoints"][0].update(weight_sha256="0" * 64))
        with self.assertRaisesRegex(ValueError, "Frozen protocol hash"):
            verify_repository(self.root)

    def test_cleanup_receipt_mutation_detected(self):
        self.change(RECEIPT, lambda d: d["cleanup"].update(status="running"))
        with self.assertRaisesRegex(ValueError, "Ledger/receipt cleanup"):
            verify_repository(self.root)

    def test_insufficient_independent_cleanup_detected(self):
        self.change(RECEIPT, lambda d: d["independent"]["cleanup"].update(checks=1))
        self.change(LEDGER, lambda d: d["attempts"][-1]["independent"]["cleanup"].update(checks=1))
        with self.assertRaisesRegex(ValueError, "Insufficient consecutive"):
            verify_repository(self.root)

    def test_cumulative_cost_drift_detected(self):
        self.change(RECEIPT, lambda d: d["launch_spec"]["ledger"].update(usd=d["launch_spec"]["ledger"]["usd"] + 0.5))
        with self.assertRaisesRegex(ValueError, "Cumulative cost/duration"):
            verify_repository(self.root)

    def test_premature_archival_claim_detected(self):
        self.change(RETENTION, lambda d: d["durable_copy"].update(status="verified"))
        with self.assertRaisesRegex(ValueError, "read-back timestamp"):
            verify_repository(self.root)

    def test_corrupt_original_zip_rejected_before_parse(self):
        fake = self.root / "fake.zip"
        fake.write_bytes(b"corrupt archive")
        with self.assertRaisesRegex(ValueError, "primary original ZIP SHA"):
            verify_archives(ROOT, fake, fake)


if __name__ == "__main__":
    unittest.main()
