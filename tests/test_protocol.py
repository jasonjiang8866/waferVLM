import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

import numpy as np
import pandas as pd

from wafervlm.core import LABELS, PROMPT, audit_dataset, metrics, parse_label, sha256_file, write_json, write_jsonl
from wafervlm.data import group_records, prepare, quarantine_conflicts, render_map, scalar, split_records
from wafervlm.acquisition import acquire
from wafervlm.experiment import compare


def records():
    return [dict(id=f"{label}-{i}", lot=f"lot-{i}", label=label,
                 map_sha256=f"{label}-{i}") for label in LABELS for i in range(100)]


class ProtocolTests(unittest.TestCase):
    def test_nested_label_and_unlabeled(self):
        self.assertEqual(scalar(np.array([["Edge-Ring"]])), "Edge-Ring")
        self.assertIsNone(scalar(np.array([])))

    def test_render_preserves_palette_and_aspect(self):
        image = render_map([[1, 2, 1, 2], [0, 1, 2, 0]], size=32)
        a = np.asarray(image)
        self.assertEqual(image.size, (32, 32))
        self.assertTrue((a[:8] == 0).all())
        self.assertEqual(set(map(tuple, a.reshape(-1, 3))), {(0, 0, 0), (0, 90, 180), (255, 220, 0)})

    def test_render_rejects_invalid_values(self):
        for a in ([[1, 3]], [[float("nan")]], [], [1, 2]):
            with self.assertRaises(ValueError):
                render_map(a)

    def test_group_split_deterministic_and_disjoint(self):
        a, b = split_records(records(), train_cap=5, eval_cap=3), split_records(list(reversed(records())), train_cap=5, eval_cap=3)
        self.assertEqual(a, b)
        lots = {s: {r["lot"] for r in rows} for s, rows in a.items()}
        self.assertFalse(lots["train"] & lots["validation"] or lots["test"] & lots["train"] or lots["test"] & lots["validation"])
        self.assertEqual(len(a["train"]), 40)
        self.assertEqual(len(a["test"]), 24)

    def test_duplicate_maps_union_lots(self):
        rows = [dict(id="a", lot="lot-a", label="center", map_sha256="same"),
                dict(id="b", lot="lot-b", label="center", map_sha256="same"),
                dict(id="c", lot="lot-b", label="scratch", map_sha256="different")]
        unique, groups = group_records(rows)
        self.assertEqual(len(unique), 2)
        self.assertEqual(groups["lot-a"], groups["lot-b"])

    def test_conflicting_duplicate_labels_rejected(self):
        with self.assertRaisesRegex(ValueError, "conflicting"):
            group_records([dict(lot="a", label="center", map_sha256="same"),
                           dict(lot="b", label="scratch", map_sha256="same")])

    def test_conflict_quarantine_excludes_every_copy(self):
        rows = [dict(lot="a", label="center", map_sha256="same"),
                dict(lot="b", label="center", map_sha256="same"),
                dict(lot="c", label="scratch", map_sha256="same"),
                dict(lot="d", label="scratch", map_sha256="other")]
        retained, excluded = quarantine_conflicts(rows)
        self.assertEqual(retained, rows[3:])
        self.assertEqual(excluded, rows[:3])
        self.assertEqual(quarantine_conflicts(list(reversed(rows)))[0], retained)

    def test_cached_archive_tampering_fails_without_extraction_or_network(self):
        with tempfile.TemporaryDirectory() as tmp:
            archive = Path(tmp) / "wm811k.zip"
            archive.write_bytes(b"bad archive")
            with patch("wafervlm.acquisition.urllib.request.urlopen") as network:
                with patch("wafervlm.acquisition.zipfile.ZipFile") as unzip:
                    with self.assertRaisesRegex(ValueError, "checksum/size"):
                        acquire(tmp)
            network.assert_not_called()
            unzip.assert_not_called()

    def test_download_failure_removes_partial_and_never_unpickles(self):
        import io
        with tempfile.TemporaryDirectory() as tmp:
            with patch("wafervlm.acquisition.urllib.request.urlopen", return_value=io.BytesIO(b"bad")):
                with self.assertRaisesRegex(ValueError, "checksum/size"):
                    acquire(tmp)
            self.assertEqual(list(Path(tmp).iterdir()), [])

    def test_missing_class_fails(self):
        with self.assertRaisesRegex(ValueError, "No"):
            split_records(records()[:2])

    def test_strict_prediction_parser(self):
        self.assertEqual(parse_label(" Edge-Ring\n"), "edge-ring")
        for response in ("It is edge-ring", "center or donut", '{"label":"center"}', "<think>x</think>center", ""):
            self.assertIsNone(parse_label(response))
        self.assertIsNone(parse_label("center", truncated=True))

    def test_invalid_predictions_count_as_errors(self):
        rows = [dict(label=k, prediction=k) for k in LABELS]
        rows[0]["prediction"] = None
        m = metrics(rows)
        self.assertEqual(m["accuracy"], 7 / 8)
        self.assertEqual(m["macro_f1"], 7 / 8)
        self.assertEqual(m["invalid_rate"], 1 / 8)
        self.assertEqual(m["per_class"][LABELS[0]]["recall"], 0)

    def test_source_digest_checked_before_pickle_load(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "source.pkl"
            p.write_bytes(b"not a pickle")
            with self.assertRaisesRegex(ValueError, "trusted"):
                prepare(p, Path(tmp) / "out", sha256_file(p))
            with self.assertRaisesRegex(ValueError, "SHA256"):
                prepare(p, Path(tmp) / "out", "0" * 64, trust_pickle=True)

    def test_end_to_end_data_and_tamper_detection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rng = np.random.default_rng(7)
            df = pd.DataFrame([dict(waferMap=rng.integers(0, 3, (12, 12), dtype=np.uint8),
                                    lotName=r["lot"], failureType=np.array([[r["label"]]])) for r in records()])
            source = root / "source.pkl"
            df.to_pickle(source)
            output = root / "prepared"
            prepare(source, output, sha256_file(source), trust_pickle=True, train_cap=2, eval_cap=1)
            splits = audit_dataset(output)
            self.assertEqual(len(splits["train"]), 16)
            p = output / splits["test"][0]["image"]
            p.write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "checksum"):
                audit_dataset(output)

    def test_real_prepare_conflict_opt_in_and_quarantine_audit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rng = np.random.default_rng(7)
            rows = [dict(waferMap=rng.integers(0, 3, (12, 12), dtype=np.uint8),
                         lotName=r["lot"], failureType=np.array([[r["label"]]])) for r in records()]
            conflict_map = np.ones((7, 9), dtype=np.uint8)
            rows.extend(dict(waferMap=conflict_map, lotName=f"conflict-{i}", failureType=label)
                        for i, label in enumerate(("center", "scratch")))
            source = root / "source.pkl"
            pd.DataFrame(rows).to_pickle(source)
            output = root / "prepared"
            with self.assertRaisesRegex(ValueError, "conflicting"):
                prepare(source, output, sha256_file(source), trust_pickle=True)
            self.assertFalse(output.exists())
            report = prepare(source, output, sha256_file(source), trust_pickle=True,
                             train_cap=2, eval_cap=1, conflict_policy="exclude")
            self.assertEqual(report["quarantined_rows"], 2)
            self.assertEqual(report["quarantined_unique_maps"], 1)
            splits = audit_dataset(output)
            # Even a rechecksummed manifest cannot reintroduce a quarantined map.
            from wafervlm.core import read_jsonl
            quarantined = read_jsonl(output / "quarantine.jsonl")
            splits["train"][0]["map_sha256"] = quarantined[0]["map_sha256"]
            write_jsonl(output / "train.jsonl", splits["train"])
            report["manifest_sha256"]["train"] = sha256_file(output / "train.jsonl")
            write_json(output / "dataset.json", report)
            with self.assertRaisesRegex(ValueError, "Quarantined"):
                audit_dataset(output)

    def test_audit_catches_cross_split_leakage_even_with_updated_manifest_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifests = {}
            image = root / "image.png"
            render_map([[1, 2], [0, 1]]).save(image)
            for split in ("train", "validation", "test"):
                rows = [dict(id=f"{split}-{k}", lot="same-lot", map_sha256=f"{split}-{k}",
                             image="image.png", image_sha256=sha256_file(image), label=k, split=split) for k in LABELS]
                write_jsonl(root / f"{split}.jsonl", rows)
                manifests[split] = sha256_file(root / f"{split}.jsonl")
            write_json(root / "dataset.json", dict(labels=list(LABELS), prompt=PROMPT, manifest_sha256=manifests))
            with self.assertRaisesRegex(ValueError, "leakage"):
                audit_dataset(root)

    def test_compare_refuses_different_eval_sets(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for i, name in enumerate(("base", "tuned")):
                p = root / name
                p.mkdir()
                write_json(p / "run.json", dict(dataset_sha256="x", model_id="q", model_revision="r", prompt=PROMPT,
                    labels=list(LABELS), config=dict(split="test", max_new_tokens=16, do_sample=False,
                    enable_thinking=False, device="cpu", adapter=bool(i))))
                write_jsonl(p / "predictions.jsonl", [dict(id=str(i), label="center", prediction="center", image_sha256="x")])
            with self.assertRaisesRegex(ValueError, "examples differ"):
                compare(root / "base", root / "tuned", root / "out")


if __name__ == "__main__":
    unittest.main()
