import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from wafervlm.core import LABELS, metrics, sha256_file, write_json, write_jsonl
from wafervlm.dinov3 import check_data, recipe
from wafervlm.experiment import seal_output
from wafervlm.vision_comparison import (completed_result, draw_matrix, draw_pairs, paired_report,
                                      summarize, validate_predictions)


def fixture(root):
    reference, predictions = [], []
    for i, label in enumerate(LABELS):
        path = root / f"{label}.png"
        Image.new("RGB", (256, 256), (i * 25, 90, 180)).save(path)
        row = dict(id=f"sample-{i}", label=label, split="validation", image=path.name,
                   image_sha256=sha256_file(path))
        reference.append(row)
        predictions.append({**row, "prediction": label, "response": label, "truncated": False})
    return reference, predictions


def result(root, rows, mode="linear", seed=17, **overrides):
    root.mkdir()
    cfg = recipe()
    write_json(root / "run.json", dict(model_id=cfg["model_id"], model_revision=cfg["model_revision"],
        dataset_sha256=cfg["dataset_sha256"], validation_sha256=cfg["validation_sha256"],
        config={**cfg, "mode": mode, "seed": seed, **overrides}))
    write_json(root / "status.json", {"status": "completed"})
    write_jsonl(root / "predictions.jsonl", [{**r, "probabilities": [float(label == r["prediction"]) for label in LABELS]} for r in rows])
    write_json(root / "metrics.json", metrics(rows))
    seal_output(root)
    return root


class DinoProtocolTests(unittest.TestCase):
    def test_wrong_dataset_rejected_before_audit(self):
        with tempfile.TemporaryDirectory() as tmp, patch("wafervlm.dinov3.audit_dataset") as audit:
            (Path(tmp) / "dataset.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "frozen"):
                check_data(tmp)
            audit.assert_not_called()

    def test_paired_categories_keep_invalid_and_accept_reordered_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            reference, left = fixture(Path(tmp))
            right = copy.deepcopy(left)
            left[1].update(prediction=None, response="unrecognized")
            left[3].update(prediction="donut", response="donut")
            right[2].update(prediction="loc", response="loc")
            right[3].update(prediction="donut", response="donut")
            report = paired_report(left, list(reversed(right)), reference)
            self.assertEqual(report["groups"]["dinov3_only_correct"], ["sample-1"])
            self.assertEqual(report["groups"]["qwen_only_correct"], ["sample-2"])
            self.assertEqual(report["groups"]["both_wrong"], ["sample-3"])
            self.assertEqual(report["qwen"]["invalid_rate"], 1 / 8)
            self.assertEqual(sum(map(len, report["groups"].values())), 8)

    def test_omissions_duplicates_wrong_images_labels_and_parser_drift_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            reference, rows = fixture(Path(tmp))
            corrupt = [rows[:-1], rows[:-1] + [rows[0]]]
            for key, value in [("image_sha256", "wrong"), ("label", "scratch"), ("prediction", "donut")]:
                changed = copy.deepcopy(rows)
                changed[0][key] = value
                corrupt.append(changed)
            for values in corrupt:
                with self.assertRaises(ValueError):
                    validate_predictions(values, reference)

    def test_forged_metric_or_failed_training_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            reference, rows = fixture(root)
            output = result(root / "result", rows)
            completed_result(output, reference)
            score = metrics(rows)
            score["macro_f1"] = .4
            write_json(output / "metrics.json", score)
            seal_output(output)
            with self.assertRaisesRegex(ValueError, "metrics"):
                completed_result(output, reference)
            write_json(output / "status.json", {"status": "failed"})
            seal_output(output)
            with self.assertRaisesRegex(ValueError, "completion"):
                completed_result(output, reference)

    def test_probability_and_argmax_drift_rejected(self):
        from wafervlm.core import read_jsonl
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            reference, rows = fixture(root)
            output = result(root / "result", rows)
            predictions = read_jsonl(output / "predictions.jsonl")
            predictions[0]["probabilities"] = [0., 1., 0., 0., 0., 0., 0., 0.]
            write_jsonl(output / "predictions.jsonl", predictions)
            seal_output(output)
            with self.assertRaisesRegex(ValueError, "probabilities"):
                completed_result(output, reference)

    def test_summary_requires_six_runs_one_recipe_per_arm(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            reference, rows = fixture(root)
            outputs = [result(root / f"{mode}-{seed}", rows, mode, seed)
                       for mode in recipe()["modes"] for seed in recipe()["seeds"]]
            with patch("wafervlm.vision_comparison.check_data", return_value={"validation": reference}):
                report = summarize(root, outputs, root / "summary")
                self.assertEqual(report["summary"]["lora"]["macro_f1"], {"mean": 1., "sample_sd": 0.})
                with self.assertRaisesRegex(ValueError, "all three seeds"):
                    summarize(root, outputs[:-1], root / "incomplete")
                with self.assertRaisesRegex(ValueError, "Duplicate"):
                    summarize(root, outputs + outputs[:1], root / "duplicate")
                runpath = outputs[-1] / "run.json"
                run = json.loads(runpath.read_text())
                run["config"]["epochs"] = 2
                write_json(runpath, run)
                seal_output(outputs[-1])
                with self.assertRaisesRegex(ValueError, "recipe differs"):
                    summarize(root, outputs, root / "drift")

    def test_renderers_generate_complete_pngs_with_empty_categories(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            reference, rows = fixture(root)
            report = paired_report(rows, rows, reference)
            draw_matrix(root / "matrix.png", root, reference, rows, "Synthetic test")
            draw_pairs(root / "pairs.png", root, reference, rows, rows, report)
            with Image.open(root / "matrix.png") as image:
                self.assertEqual(image.size, (1180, 1190))
            with Image.open(root / "pairs.png") as image:
                self.assertEqual(image.size, (1240, 1050))


if __name__ == "__main__":
    unittest.main()
