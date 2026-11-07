"""Shared protocol, integrity checks, and classification metrics (no Torch)."""

import hashlib
import json
from pathlib import Path

LABELS = ("center", "donut", "edge-loc", "edge-ring", "loc", "near-full", "random", "scratch")
PROMPT = (
    "Classify the defect pattern in this wafer map. Yellow cells are defective, "
    "blue cells are good dies, and black is outside the wafer. "
    "Return exactly one label from: " + ", ".join(LABELS) + ". Do not explain."
)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def write_jsonl(path, rows):
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def checked_image(root, row):
    root = Path(root).resolve()
    p = (root / row["image"]).resolve()
    if not p.is_relative_to(root):
        raise ValueError("Image path escapes dataset directory")
    if sha256_file(p) != row["image_sha256"]:
        raise ValueError(f"Image checksum mismatch: {row['id']}")
    return p


def audit_dataset(root):
    root = Path(root)
    report = json.loads((root / "dataset.json").read_text())
    if report["labels"] != list(LABELS) or report["prompt"] != PROMPT:
        raise ValueError("Dataset protocol does not match this version")
    seen = {key: {} for key in ("id", "lot", "map_sha256", "image_sha256")}
    all_rows = {}
    for split in ("train", "validation", "test"):
        path = root / f"{split}.jsonl"
        if sha256_file(path) != report["manifest_sha256"][split]:
            raise ValueError(f"Manifest checksum mismatch: {split}")
        rows = read_jsonl(path)
        if not rows:
            raise ValueError(f"Empty split: {split}")
        for row in rows:
            if row["label"] not in LABELS or row["split"] != split:
                raise ValueError("Invalid label or split")
            checked_image(root, row)
            for key, values in seen.items():
                value = row[key]
                if value in values and (key == "id" or values[value] != split):
                    raise ValueError(f"Duplicate/leakage in {key}: {value}")
                values[value] = split
        if set(r["label"] for r in rows) != set(LABELS):
            raise ValueError(f"Missing class in {split}")
        all_rows[split] = rows
    return all_rows


def parse_label(text, truncated=False):
    """Accept only a complete single label; never search explanations for a label."""
    if truncated:
        return None
    answer = text.strip().lower()
    return answer if answer in LABELS else None


def metrics(rows):
    if not rows:
        raise ValueError("Cannot evaluate zero examples")
    matrix = {label: {pred: 0 for pred in (*LABELS, "invalid")} for label in LABELS}
    for row in rows:
        if row["label"] not in LABELS or row["prediction"] not in (*LABELS, None):
            raise ValueError("Unknown class in prediction record")
        matrix[row["label"]][row["prediction"] or "invalid"] += 1
    per_class = {}
    for label in LABELS:
        tp = matrix[label][label]
        support = sum(matrix[label].values())
        predicted = sum(matrix[actual][label] for actual in LABELS)
        precision = tp / predicted if predicted else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[label] = dict(precision=precision, recall=recall, f1=f1, support=support)
    return {
        "n": len(rows),
        "accuracy": sum(matrix[k][k] for k in LABELS) / len(rows),
        "macro_f1": sum(x["f1"] for x in per_class.values()) / len(LABELS),
        "invalid_rate": sum(matrix[k]["invalid"] for k in LABELS) / len(rows),
        "per_class": per_class,
        "confusion_matrix": matrix,
    }
