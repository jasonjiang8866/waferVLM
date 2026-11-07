"""Convert trusted WM811K pickles to lot-isolated, audited image manifests."""

import hashlib
import random
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from .core import LABELS, PROMPT, audit_dataset, sha256_file, write_json, write_jsonl

PALETTE = np.array([[0, 0, 0], [0, 90, 180], [255, 220, 0]], dtype=np.uint8)


def scalar(value):
    while isinstance(value, (list, tuple, np.ndarray)):
        if np.size(value) == 0:
            return None
        value = np.asarray(value, dtype=object).reshape(-1)[0]
    return value


def render_map(wafer_map, size=256):
    a = np.asarray(wafer_map)
    if a.ndim != 2 or not a.size or not np.isin(a, [0, 1, 2]).all():
        raise ValueError("Expected non-empty 2D wafer map with values 0, 1, 2")
    image = Image.fromarray(PALETTE[a.astype(np.uint8)])
    w, h = image.size
    ratio = size / max(w, h)
    image = image.resize((max(1, round(w * ratio)), max(1, round(h * ratio))), Image.Resampling.NEAREST)
    canvas = Image.new("RGB", (size, size), (0, 0, 0))
    canvas.paste(image, ((size - image.width) // 2, (size - image.height) // 2))
    return canvas


def group_records(records):
    """Union lots linked by duplicate raw maps; conflicting labels are rejected."""
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != x:
            nxt = parent[x]
            parent[x] = root
            x = nxt
        return root

    first = {}
    unique = []
    for row in records:
        find(row["lot"])
        key = row["map_sha256"]
        if key in first:
            old = first[key]
            if old["label"] != row["label"]:
                raise ValueError("Identical raw map has conflicting labels")
            a, b = find(old["lot"]), find(row["lot"])
            parent[max(a, b)] = min(a, b)
        else:
            first[key] = row
            unique.append(row)
    return unique, {r["lot"]: find(r["lot"]) for r in records}


def split_records(records, seed=17, train_cap=100, eval_cap=20):
    unique, groups = group_records(records)
    splits = {s: [] for s in ("train", "validation", "test")}
    for row in unique:
        # Group allocation does not use labels, prompts, or model results.
        h = hashlib.sha256(f"{seed}:{groups[row['lot']]}".encode()).digest()
        bucket = int.from_bytes(h[:8], "big") / 2**64
        split = "train" if bucket < 0.7 else "validation" if bucket < 0.85 else "test"
        splits[split].append({**row, "split": split})
    rng = random.Random(seed)
    selected = {}
    for split, rows in splits.items():
        selected[split] = []
        for label in LABELS:
            candidates = sorted((r for r in rows if r["label"] == label), key=lambda r: r["id"])
            rng.shuffle(candidates)
            if not candidates:
                raise ValueError(f"No {label} in {split}; need more independent lots or a declared new split seed")
            cap = train_cap if split == "train" else eval_cap
            selected[split].extend(candidates[:cap])
        selected[split].sort(key=lambda r: r["id"])
    return selected


def prepare(source, output, expected_sha256, trust_pickle=False, seed=17, train_cap=100, eval_cap=20):
    source, output = Path(source), Path(output)
    if not trust_pickle:
        raise ValueError("Pickle can execute code: require --trust-pickle for a trusted source")
    actual = sha256_file(source)
    if actual != expected_sha256.lower():
        raise ValueError("Source SHA256 does not match")
    if train_cap < 1 or eval_cap < 1:
        raise ValueError("Sample caps must be positive")
    if output.exists():
        raise ValueError("Output already exists; use a new directory")
    df = pd.read_pickle(source)
    required = {"waferMap", "failureType", "lotName"}
    if not required.issubset(df.columns):
        raise ValueError(f"Missing columns: {required - set(df.columns)}")
    records, arrays = [], {}
    excluded = 0
    for index, row in enumerate(df.itertuples(index=False)):
        label = scalar(row.failureType)
        label = str(label).strip().lower() if label is not None else None
        if label not in LABELS:
            excluded += 1
            continue
        lot = scalar(row.lotName)
        if lot is None or pd.isna(lot) or not str(lot).strip():
            raise ValueError(f"Missing lotName on labeled row {index}")
        a = np.asarray(row.waferMap)
        if a.ndim != 2 or not a.size or not np.isin(a, [0, 1, 2]).all():
            raise ValueError(f"Invalid wafer map on row {index}")
        a = a.astype(np.uint8)
        digest = hashlib.sha256(str(a.shape).encode() + a.tobytes()).hexdigest()
        identity = hashlib.sha256(f"{actual}:{index}".encode()).hexdigest()[:24]
        records.append(dict(id=identity, source_row=index, lot=str(lot), label=label, map_sha256=digest))
        arrays.setdefault(digest, a)
    selected = split_records(records, seed, train_cap, eval_cap)
    output.mkdir(parents=True)
    (output / "images").mkdir()
    for split, rows in selected.items():
        for row in rows:
            row["image"] = f"images/{row['id']}.png"
            p = output / row["image"]
            render_map(arrays[row["map_sha256"]]).save(p)
            row["image_sha256"] = sha256_file(p)
        write_jsonl(output / f"{split}.jsonl", rows)
    report = {
        "schema_version": 1, "source_sha256": actual, "seed": seed,
        "labels": list(LABELS), "prompt": PROMPT,
        "split_protocol": "sha256-lot-group-70-15-15-with-duplicate-union",
        "train_per_class_cap": train_cap, "eval_per_class_cap": eval_cap,
        "excluded_unlabeled_or_none": excluded,
        "source_labeled_rows": len(records),
        "render": "256x256 nearest-neighbor, aspect-preserving, black padding; palette 0:black 1:blue 2:yellow",
        "counts": {s: {k: sum(r['label'] == k for r in rows) for k in LABELS} for s, rows in selected.items()},
        "manifest_sha256": {s: sha256_file(output / f"{s}.jsonl") for s in selected},
    }
    write_json(output / "dataset.json", report)
    # Includes rendered-image collision checks, not just raw-map identity checks.
    audit_dataset(output)
    return report
