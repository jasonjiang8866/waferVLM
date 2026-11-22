"""CPU-only, hash-gated comparison of published DINO and Qwen validation outputs."""
import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path

from wafervlm.core import LABELS, checked_image, metrics, read_jsonl, sha256_file, write_json
from wafervlm.dinov3 import check_data, recipe
from wafervlm.vision_comparison import draw_matrix, paired_report, validate_predictions

ROOT = Path(__file__).resolve().parents[1]
QWEN_SHA = "ae2cd56f7047c3015a34711324072f90f70dac2e"
DINO = ROOT / "docs/showcase/dinov3-seed17"
ENDPOINTS = [f"seed{s}-step{t}" for s in (17, 42, 123) for t in (200, 250)]


def verified(path, digest, reference=None, probabilities=False):
    if sha256_file(path) != digest:
        raise ValueError(f"SHA-256 mismatch: {path}")
    rows = read_jsonl(path)
    if reference is not None:
        validate_predictions(rows, reference)
    if probabilities:
        for row in rows:
            p = row.get("probabilities", [])
            if (len(p) != 8 or any(not math.isfinite(v) or not 0 <= v <= 1 for v in p)
                    or not math.isclose(sum(p), 1, abs_tol=1e-5)
                    or LABELS[max(range(8), key=p.__getitem__)] != row["prediction"]):
                raise ValueError("Invalid probability vector or argmax")
    return rows


def analyze():
    cfg = recipe()
    reference = verified(DINO / "validation.jsonl", cfg["validation_sha256"])
    if len(reference) != 160 or Counter(r["label"] for r in reference) != Counter(dict.fromkeys(LABELS, 20)):
        raise ValueError("Expected the complete balanced 160-map validation set")
    hashes = json.loads((DINO / "checksums.json").read_text())
    source_hashes = {"validation.jsonl": cfg["validation_sha256"]}
    models = {"qwen-published": verified(ROOT / "docs/showcase/checkpoint-250/predictions.jsonl",
                                        cfg["qwen_predictions_sha256"], reference)}
    source_hashes["qwen-published"] = cfg["qwen_predictions_sha256"]
    for name, filename in (("linear", "linear-predictions.jsonl"), ("lora", "predictions.jsonl"),
                           ("linear-reload", "linear-reload-predictions.jsonl"),
                           ("lora-reload", "lora-reload-predictions.jsonl")):
        models[name] = verified(DINO / filename, hashes[filename], reference, True)
        source_hashes[name] = hashes[filename]
    receipt = json.loads((ROOT / "docs/evidence/vlm-repeat-complete.json").read_text())
    qwen = {}
    for pair in receipt["pairs"]:
        endpoint = f"seed{pair['seed']}-step{pair['step']}"
        first, = [r for r in pair["passes"] if r["repeat"] == 1]
        if endpoint not in ENDPOINTS or first["output"] != endpoint + "-repeat1" or endpoint in qwen:
            raise ValueError("Qwen first-pass identity mismatch")
        path = ROOT / "docs/evidence/qwen-validation-first-pass" / (endpoint + ".jsonl")
        qwen[endpoint] = verified(path, first["predictions_sha256"], reference)
        source_hashes[endpoint] = first["predictions_sha256"]
    if set(qwen) != set(ENDPOINTS):
        raise ValueError("Incomplete six-Qwen matrix")
    indexed = {name: {r["id"]: r for r in rows} for name, rows in {**qwen, **models}.items()}
    cases = []
    for row in sorted(reference, key=lambda r: r["id"]):
        case = {k: row[k] for k in ("id", "label", "image_sha256")}
        case.update({name: indexed[name][row["id"]]["prediction"] for name in indexed})
        case["qwen_correct_count"] = sum(case[e] == case["label"] for e in ENDPOINTS)
        cases.append(case)
    common = [r for r in cases if r["qwen_correct_count"] == 0]
    diagnostics = [r for r in cases if r["label"] == "loc" and r["qwen_correct_count"] in (0, 6)]
    if len(common) != 16 or len(diagnostics) != 10 or sum(r["qwen_correct_count"] == 0 for r in diagnostics) != 7:
        raise ValueError("PR54 diagnostic membership changed")
    paired = {m: paired_report(models["qwen-published"], models[m], reference) for m in ("linear", "lora")}
    for result in paired.values():
        if sum(map(len, result["groups"].values())) != 160:
            raise ValueError("Four-way counts must sum to 160")
    rescues = {m: {"count": sum(r[m] == r["label"] for r in common),
                   "by_class": dict(Counter(r["label"] for r in common if r[m] == r["label"])),
                   "rescued_ids": [r["id"] for r in common if r[m] == r["label"]],
                   "remaining_ids": [r["id"] for r in common if r[m] != r["label"]]} for m in ("linear", "lora")}
    reloads = {m: {"metrics": metrics(models[m + "-reload"]),
                   "changes": [{"id": r["id"], "label": r["label"], "selected": r[m], "reload": r[m + "-reload"]}
                               for r in cases if r[m] != r[m + "-reload"]]} for m in ("linear", "lora")}
    report = dict(samples=160, qwen_pr54_sha=QWEN_SHA, source_sha256=source_hashes,
                  primary_rule="Original published DINO selected-epoch batch-8 outputs; published Qwen seed17 step250",
                  diagnostic_rule="All seven loc maps missed by six Qwen first passes plus all three unanimously correct loc maps",
                  metrics={m: metrics(models[m]) for m in ("qwen-published", "linear", "lora")},
                  paired=paired, common_errors=common, rescues=rescues, loc_diagnostics=diagnostics,
                  reload_checks=reloads, holdout_evaluated=False, training_seeds=[17])
    return report, cases, reference, models


def gallery(path, dataset, reference, cases, predictions=True):
    from PIL import Image, ImageDraw, ImageFont
    lookup = {r["id"]: r for r in reference}
    height = 470 if predictions else 310
    canvas = Image.new("RGB", (1500, height * 2 + 90), "#f3f6fa")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=16)
    draw.text((20, 15), "Loc diagnostics: seven six-Qwen misses + three all-correct controls", fill="#152b42", font=ImageFont.load_default(size=24))
    draw.text((20, 48), "Actual hash-verified images. Yellow = failed die; blue = passing die; black = outside wafer.", fill="#152b42", font=font)
    for i, case in enumerate(cases):
        x, y = 20 + (i % 5) * 300, 90 + (i // 5) * height
        with Image.open(checked_image(dataset, lookup[case["id"]])) as wafer:
            canvas.paste(wafer.convert("RGB"), (x, y))
        draw.text((x, y + 263), case["id"], fill="#152b42", font=font)
        if predictions:
            lines = [f"True: loc | Qwen correct: {case['qwen_correct_count']}/6"]
            lines += [f"Q{s} 200/250: {case[f'seed{s}-step200']} / {case[f'seed{s}-step250']}" for s in (17, 42, 123)]
            lines += [f"DINO linear: {case['linear']}", f"DINO LoRA: {case['lora']}"]
            for n, text in enumerate(lines):
                draw.text((x, y + 290 + n * 24), text, fill="#152b42", font=font)
    canvas.save(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--dataset", type=Path, help="Optional frozen dataset for image SHA verification and real-image graphics")
    args = parser.parse_args()
    report, cases, reference, models = analyze()
    if args.output:
        args.output.mkdir(parents=True, exist_ok=True)
        for name, rows in (("all-160", cases), ("common-errors-16", report["common_errors"]), ("loc-10", report["loc_diagnostics"])):
            with (args.output / (name + ".csv")).open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(cases[0]), lineterminator="\n")
                writer.writeheader()
                writer.writerows(rows)
        if args.dataset:
            check_data(args.dataset)  # Audits original image bytes across the frozen split.
            for r in reference:
                checked_image(args.dataset, r)
            report["image_bytes_verified"] = 160
            gallery(args.output / "loc-diagnostics.png", args.dataset, reference, report["loc_diagnostics"])
            gallery(args.output / "loc-morphology.png", args.dataset, reference, report["loc_diagnostics"], predictions=False)
            for name in ("qwen-published", "linear", "lora"):
                draw_matrix(args.output / (name + "-matrix.png"), args.dataset, reference, models[name], name)
        write_json(args.output / "analysis.json", report)
    print(json.dumps({"samples": report["samples"], "rescues": report["rescues"],
                      "four_way": {m: {k: len(v) for k, v in r["groups"].items()} for m, r in report["paired"].items()}}, indent=2))


if __name__ == "__main__":
    main()
