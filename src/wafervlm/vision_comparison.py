"""Compare paired validation predictions and render data-derived wafer examples."""
import json
import math
import statistics
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from .core import LABELS, checked_image, metrics, parse_label, read_jsonl, sha256_file, write_json
from .dinov3 import ROOT, check_data, recipe


def validate_predictions(rows, reference):
    lookup = {r["id"]: r for r in reference}
    if len(lookup) != len(reference) or len(rows) != len(reference) or {r["id"] for r in rows} != set(lookup):
        raise ValueError("Predictions must cover every validation example exactly once")
    for row in rows:
        actual = lookup[row["id"]]
        if actual["split"] != "validation" or any(row[k] != actual[k] for k in ("label", "image_sha256")):
            raise ValueError("Prediction/validation image identity mismatch")
        if parse_label(row["response"], row["truncated"]) != row["prediction"]:
            raise ValueError("Prediction differs from recorded output")
    metrics(rows)  # Also reject unknown prediction labels.
    return {r["id"]: r for r in rows}


def paired_report(left, right, reference):
    a, b = validate_predictions(left, reference), validate_predictions(right, reference)
    groups = {k: [] for k in ("both_correct", "qwen_only_correct", "dinov3_only_correct", "both_wrong")}
    for identity in sorted(a):
        x, y = a[identity], b[identity]
        correct = x["prediction"] == x["label"], y["prediction"] == y["label"]
        group = {(True, True): "both_correct", (True, False): "qwen_only_correct",
                 (False, True): "dinov3_only_correct", (False, False): "both_wrong"}[correct]
        groups[group].append(identity)
    scores = [metrics(left), metrics(right)]
    return {"qwen": scores[0], "dinov3": scores[1], "groups": groups,
            "macro_f1_delta": scores[1]["macro_f1"] - scores[0]["macro_f1"],
            "accuracy_delta": scores[1]["accuracy"] - scores[0]["accuracy"],
            "thumbnail_rule": "lowest sample IDs per category/cell; no confidence selection",
            "axes": {"x": "true_label", "y": "predicted_label"},
            "interpretation": "Paired validation comparison; architecture/pretraining/adaptation differ. Not a holdout result."}


def completed_result(root, reference):
    from .confirmation import verify_tree
    root = Path(root)
    hashes = verify_tree(root)
    info = json.loads((root / "run.json").read_text())
    cfg = recipe()
    if (json.loads((root / "status.json").read_text())["status"] != "completed"
            or info["model_id"] != cfg["model_id"] or info["model_revision"] != cfg["model_revision"]
            or info["dataset_sha256"] != cfg["dataset_sha256"]
            or info["validation_sha256"] != cfg["validation_sha256"]
            or info["config"]["split"] != "validation"):
        raise ValueError("DINOv3 result identity/completion mismatch")
    rows = read_jsonl(root / "predictions.jsonl")
    validate_predictions(rows, reference)
    for row in rows:
        probs = row.get("probabilities", [])
        if (len(probs) != len(LABELS) or any(not math.isfinite(p) or not 0 <= p <= 1 for p in probs)
                or not math.isclose(sum(probs), 1.0, abs_tol=1e-5)
                or LABELS[max(range(len(LABELS)), key=probs.__getitem__)] != row["prediction"]):
            raise ValueError("DINOv3 probabilities disagree with predicted class")
    score = metrics(rows)
    recorded = json.loads((root / "metrics.json").read_text())
    if any(recorded[k] != score[k] for k in score):
        raise ValueError("Stored metrics disagree with predictions")
    return info, rows, hashes


def draw_matrix(path, dataset, reference, rows, title):
    image = Image.new("RGB", (1180, 1190), "#f3f6fa")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=17)
    small = ImageFont.load_default(size=12)
    draw.text((30, 20), title + " / validation", font=ImageFont.load_default(size=25), fill="#152b42")
    draw.text((310, 66), "X: TRUE LABEL    |    Y: PREDICTED LABEL", font=font, fill="#152b42")
    lookup = {r["id"]: r for r in reference}
    for col, true in enumerate(LABELS):
        draw.text((166 + col * 125, 105), true, font=font, fill="#152b42")
    for rownum, predicted in enumerate((*LABELS, None)):
        y = 145 + rownum * 108
        draw.text((12, y + 35), predicted or "invalid", font=font, fill="#152b42")
        for col, true in enumerate(LABELS):
            x = 155 + col * 125
            selected = sorted((r for r in rows if r["label"] == true and r["prediction"] == predicted), key=lambda r: r["id"])
            total = sum(r["label"] == true for r in rows)
            draw.rectangle((x, y, x + 119, y + 102), fill="#e5f4ef" if predicted == true else "#fff3e9")
            draw.text((x + 5, y + 3), f"{len(selected)}/{total} ({len(selected)/total:.0%})", font=small, fill="#152b42")
            if selected:
                with Image.open(checked_image(dataset, lookup[selected[0]["id"]])) as wafer:
                    image.paste(wafer.convert("RGB").resize((64, 64), Image.Resampling.NEAREST), (x + 26, y + 23))
                draw.text((x + 5, y + 88), selected[0]["id"][:12], font=small, fill="#53667b")
    draw.text((30, 1140), "Counts include every example; thumbnail = lowest ID. Percentages use true-label columns.", font=small, fill="#53667b")
    image.save(path)


def draw_pairs(path, dataset, reference, left, right, report):
    image = Image.new("RGB", (1240, 1050), "#f3f6fa")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=18)
    small = ImageFont.load_default(size=14)
    draw.text((24, 22), "Same wafer, two predictions / validation", font=ImageFont.load_default(size=28), fill="#152b42")
    a, b = {r["id"]: r for r in left}, {r["id"]: r for r in right}
    lookup = {r["id"]: r for r in reference}
    titles = ["Both correct", "Only Qwen correct", "Only DINOv3 correct", "Both wrong"]
    for col, ((group, ids), title) in enumerate(zip(report["groups"].items(), titles)):
        x = 20 + 305 * col
        draw.text((x, 80), f"{title}: {len(ids)}", font=font, fill="#152b42")
        if not ids:
            draw.text((x, 125), "No examples", font=font, fill="#53667b")
        for rownum, identity in enumerate(ids[:3]):
            y = 125 + rownum * 285
            with Image.open(checked_image(dataset, lookup[identity])) as wafer:
                image.paste(wafer.convert("RGB").resize((160, 160), Image.Resampling.NEAREST), (x + 45, y))
            for index, text in enumerate((f"True: {a[identity]['label']}",
                    f"Qwen: {a[identity]['prediction'] or 'invalid'}",
                    f"DINOv3: {b[identity]['prediction'] or 'invalid'}", identity)):
                draw.text((x, y + 169 + 22 * index), text, font=small, fill="#152b42")
    draw.text((24, 1000), "Up to 3 lowest IDs per category; complete category membership is in comparison.json.", font=small, fill="#53667b")
    image.save(path)


def compare(dataset, dinov3_result, output):
    from .experiment import new_output, seal_output
    reference = check_data(dataset)["validation"]
    qwen = ROOT / "docs/showcase/checkpoint-250/predictions.jsonl"
    if sha256_file(qwen) != recipe()["qwen_predictions_sha256"]:
        raise ValueError("Released Qwen predictions changed")
    left = read_jsonl(qwen)
    info, right, hashes = completed_result(dinov3_result, reference)
    report = paired_report(left, right, reference)
    report.update(dinov3_config=info["config"], qwen_prediction_sha256=sha256_file(qwen),
                  dinov3_prediction_sha256=hashes["predictions.jsonl"])
    output = new_output(output)
    write_json(output / "comparison.json", report)
    draw_matrix(output / "qwen-matrix.png", dataset, reference, left, "Qwen checkpoint 250")
    draw_matrix(output / "dinov3-matrix.png", dataset, reference, right, "DINOv3 " + info["config"]["mode"])
    draw_pairs(output / "paired-examples.png", dataset, reference, left, right, report)
    lines = ["# Qwen / DINOv3 validation comparison", "", report["interpretation"], "",
             "| Model | Accuracy | Macro-F1 |", "|---|---:|---:|"]
    for label, key in (("Qwen checkpoint 250 (seed 17)", "qwen"),
                       (f"DINOv3 {info['config']['mode']} (seed {info['config']['seed']})", "dinov3")):
        lines.append(f"| {label} | {report[key]['accuracy']:.3%} | {report[key]['macro_f1']:.6f} |")
    lines += ["", "DINOv3 chooses one of eight classes by construction; zero invalid outputs is not a quality gain.",
              "Latency is not compared with historical Qwen generation timings: rerun both models on the same host with matching timing boundaries.",
              "", "![Paired wafer examples](paired-examples.png)", "",
              "X = true label; Y = predicted label (including invalid). Every validation example is counted.", "",
              "![Qwen confusion matrix](qwen-matrix.png)", "", "![DINOv3 confusion matrix](dinov3-matrix.png)"]
    (output / "README.md").write_text("\n".join(lines) + "\n")
    seal_output(output)
    return report


def summarize(dataset, results, output):
    """Require all six runs and one fixed recipe per arm; never select the best seed."""
    from .experiment import new_output, seal_output
    reference = check_data(dataset)["validation"]
    cfg, runs, recipes = recipe(), {}, {}
    for path in results:
        info, rows, hashes = completed_result(path, reference)
        config = info["config"]
        key = (config["mode"], config["seed"])
        if key in runs:
            raise ValueError("Duplicate mode/seed")
        fixed = {k: v for k, v in config.items() if k != "seed"}
        if key[0] in recipes and recipes[key[0]] != fixed:
            raise ValueError("Training recipe differs between seeds")
        recipes[key[0]] = fixed
        runs[key] = {"mode": key[0], "seed": key[1], "metrics": metrics(rows),
                     "predictions_sha256": hashes["predictions.jsonl"]}
    if set(runs) != {(m, s) for m in cfg["modes"] for s in cfg["seeds"]}:
        raise ValueError("Expected linear and LoRA results for all three seeds")
    summary = {m: {metric: {"mean": statistics.mean(runs[m, s]["metrics"][metric] for s in cfg["seeds"]),
                           "sample_sd": statistics.stdev(runs[m, s]["metrics"][metric] for s in cfg["seeds"])}
                   for metric in ("accuracy", "macro_f1")} for m in cfg["modes"]}
    deltas = [{"seed": s, "macro_f1_delta": runs["lora", s]["metrics"]["macro_f1"] - runs["linear", s]["metrics"]["macro_f1"]}
              for s in cfg["seeds"]]
    report = {"runs": list(runs.values()), "summary": summary, "paired_lora_minus_linear": deltas,
              "recipes": recipes, "seed17_role": "development", "holdout_evaluated": False,
              "interpretation": "Validation seed dispersion; not a confidence interval or final generalization score."}
    output = new_output(output)
    write_json(output / "summary.json", report)
    seal_output(output)
    return report
