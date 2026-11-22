"""Audit fixed-budget review on the published DINO repeat-1 predictions."""
import argparse
import csv
import json
import shutil
from pathlib import Path

from runpod.dinov3_study import ROOT, names, reference_for, verify_predictions
from wafervlm.core import read_jsonl, sha256_file

SHOWCASE = ROOT / "docs/showcase"
OUTPUT = SHOWCASE / "dino-review-queue"
BUDGETS = (5, 10, 20)


def rank_maps(predictions):
    ordered = sorted(predictions, key=lambda r: (max(r["probabilities"]), r["id"]))
    return [
        dict(rank=i, id=row["id"], image_sha256=row["image_sha256"],
             label=row["label"], prediction=row["prediction"], confidence=max(row["probabilities"]))
        for i, row in enumerate(ordered, 1)
    ]


def error_counts(rows):
    errors = [row for row in rows if row["prediction"] != row["label"]]
    return (len(errors), sum(row["label"] == "loc" for row in errors),
            sum(row["prediction"] == "loc" for row in errors))


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0], lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def gallery(rows, images, output):
    misses = [r for r in rows if r["label"] == "loc" and r["prediction"] != "loc"]
    limit = len(rows) // 10
    shown = [r for r in misses if r["rank"] <= limit][:2]
    shown += sorted((r for r in misses if r["rank"] > limit),
                    key=lambda r: (-r["confidence"], r["id"]))
    lines = ["## Which loc errors reach the reviewer?", "",
             "Seed-17 LoRA on development. The queue ends at rank 16. Each thumbnail links to its original PNG.", "",
             "| Map | Sample ID | Rank | Review | True → predicted | Max probability |",
             "|---|---|---:|---|---|---:|"]
    for row in shown:
        filename = row["id"] + ".png"
        source = images / filename
        if sha256_file(source) != row["image_sha256"]:
            raise ValueError(f"Image hash mismatch for {filename}")
        dest = output / "images" / filename
        dest.parent.mkdir(parents=True, exist_ok=True)
        if source.resolve() != dest.resolve():
            shutil.copyfile(source, dest)
        path = "images/" + filename
        review = "sent" if row["rank"] <= limit else "outside"
        lines.append(f'| [<img src="{path}" width="96" alt="Wafer map">]({path}) '
                     f'| {row["id"]} | {row["rank"]} | {review} | loc → {row["prediction"]} '
                     f'| {row["confidence"]:.2%} |')
    return lines


def generate(output, images):
    summaries = []
    sections = []
    for split in ("validation", "holdout"):
        source = SHOWCASE / f"dinov3-confirmation-{split}"
        accepted = json.loads((source / "analysis.json").read_text())
        reference = reference_for(split)
        sections += [f"## {split.capitalize()} at the 10% budget", "",
                     "| Model | Reviewed | Errors sent / all | Capture | Errors left / unreviewed | Residual error rate | loc misses sent / all | False loc calls sent / all |",
                     "|---|---:|---:|---:|---:|---:|---:|---:|"]
        for name in names():
            path = source / "predictions" / name / "repeat1/predictions.jsonl"
            if sha256_file(path) != accepted["passes"][name][0]["predictions_sha256"]:
                raise ValueError(f"Expected the accepted primary predictions for {name}")
            predictions = read_jsonl(path)
            verify_predictions(predictions, reference)
            rows = rank_maps(predictions)
            write_csv(output / "ranks" / split / f"{name}.csv", rows)
            if (split, name) == ("validation", "seed17-lora"):
                examples = gallery(rows, images, output)
            errors, misses, false_calls = error_counts(rows)
            for budget in BUDGETS:
                k = len(rows) * budget // 100
                sent, sent_misses, sent_false = error_counts(rows[:k])
                left = errors - sent
                unreviewed = len(rows) - k
                summaries.append(dict(split=split, model=name, budget_percent=budget, maps=len(rows),
                                      reviewed=k, realized_review_fraction=k / len(rows),
                                      errors=errors, errors_sent=sent, error_capture_fraction=sent / errors,
                                      unreviewed=unreviewed, errors_left=left, residual_error_rate=left / unreviewed,
                                      loc_misses=misses, loc_misses_sent=sent_misses,
                                      false_loc_calls=false_calls, false_loc_calls_sent=sent_false))
                if budget == 10:
                    sections.append(f"| {name} | {k}/{len(rows)} | {sent}/{errors} | {sent/errors:.1%} "
                                    f"| {left}/{unreviewed} | {left/unreviewed:.1%} | {sent_misses}/{misses} "
                                    f"| {sent_false}/{false_calls} |")
        sections.append("")
    write_csv(output / "summary.csv", summaries)
    intro = ["# Which wafer maps should an engineer review first?", "",
             "The tables count classification errors that enter a fixed-size review queue. "
             "A human review outcome has yet to be measured. "
             "The maximum class probability is used only for ranking within a batch.", "",
             "The [protocol](protocol.md) fixes a 10% primary budget and 5% and 20% sensitivity checks. "
             "The development pilot informed this analysis. The holdout results below are retrospective "
             "on previously published predictions. Both splits use each model’s first process.", "",
             "Uniform random review would capture 10% of development errors and 14/148 (9.46%) of holdout errors in expectation. "
             "Residual error rate divides errors left by unreviewed maps. A loc miss has true label loc. "
             "A false loc call has predicted label loc and another true label.", "",
             "[All 36 budget comparisons](summary.csv) · [Per-map rankings for both splits](ranks)", ""]
    (output / "README.md").write_text("\n".join(intro + sections + examples + ["", "Reproduce from the repository root with "
            "`uv run --frozen python -m analysis.dino_review_queue`. Pass `--images` to check the gallery against "
            "the images directory produced by `scripts/prepare_public_data.py`.", ""]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--images", type=Path, default=OUTPUT / "images")
    args = parser.parse_args()
    generate(args.output, args.images)
