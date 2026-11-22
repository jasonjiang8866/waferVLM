"""Analyze Qwen loc confusion using quartile boundaries fitted on training images."""
import argparse
import csv
import json
from bisect import bisect_left
from collections import Counter
from fractions import Fraction
from pathlib import Path

from qwen_loc_review import area_counts
from qwen_validation_errors import ROOT, load_predictions, verified_rows


def write_csv(path, fields, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def checkpoint_bins(maps, endpoints, bins, labels):
    others = [label for label in labels if label != "loc"]
    rows = []
    for bin_id in range(1, 5):
        members = [r for r in maps if bins[r["id"]] == bin_id]
        for endpoint in endpoints:
            misses = Counter()
            false_calls = Counter()
            loc_hits = other_hits = 0
            for row in members:
                label, guess = row["label"], row[endpoint]
                if label == "loc":
                    if guess == "loc":
                        loc_hits += 1
                    else:
                        misses[guess] += 1
                elif guess == "loc":
                    false_calls[label] += 1
                elif guess == label:
                    other_hits += 1
            nloc = loc_hits + misses.total()
            rows.append(dict(bin=bin_id, checkpoint=endpoint, loc_maps=nloc,
                             other_maps=len(members) - nloc, loc_hits=loc_hits,
                             loc_misses=misses.total(), false_loc=false_calls.total(),
                             other_correct=other_hits,
                             **{f"miss_to_{k}": misses[k] for k in others},
                             miss_to_invalid=sum(n for k, n in misses.items() if k not in labels),
                             **{f"false_from_{k}": false_calls[k] for k in others}))
    return rows


def run(dataset, output):
    receipt = json.loads((ROOT / "docs/evidence/wm811k-lot17.json").read_text())["dataset"]
    hashes = receipt["manifest_sha256"]
    train = verified_rows(dataset / "train.jsonl", hashes["train"])
    train_areas = [(r["label"], Fraction(*area_counts(dataset, r))) for r in train.values()]
    ordered = sorted(area for _, area in train_areas)
    # Nearest rank, one-based. Compare exact pixel ratios and keep ties in the lower bin.
    cuts = [ordered[(len(ordered) * k + 3) // 4 - 1] for k in (1, 2, 3)]
    train_counts = Counter(bisect_left(cuts, area) + 1 for area in ordered)
    per_class = Counter((label, bisect_left(cuts, area) + 1) for label, area in train_areas)

    validation = verified_rows(dataset / "validation.jsonl", hashes["validation"])
    maps, endpoints, _ = load_predictions()
    if {r["id"] for r in maps} != validation.keys():
        raise ValueError("Validation predictions do not match the area measurement sample")
    measured = []
    bins = {}
    for key, ref in validation.items():
        yellow, valid = area_counts(dataset, ref)
        bins[key] = bisect_left(cuts, Fraction(yellow, valid)) + 1
        measured.append(dict(id=key, label=ref["label"], yellow_pixels=yellow,
                             wafer_pixels=valid, bin=bins[key]))

    summary = checkpoint_bins(maps, endpoints, bins, receipt["labels"])
    output.mkdir(exist_ok=True, parents=True)
    fit = dict(train_manifest_sha256=hashes["train"], training_maps=len(ordered),
               rule="nearest-rank quartiles, upper endpoint inclusive",
               cut_fractions=[str(x) for x in cuts],
               train_bin_sizes=[train_counts[k] for k in range(1, 5)],
               training_maps_per_class_by_bin={
                   label: [per_class[label, k] for k in range(1, 5)] for label in receipt["labels"]})
    (output / "fit.json").write_text(json.dumps(fit, indent=2) + "\n")
    write_csv(output / "validation.csv", measured[0].keys(), measured)
    write_csv(output / "checkpoints.csv", summary[0].keys(), summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path, help="Prepared wm811k-lot17 directory")
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    run(args.dataset, args.output)
