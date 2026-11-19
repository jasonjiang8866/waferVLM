"""Compare the six frozen Qwen checkpoints on their first validation pass."""
import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "docs/evidence/qwen-validation-first-pass"
REFERENCE = ROOT / "docs/showcase/dinov3-seed17/validation.jsonl"


def verified_rows(path, digest):
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != digest:
        raise ValueError(f"Unexpected SHA-256 for {path.name}")
    rows = [json.loads(line) for line in data.splitlines()]
    indexed = {row["id"]: row for row in rows}
    if len(indexed) != len(rows):
        raise ValueError(f"Duplicate sample ID in {path.name}")
    return indexed


def load_predictions():
    config = json.loads((ROOT / "configs/seed-confirmation.json").read_text())
    receipt = json.loads((ROOT / "docs/evidence/vlm-repeat-complete.json").read_text())
    reference = verified_rows(REFERENCE, config["validation_sha256"])
    if not reference or any(row["split"] != "validation" for row in reference.values()):
        raise ValueError("Expected the frozen validation split")
    maps = {key: {"id": key, "label": row["label"]} for key, row in reference.items()}
    endpoints = []
    for pair in receipt["pairs"]:
        first, = [p for p in pair["passes"] if p["repeat"] == 1]
        endpoint = f"seed{pair['seed']}-step{pair['step']}"
        if first["output"] != endpoint + "-repeat1":
            raise ValueError(f"Wrong first pass for {endpoint}")
        rows = verified_rows(RAW / (endpoint + ".jsonl"), first["predictions_sha256"])
        if rows.keys() != maps.keys():
            raise ValueError(f"Validation IDs differ for {endpoint}")
        for key, row in rows.items():
            if any(row[k] != reference[key][k] for k in ("image_sha256", "label")):
                raise ValueError(f"Validation identity differs at {endpoint}/{key}")
            maps[key][endpoint] = row["prediction"]
        endpoints.append(endpoint)
    expected = [f"seed{s}-step{t}" for s in config["seeds"] for t in config["endpoints"]]
    if sorted(endpoints) != sorted(expected):
        raise ValueError("Incomplete frozen checkpoint matrix")
    return list(maps.values()), expected, config["seeds"]


def summarize(maps, endpoints, seeds):
    overall = Counter()
    by_class = {}
    wrong_consensus = Counter()
    transitions = {seed: Counter() for seed in seeds}
    for row in maps:
        label = row["label"]
        guesses = [row[e] for e in endpoints]
        correct = guesses.count(label)
        if correct == len(endpoints):
            group = "all_correct"
        elif correct:
            group = "some_correct"
        elif len(set(guesses)) == 1:
            group = "all_wrong_same_prediction"
            wrong_consensus[(label, guesses[0])] += 1
        else:
            group = "all_wrong_different_predictions"
        overall[group] += 1
        by_class.setdefault(label, Counter())[group] += 1
        for seed in seeds:
            early, late = (row[f"seed{seed}-step{step}"] for step in (200, 250))
            if early == late:
                change = "unchanged"
            elif early == label:
                change = "correct_to_wrong"
            elif late == label:
                change = "wrong_to_correct"
            else:
                change = "wrong_to_other_wrong"
            transitions[seed][change] += 1
    return dict(maps=len(maps), overall=overall, by_class=by_class, transitions=transitions,
                unanimous_errors=[dict(label=a, prediction=b, count=n)
                                  for (a, b), n in wrong_consensus.most_common()])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, help="Write all 160 sample IDs and six decoded labels")
    args = parser.parse_args()
    maps, endpoints, seeds = load_predictions()
    if args.csv:
        with args.csv.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=["id", "label", *endpoints])
            writer.writeheader()
            writer.writerows(maps)
    print(json.dumps(summarize(maps, endpoints, seeds), indent=2))


if __name__ == "__main__":
    main()
