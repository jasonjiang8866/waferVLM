"""Run the single-seed matched-area order ablation on the frozen development split."""
import argparse
import csv
import gc
import json
from fractions import Fraction
from pathlib import Path

from qwen_loc_review import area_counts
from qwen_validation_errors import ROOT, verified_rows
from wafervlm.core import LABELS, metrics, parse_label, read_jsonl, sha256_file, write_json

FIT = ROOT / "docs/evidence/qwen-validation-area/fit.json"
DATA = ROOT / "docs/evidence/wm811k-lot17.json"
PEERS = ("center", "scratch", "edge-loc")
ACCUMULATION = 8


def training_pairs(dataset):
    fit = json.loads(FIT.read_text())
    rows = verified_rows(dataset / "train.jsonl", fit["train_manifest_sha256"])
    low, high = map(Fraction, fit["cut_fractions"][:2])
    pools = {label: {} for label in ("loc", *PEERS)}
    for key, row in rows.items():
        if row["label"] in pools:
            area = Fraction(*area_counts(dataset, row))
            if low < area <= high:
                pools[row["label"]][key] = area
    pairs = []
    for i, (anchor, area) in enumerate(sorted(pools["loc"].items(), key=lambda item: (item[1], item[0]))):
        label = PEERS[i % len(PEERS)]
        candidates = pools[label]
        peer = min(candidates, key=lambda key: (abs(candidates[key] - area), key))
        pairs.append(dict(anchor=anchor, peer=peer, peer_label=label, anchor_area=str(area),
                          peer_area=str(candidates.pop(peer))))
    if len(pairs) != 37:
        raise ValueError("Expected 37 frozen Q2 loc anchors")
    return pairs


def pair_epoch(control, pairs):
    position = {key: i for i, key in enumerate(control)}
    free = set(range(0, len(control) - 1, ACCUMULATION))
    if len(pairs) > len(free):
        raise ValueError("Not enough optimizer updates to separate the pairs")
    paired = [p[k] for p in pairs for k in ("anchor", "peer")]
    if len(position) != len(control) or len(set(paired)) != len(paired) or not set(paired) <= position.keys():
        raise ValueError("Pair and control identities must be unique training IDs")
    slots = {}
    for pair in sorted(pairs, key=lambda p: position[p["anchor"]]):
        i = position[pair["anchor"]]
        start = min(free, key=lambda n: (abs(i // ACCUMULATION - n // ACCUMULATION), n))
        free.remove(start)
        offset = min(i % ACCUMULATION, min(ACCUMULATION, len(control) - start) - 2)
        slots[start + offset] = pair["anchor"]
        slots[start + offset + 1] = pair["peer"]
    used = set(paired)
    rest = iter(key for key in control if key not in used)
    return [slots[i] if i in slots else next(rest) for i in range(len(control))]


def summarize(dataset, root):
    digest = json.loads(DATA.read_text())["dataset"]["manifest_sha256"]["validation"]
    reference = verified_rows(dataset / "validation.jsonl", digest)
    with (FIT.parent / "validation.csv").open() as f:
        q2 = {r["id"] for r in csv.DictReader(f) if r["bin"] == "2" and r["label"] == "loc"}
    if len(q2) != 9 or not q2 <= reference.keys():
        raise ValueError("Frozen Q2 validation subset mismatch")
    scores, calls = {}, {}
    for arm in ("control", "matched"):
        pred = read_jsonl(root / arm / "validation/predictions.jsonl")
        if [r["id"] for r in pred] != list(reference):
            raise ValueError("Incomplete or reordered validation predictions")
        for r in pred:
            if any(r[k] != reference[r["id"]][k] for k in ("image_sha256", "label")):
                raise ValueError("Validation identity mismatch")
            if parse_label(r["response"], r["truncated"]) != r["prediction"]:
                raise ValueError("Decoded label mismatch")
        m = metrics(pred)
        false = {k: m["confusion_matrix"][k]["loc"] for k in LABELS if k != "loc"}
        calls[arm] = {r["id"]: r["prediction"] for r in pred}
        scores[arm] = dict(q2_loc_hits=sum(calls[arm][k] == "loc" for k in q2), q2_loc_maps=len(q2),
                           loc_recall=m["per_class"]["loc"]["recall"], false_loc=sum(false.values()),
                           false_loc_by_class=false, accuracy=m["accuracy"], macro_f1=m["macro_f1"],
                           invalid_rate=m["invalid_rate"])
    return dict(arms=scores, delta_matched_minus_control={k: scores["matched"][k] - scores["control"][k]
                for k in ("q2_loc_hits", "loc_recall", "false_loc", "macro_f1", "accuracy", "invalid_rate")},
                q2_paired=[dict(id=k, control=calls["control"][k], matched=calls["matched"][k]) for k in sorted(q2)])


def run(dataset, root):
    from wafervlm.experiment import evaluate, seal_output, train
    if sha256_file(dataset / "dataset.json") != json.loads(DATA.read_text())["dataset_sha256"]:
        raise ValueError("Frozen dataset mismatch")
    pairs = training_pairs(dataset)
    if pairs != json.loads((FIT.parent / "training-pairs.json").read_text()):
        raise ValueError("Reconstructed pairs do not match the frozen identities")
    root.mkdir(parents=True, exist_ok=False)
    write_json(root / "pairs.json", pairs)
    init = None
    order = None
    for arm in ("control", "matched"):
        work = root / arm
        trained = train(dataset, work / "train", steps=99, accumulation=ACCUMULATION,
                        seed=17, device="cuda", record_epoch=True, epoch_order=order)
        diagnostics = json.loads((trained / "training.json").read_text())
        if diagnostics["nonzero_gradient_checks"] != 99 or not diagnostics["peak_allocated_bytes"]:
            raise ValueError("Missing CUDA training or gradient evidence")
        if arm == "control":
            init = diagnostics["initial_trainable_sha256"]
            control = json.loads((trained / "epoch-order.json").read_text())
            order = control, pair_epoch(control, pairs)
        elif init != diagnostics["initial_trainable_sha256"]:
            raise ValueError("Arms started from different trainable weights")
        gc.collect()
        evaluate(dataset, work / "validation", split="validation", adapter=trained / "adapter")
        gc.collect()
    report = summarize(dataset, root)
    write_json(root / "comparison.json", report)
    seal_output(root)
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("pairs", "run", "score"))
    p.add_argument("dataset", type=Path)
    p.add_argument("output", type=Path)
    args = p.parse_args()
    if args.mode == "pairs":
        write_json(args.output, training_pairs(args.dataset))
    elif args.mode == "score":
        print(json.dumps(summarize(args.dataset, args.output), indent=2))
    else:
        run(args.dataset, args.output)
