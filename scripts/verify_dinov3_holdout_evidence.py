"""Read-only DINOv3 holdout evidence audit. Stdlib only; no network or GPU."""
import argparse
import hashlib
import io
import json
import math
import tarfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FREEZE = "docs/evidence/dinov3-six-endpoints.json"
RECEIPT = "docs/evidence/dinov3-holdout-complete.json"
LEDGER = "runpod/dinov3-study-ledger.json"
REPORT = "docs/showcase/dinov3-confirmation-holdout/analysis.json"
CHECKSUMS = "docs/showcase/dinov3-confirmation-holdout/checksums.json"
RETENTION = "docs/evidence/dinov3-holdout-retention.json"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(root, relative):
    return json.loads((Path(root) / relative).read_text(encoding="utf-8"))


def digest(data):
    return hashlib.sha256(data).hexdigest()


def equal_number(first, second):
    return math.isclose(first, second, abs_tol=1e-9, rel_tol=0)


def verify_repository(root=ROOT):
    """Check the committed freeze, results, archival pins, ledger and cleanup chain."""
    root = Path(root)
    frozen, receipt, ledger = (read(root, p) for p in (FREEZE, RECEIPT, LEDGER))
    report, retention = (read(root, p) for p in (REPORT, RETENTION))
    run_id = str(receipt["run_id"])
    require(retention["schema"] == 1 and retention["run_id"] == run_id,
            "Retention identity changed")
    require(retention["freeze_commit_sha"] == receipt["freeze_commit_sha"],
            "Freeze commit differs")
    freeze_hash = digest((root / FREEZE).read_bytes())
    require(freeze_hash == receipt["freeze_sha256"] == retention["freeze_sha256"]
            == receipt["launch_spec"]["frozen_sha256"], "Frozen protocol hash differs")
    require(frozen["holdout_predictions_read"] is False,
            "Freeze must precede DINO holdout predictions")
    endpoints = frozen["endpoints"]
    weights = {e["name"]: e["weight_sha256"] for e in endpoints}
    require(len(weights) == len(endpoints) == 6
            and weights == retention["selected_exports"], "Frozen six weights differ")
    require(report["status"] == "accepted" and report["phase"] == "holdout"
            and report["endpoints"] == endpoints, "Published report/freeze mismatch")
    require(report["holdout_predictions"] == receipt["holdout_predictions_this_run"] == 888
            and report["validation_passes"] == receipt["validation_passes_this_run"] == 0,
            "Holdout/validation counts differ")
    spec = receipt["launch_spec"]
    require(report["execution"]["run_id"] == run_id
            and report["execution"]["controller_sha"] == receipt["controller_sha"]
            and spec["run_id"] == run_id and spec["controller_sha"] == receipt["controller_sha"]
            and spec["phase"] == "holdout", "Controller/phase identity differs")
    for key in ("image", "image_source_sha", "plan_sha256", "overlay_sha256"):
        require(spec[key] == report["execution"][key], f"Launch/report {key} differs")
    require(spec["plan_sha256"] == frozen["plan_sha256"]
            and spec["input_artifact"] == frozen["artifact"],
            "Frozen input and plan differ")
    require(receipt["conclusion"] == "success"
            and ledger["approved_usd"] == 10 and ledger["approved_seconds"] == 30000,
            "Study status or budget cap differs")
    attempts = ledger["attempts"]
    ids = [str(a["run_id"]) for a in attempts]
    require(len(attempts) >= 2 and len(ids) == len(set(ids))
            and ids[-1] == run_id, "Ledger history changed or reused")
    last = attempts[-1]
    for key in ("run_id", "controller_sha", "conclusion", "artifact_id",
                "zip_sha256", "cleanup", "independent"):
        require(last[key] == receipt[key], f"Ledger/receipt {key} differs")
    first = receipt["cleanup"]
    other = receipt["independent"]["cleanup"]
    for name, result in (("primary", first), ("independent", other)):
        require(result["status"] == "verified_absent" and str(result["run_id"]) == run_id
                and result["persistent_volumes_created"] is False,
                f"{name} cleanup does not prove owned Pod absence")
    require(first["checks"] >= 3 and other["checks"] >= 12,
            "Insufficient consecutive inventory absence checks")
    require(str(receipt["independent"]["run_id"]) != run_id
            and other["checked_epoch"] > first["checked_epoch"],
            "Independent cleanup must follow primary and be a separate run")
    require(first["within_cap"] is True
            and 0 < first["estimated_usd"] < ledger["approved_usd"]
            and 0 < first["elapsed_billing_bound_s"] < ledger["approved_seconds"],
            "Unbounded allocation cost/duration")
    previous = attempts[-2]["cleanup"]
    require(equal_number(previous["cumulative_estimated_usd"], spec["ledger"]["usd"])
            and equal_number(previous["cumulative_runtime_bound_s"], spec["ledger"]["seconds"])
            and equal_number(first["cumulative_estimated_usd"],
                             spec["ledger"]["usd"] + first["estimated_usd"])
            and equal_number(first["cumulative_runtime_bound_s"],
                             spec["ledger"]["seconds"] + first["elapsed_billing_bound_s"]),
            "Cumulative cost/duration chain differs")
    require(first["cumulative_estimated_usd"] < ledger["approved_usd"]
            and first["cumulative_runtime_bound_s"] < ledger["approved_seconds"],
            "Cumulative budget cap exceeded")
    for name, aid, hash_value in (
        ("primary", receipt["artifact_id"], receipt["zip_sha256"]),
        ("independent_cleanup", receipt["independent"]["artifact_id"],
         receipt["independent"]["zip_sha256"]),
    ):
        entry = retention["artifacts"][name]
        require(entry["artifact_id"] == aid and entry["sha256"] == hash_value,
                f"Retention {name} digest differs")
    require(retention["inner_tar_sha256"] == receipt["inner_sha256"],
            "Inner TAR hash differs")
    intent = retention["artifacts"]["intent"]
    require(intent["artifact_id"] == 14681916299 and
            intent["sha256"] ==
            "4de7fdeacf0f44f190329dd7292a22e295ef57b5ae537ed672124df0af10b1e8",
            "Original dispatch intent digest differs")
    mirror = retention["durable_copy"]
    require(mirror["status"] in ("pending", "verified"), "Unknown durable-copy state")
    if mirror["status"] == "verified":
        require(bool(mirror.get("location")) and bool(mirror.get("verified_at")),
                "Verified durable copy must include location and read-back timestamp")
    else:
        require(mirror.get("location") is None,
                "Pending durable copy must not claim a location")
    return {"run_id": run_id, "models": 6, "holdout_predictions": 888,
            "independent_cleanup": "verified_absent",
            "durable_copy_status": mirror["status"]}


def verify_archives(root, primary, cleanup, intent=None):
    """Inspect original ZIP bytes including inner selected weights and prediction hashes."""
    summary = verify_repository(root)
    receipt, retention, frozen = (read(root, p) for p in (RECEIPT, RETENTION, FREEZE))
    report, checksums = (read(root, p) for p in (REPORT, CHECKSUMS))
    for name, filename in (("primary", primary), ("independent_cleanup", cleanup)):
        require(digest(Path(filename).read_bytes()) == retention["artifacts"][name]["sha256"],
                f"{name} original ZIP SHA-256 differs")
    with zipfile.ZipFile(primary) as archive:
        require(json.loads(archive.read("cleanup.json")) == receipt["cleanup"],
                "Primary ZIP cleanup differs")
        require(json.loads(archive.read("launch-spec.json")) == receipt["launch_spec"],
                "Original launch-spec differs")
        accepted = json.loads(archive.read("acceptance.json"))
        for key in ("endpoints", "holdout_predictions", "lot_group_percentile_95",
                    "seed17_lora_minus_qwen", "models", "qwen_published_sha256"):
            require(accepted[key] == report[key],
                    f"Original acceptance {key} differs from Git")
        inner = archive.read("experiment.tar.gz")
        require(digest(inner) == receipt["inner_sha256"], "Original inner TAR differs")
    with zipfile.ZipFile(cleanup) as archive:
        require(json.loads(archive.read("cleanup.json")) ==
                receipt["independent"]["cleanup"], "Independent cleanup ZIP differs")
    if intent is not None:
        require(digest(Path(intent).read_bytes()) ==
                retention["artifacts"]["intent"]["sha256"],
                "Original dispatch-intent ZIP SHA-256 differs")
        with zipfile.ZipFile(intent) as archive:
            row = json.loads(archive.read("intent.json"))
            require(row["run_id"] == summary["run_id"]
                    and row["controller_sha"] == receipt["controller_sha"]
                    and row["frozen_sha256"] == receipt["freeze_sha256"],
                    "Dispatch intent identity differs")
    with tarfile.open(fileobj=io.BytesIO(inner), mode="r:gz") as archive:
        require(json.load(archive.extractfile("results/endpoints.json")) ==
                frozen["endpoints"], "Selected endpoint manifest differs")
        for ep in frozen["endpoints"]:
            name = ep["name"]
            weight = archive.extractfile(
                f"results/models/{name}/selected.safetensors").read()
            require(digest(weight) == ep["weight_sha256"], f"{name} weight SHA-256 differs")
            relpath = f"predictions/{name}/repeat1/predictions.jsonl"
            predictions = archive.extractfile(
                f"results/holdout/{name}/repeat1/predictions.jsonl").read()
            require(digest(predictions) == checksums[relpath],
                    f"{name} prediction SHA-256 differs")
            require(len(predictions.splitlines()) == 148, f"{name} count differs")
    return dict(summary, original_zip_verified=2 + int(intent is not None),
                selected_weights_verified=6)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--primary", type=Path)
    parser.add_argument("--cleanup", type=Path)
    parser.add_argument("--intent", type=Path)
    args = parser.parse_args()
    require((args.primary is None) == (args.cleanup is None),
            "--primary and --cleanup are required together")
    require(args.intent is None or args.primary is not None,
            "--intent needs --primary and --cleanup")
    result = (verify_archives(args.root, args.primary, args.cleanup, args.intent)
              if args.primary else verify_repository(args.root))
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
