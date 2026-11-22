"""Admission and independent evidence checks for the one-epoch Qwen Q2 pilot."""
import base64
import hashlib
import io
import json
import lzma
import math
import os
import re
import tempfile
import zipfile
from pathlib import Path

from runpod.provider import REPOSITORY, finite, request

ROOT = Path(__file__).resolve().parents[1]
MARKER = "qwen-q2-pair-pilot-paid-v1"
WORK_SECONDS = 90 * 60
PRIOR_USD, PRIOR_MINUTES = 11.4232, 791.23  # round up the completed Qwen holdout ledger
FILES = sorted(["analysis/qwen_q2_pilot.py", "analysis/qwen_loc_review.py",
                "analysis/qwen_validation_errors.py", "runpod/__init__.py", "runpod/q2_worker.py", "runpod/vlm_worker.py",
                "scripts/prepare_public_data.py", "docs/evidence/wm811k-lot17.json",
                "docs/evidence/qwen-validation-area/fit.json", "docs/evidence/qwen-validation-area/validation.csv",
                "docs/evidence/qwen-validation-area/training-pairs.json"] +
               [str(p.relative_to(ROOT)) for p in (ROOT / "src/wafervlm").glob("*.py")])


def enabled():
    return os.environ.get("WAFER_VLM_PHASE") == "q2-pilot"


def previous_attempt():
    return json.loads((ROOT / "runpod/q2-startup-receipt.json").read_text())


def recovery_ledger():
    from runpod.vlm_ledger import artifact_bytes, zip_json
    receipt = previous_attempt()
    api = f"https://api.github.com/repos/{REPOSITORY}/actions/runs/"
    for run_id, workflow, outcome, event in (
            (receipt["run_id"], "qwen-q2", "failure", "workflow_dispatch"),
            (receipt["independent_run_id"], "qwen-q2-cleanup", "success", "workflow_run")):
        run = request(api + str(run_id), token=os.environ["GH_TOKEN"])
        expected = dict(path=f".github/workflows/{workflow}.yml", conclusion=outcome,
                        run_attempt=1, event=event, head_branch="main", status="completed")
        if workflow == "qwen-q2":
            expected["head_sha"] = receipt["controller_sha"]
        if any(run.get(k) != v for k, v in expected.items()):
            raise RuntimeError("Q2 startup or independent cleanup run identity mismatch")
    primary = artifact_bytes(receipt["result_id"], receipt["result_sha256"], 256_000)
    independent = artifact_bytes(receipt["independent_id"], receipt["independent_sha256"], 64_000)
    with zipfile.ZipFile(io.BytesIO(primary)) as archive:
        if {"experiment.tar.gz", "result.json", "acceptance.json"}.intersection(archive.namelist()):
            raise RuntimeError("Prior Q2 attempt produced a worker result")
    spec = zip_json(primary, "launch-spec.json")
    cleanup = zip_json(primary, "cleanup.json")
    other = zip_json(independent, "cleanup.json")
    state = zip_json(primary, "provision.json")
    # This artifact is emitted only when /result was never reached before timeout.
    zip_json(primary, "startup-timeout-diagnostics.json")
    name = f"wafer-20261107-usd20-q2pilot-{receipt['run_id']}"
    if (any(r.get("status") != "verified_absent" or r.get("name") != name
            or r.get("consecutive_absent_checks", 0) < 3 for r in (cleanup, other))
            or not state.get("pod_id") or state.get("name") != name
            or spec.get("controller_sha") != receipt["controller_sha"] or spec.get("phase") != "q2-pair-pilot"
            or other["checked_epoch"] < cleanup["checked_epoch"]
            or state["cost"]["prior_spend_reserve_usd"] != PRIOR_USD
            or state["cost"]["prior_runtime_reserve_seconds"] != PRIOR_MINUTES * 60
            or cleanup.get("creation_confirmed") is not True
            or cleanup.get("within_budget") is not True or cleanup.get("within_runtime") is not True
            or cleanup.get("persistent_volumes_created") is not False):
        raise RuntimeError("Q2 startup and deletion receipts do not reconcile")
    elapsed = cleanup["elapsed_billing_bound_s"]
    rate = state["cost"]["gpu_hourly_usd"] + state["cost"]["disk_hourly_usd"]
    if (not finite(elapsed, .001) or not finite(rate, .001)
            or abs(cleanup["estimated_total_seconds"] - (PRIOR_MINUTES * 60 + elapsed)) > 1e-6
            or abs(cleanup["estimated_total_usd"] - (PRIOR_USD + elapsed * rate / 3600)) > 1e-8):
        raise RuntimeError("Prior Q2 cost or duration does not reconcile")
    return {"attempts": [receipt["run_id"]], "usd": cleanup["estimated_total_usd"],
            "seconds": cleanup["estimated_total_seconds"]}


def prices(rate, disk_hourly, ledger):
    spent_seconds = ledger["seconds"] - PRIOR_MINUTES * 60
    spent_usd = ledger["usd"] - PRIOR_USD
    if not finite(spent_seconds, .001) or not finite(spent_usd, .000001):
        raise ValueError("Q2 prior allocation must be accounted for")
    work = WORK_SECONDS - math.ceil(spent_seconds)
    total = spent_usd + (rate + disk_hourly) * (work + 600) / 3600
    if work <= 0 or not finite(rate, .001) or total > 5:
        raise ValueError("Q2 pilot quote exceeds remaining sub-budget")
    return dict(gpu_hourly_usd=rate, disk_hourly_usd=disk_hourly, work_seconds=work,
                cleanup_seconds=600, prior_spend_reserve_usd=ledger["usd"],
                prior_runtime_reserve_seconds=ledger["seconds"], pilot_cap_usd=5,
                pilot_cap_minutes=100, planned_pilot_usd=total)


def guard(run):
    token = os.environ["GH_TOKEN"]
    receipt = previous_attempt()
    url = f"https://api.github.com/repos/{REPOSITORY}/actions/artifacts?name={MARKER}&per_page=100"
    listing = request(url, token=token)
    items = listing.get("artifacts", [])
    if listing.get("total_count") != 2 or len(items) != 2:
        raise RuntimeError("Expected only the reconciled startup and this Q2 intent")
    prior_id = receipt["marker_id"]
    current_id = int(os.environ["AUTHORIZATION_ARTIFACT_ID"])
    expected = {prior_id: (str(receipt["run_id"]), receipt["controller_sha"]),
                current_id: (str(run), os.environ["EXPECTED_SHA"])}
    if len(expected) != 2 or {i["id"] for i in items} != expected.keys():
        raise RuntimeError("Unexpected Q2 pilot paid marker")
    for item in items:
        owner = item["workflow_run"]
        if (item["name"] != MARKER or item["expired"] is not False
                or (str(owner["id"]), owner["head_sha"]) != expected[item["id"]]):
            raise RuntimeError("Q2 pilot marker belongs to another run")


def bootstrap(sha):
    if not re.fullmatch("[a-f0-9]{40}", sha):
        raise ValueError("Invalid reviewed SHA")
    # Transport checkout bytes in the same single-line form as the accepted DINO worker.
    payload = json.dumps({name: (ROOT / name).read_text() for name in FILES},
                         sort_keys=True).encode()
    packed = base64.b64encode(lzma.compress(payload)).decode()
    digest = hashlib.sha256(payload).hexdigest()
    program = f"""import base64, hashlib, hmac, json, lzma, os, pathlib, runpy, sys
if os.environ['WAFER_CONTROLLER_SHA'] != {sha!r}:
    raise ValueError('Q2 controller revision mismatch')
raw = lzma.decompress(base64.b64decode({packed!r}))
if not hmac.compare_digest(hashlib.sha256(raw).hexdigest(), {digest!r}):
    raise ValueError('Q2 source bundle checksum mismatch')
root = pathlib.Path('/workspace/q2-source')
root.mkdir(parents=True, exist_ok=True)
for name, content in json.loads(raw).items():
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
os.chdir(root)
sys.path[:0] = [str(root), str(root/'src')]
os.environ['PYTHONPATH'] = str(root/'src')
runpy.run_path(str(root/'runpod/q2_worker.py'), run_name='__main__')
"""
    # repr escapes line breaks before the controller quotes the entire shell argument.
    # The worker still refuses to overwrite its output or rerun a training attempt.
    return "exec(" + repr(program) + ")"


def verify(archive, spec):
    import sys
    sys.path.insert(0, str(ROOT / "analysis"))
    from qwen_q2_pilot import pair_epoch, summarize
    from wafervlm.core import sha256_file
    from runpod.confirmation_verify import unpack
    with tempfile.TemporaryDirectory() as tmp:
        root = unpack(archive, Path(tmp) / "unpacked")
        def load(path):
            return json.loads((root / path).read_text())
        frozen = json.loads((ROOT / "docs/evidence/wm811k-lot17.json").read_text())
        for split in ("train", "validation"):
            if sha256_file(root / f"{split}.jsonl") != frozen["dataset"]["manifest_sha256"][split]:
                raise ValueError("Pilot dataset mismatch")
        pairs = load("pairs.json")
        if pairs != json.loads((ROOT / "docs/evidence/qwen-validation-area/training-pairs.json").read_text()):
            raise ValueError("Training pairs changed")
        control = load("control/train/epoch-order.json")
        ids = [json.loads(s)["id"] for s in (root / "train.jsonl").read_text().splitlines()]
        if sorted(ids) != sorted(control) or load("matched/train/epoch-order.json") != pair_epoch(control, pairs):
            raise ValueError("Exposure mismatch between arms")
        initialization, environment = [], []
        for arm in ("control", "matched"):
            train = load(f"{arm}/train/training.json")
            if (any(train[k] != 99 for k in ("gradient_checks", "nonzero_gradient_checks", "end_step", "executed_steps"))
                    or train["start_step"] != 0 or not train["frozen_parameters_verified"]
                    or not finite(train["peak_allocated_bytes"], 1)):
                raise ValueError("Training evidence mismatch")
            initialization.append(train["initial_trainable_sha256"])
            training = load(f"{arm}/train/run.json")
            evaluation = load(f"{arm}/validation/run.json")
            if training["config"] != dict(steps=99, seed=17, accumulation=8, rank=8, learning_rate=.0002,
                                           max_length=1024, device="cuda") or training["resume"] is not None:
                raise ValueError("Pilot training recipe changed")
            if evaluation["adapter_sha256"] != sha256_file(root / arm / "train/adapter/adapter_model.safetensors"):
                raise ValueError("Scored adapter mismatch")
            if evaluation["config"] != dict(split="validation", adapter=True, device="cuda", max_new_tokens=16,
                                           include_loss=False, seed=17, do_sample=False, enable_thinking=False):
                raise ValueError("Pilot evaluation recipe changed")
            for r in (training, evaluation):
                if (r["source_sha"] != spec["controller_sha"] or r["image_digest"] != spec["image"].split("@")[1]
                        or r["dataset_sha256"] != frozen["dataset_sha256"] or not r["gpu"]):
                    raise ValueError("Pilot source, image or device mismatch")
            environment.append([evaluation[k] for k in ("gpu","cuda","driver","model_revision","base_weights_sha256",
                                                         "processor_identity","packages","generation")])
        if initialization[0] != initialization[1] or environment[0] != environment[1]:
            raise ValueError("Initial weights or evaluation environments differ")
        report = summarize(root, root)
        if report != load("comparison.json"):
            raise ValueError("Retrieved raw predictions disagree with worker metrics")
        return report
