"""Admission and independent evidence checks for the one-epoch Qwen Q2 pilot."""
import hashlib
import json
import os
import re
import tempfile
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


def prices(rate, disk_hourly):
    hours = (WORK_SECONDS + 600) / 3600
    if not finite(rate, .001) or (rate + disk_hourly) * hours > 5:
        raise ValueError("Q2 pilot quote exceeds sub-budget")
    return dict(gpu_hourly_usd=rate, disk_hourly_usd=disk_hourly, work_seconds=WORK_SECONDS,
                cleanup_seconds=600, prior_spend_reserve_usd=PRIOR_USD,
                prior_runtime_reserve_seconds=PRIOR_MINUTES * 60, pilot_cap_usd=5,
                pilot_cap_minutes=100, planned_pilot_usd=(rate + disk_hourly) * hours)


def guard(run):
    token = os.environ["GH_TOKEN"]
    url = f"https://api.github.com/repos/{REPOSITORY}/actions/artifacts?name={MARKER}&per_page=100"
    listing = request(url, token=token)
    items = listing.get("artifacts", [])
    if listing.get("total_count") != 1 or len(items) != 1:
        raise RuntimeError("Missing or previously consumed Q2 pilot budget marker")
    item = items[0]
    if (item["name"] != MARKER or item["expired"] or str(item["id"]) != os.environ["AUTHORIZATION_ARTIFACT_ID"]
            or str(item["workflow_run"]["id"]) != run or item["workflow_run"]["head_sha"] != os.environ["EXPECTED_SHA"]):
        raise RuntimeError("Q2 pilot marker belongs to another run")


def bootstrap(sha):
    if not re.fullmatch("[a-f0-9]{40}", sha):
        raise ValueError("Invalid reviewed SHA")
    # The Pod downloads only the reviewed files. Resolve and verify every byte before importing the suite.
    hashes = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in FILES}
    return f"""import hashlib, os, pathlib, runpy, sys, urllib.request
root = pathlib.Path('/workspace/q2-source')
root.mkdir(exist_ok=False)
for name, digest in {hashes!r}.items():
    url = 'https://raw.githubusercontent.com/{REPOSITORY}/{sha}/' + name
    with urllib.request.urlopen(url, timeout=30) as stream:
        data = stream.read(2000001)
    if len(data) > 2000000 or hashlib.sha256(data).hexdigest() != digest:
        raise ValueError('Q2 source checksum mismatch')
    path = root / name
    path.parent.mkdir(exist_ok=True, parents=True)
    path.write_bytes(data)
os.chdir(root)
sys.path[:0] = [str(root), str(root/'src')]
os.environ['PYTHONPATH'] = str(root/'src')
runpy.run_path(str(root/'runpod/q2_worker.py'), run_name='__main__')
"""


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
