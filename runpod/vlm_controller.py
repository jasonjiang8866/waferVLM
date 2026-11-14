"""One seed-42 VLM training lifecycle; shares the existing $30 / 1100-minute ledger.

Transport and ownership cleanup follow controller.py. Kept separately so DINOv3
and the historical continuation workflow cannot silently change this workload.
"""

import argparse
import base64
import hashlib
import io
import json
import math
import os
import re
import secrets
import shlex
import signal
import subprocess
import time
import urllib.parse
import urllib.request
from pathlib import Path

from runpod.provider import (
    MIG,
    REPOSITORY,
    NoRedirect,
    RemoteHTTPError,
    Runpod,
    budget,
    finite,
    request,
)

SOURCE = "11d2e5f088b350dd9391864dbe2869417fff5af7"
IMAGE = "ghcr.io/jasonjiang8866/wafervlm@sha256:a6d31b4f12c2c3ee64c59d08b1e67c89e9bbf9b46cbdb4179417f1d1efa3fa79"

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runpod-results"
# Original GPU/rate authorization retained; user added $10 / 600 minutes on 2026-11-09.
GPU_CAPS = {
    MIG: 0.69,
    "NVIDIA GeForce RTX 4090": 0.89,
    "NVIDIA GeForce RTX 3090": 0.50,
    "NVIDIA GeForce RTX 5090": 1.19,
}
MIN_CUDA_BY_GPU = {name: "12.8" for name in GPU_CAPS}
GPUS = ("NVIDIA GeForce RTX 4090", MIG, "NVIDIA GeForce RTX 3090", "NVIDIA GeForce RTX 5090")
OWNER = "WAFER_MANAGED_BY"
AUTHORIZATION = "wafer-20261107-usd20"
AUTHORIZATION_EXPIRES = 1794873600  # 2026-11-17 UTC; bounded continuation window
DISK_GB = 60
DISK_HOURLY = DISK_GB * 0.10 / (28 * 24)
CLEANUP_SECONDS = 600
MAX_WORK_SECONDS = 185 * 60  # per attempt; below the Actions job limit
APPROVED_TOTAL_SECONDS = 1100 * 60
PRIOR_RUNTIME_RESERVE_SECONDS = 491.911 * 60  # seven reconciled attempts, rounded UP
MAX_RATE = max(GPU_CAPS.values())
MAX_ARCHIVE = 2 * 1024**3
MARKER = "runpod-authorization-" + AUTHORIZATION
PRIOR_SPEND_RESERVE = 5.903039  # cumulative conservative estimate, rounded UP
APPROVED_USD = 30.0


def save(name, value):
    OUT.mkdir(exist_ok=True)
    temporary = OUT / (name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(OUT / name)


def identity():
    if os.environ.get("GITHUB_REPOSITORY") != REPOSITORY:
        raise RuntimeError("Unexpected repository")
    run = os.environ.get("GITHUB_RUN_ID", "")
    if not run.isdigit():
        raise RuntimeError("Invalid Actions run ID")
    return run


def quote(rate):
    if not finite(rate, 0.001) or rate > MAX_RATE:
        raise ValueError("GPU quote exceeds authorized maximum")
    hourly = rate + DISK_HOURLY
    work = min(MAX_WORK_SECONDS, APPROVED_TOTAL_SECONDS - PRIOR_RUNTIME_RESERVE_SECONDS - CLEANUP_SECONDS, math.floor((APPROVED_USD - PRIOR_SPEND_RESERVE - 2) / hourly * 3600) - CLEANUP_SECONDS)
    if work < 600:
        raise ValueError("Insufficient budget or aggregate runtime")
    estimate = PRIOR_SPEND_RESERVE + hourly * (work + CLEANUP_SECONDS) / 3600
    return {"gpu_hourly_usd": rate, "disk_hourly_usd": DISK_HOURLY,
            "work_seconds": work, "cleanup_seconds": CLEANUP_SECONDS,
            "planned_total_usd": estimate, "approved_usd": APPROVED_USD,
            "prior_spend_reserve_usd": PRIOR_SPEND_RESERVE,
            "prior_runtime_reserve_seconds": PRIOR_RUNTIME_RESERVE_SECONDS,
            "approved_total_seconds": APPROVED_TOTAL_SECONDS,
            "planned_total_seconds": PRIOR_RUNTIME_RESERVE_SECONDS + work + CLEANUP_SECONDS,
            "unallocated_reserve_usd": APPROVED_USD - estimate,
            "transfer_usd": 0, "storage_gb": DISK_GB,
            "pricing_source": "https://docs.runpod.io/pods/pricing",
            "note": "Conservative estimate, not a provider hard cap. Cleanup failure can overrun."}


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def verify_image_manifest():
    # Public GHCR pull token is short-lived, anonymous, and never printed or persisted.
    with urllib.request.urlopen("https://ghcr.io/token?service=ghcr.io&scope=repository:jasonjiang8866/wafervlm:pull", timeout=30) as r:
        token = json.load(r)["token"]
    def fetch(kind, digest):
        if not re.fullmatch(r'sha256:[0-9a-f]{64}', digest):
            raise RuntimeError('Invalid image object digest')
        req = urllib.request.Request(f'https://ghcr.io/v2/jasonjiang8866/wafervlm/{kind}/{digest}',
            headers={'Authorization': 'Bearer ' + token,
                     'Accept': 'application/vnd.oci.image.index.v1+json,application/vnd.oci.image.manifest.v1+json,application/vnd.docker.distribution.manifest.v2+json'})
        # Only an anonymous scoped pull token is used here, never account credentials.
        with urllib.request.urlopen(req, timeout=30) as response:
            raw = response.read(8_000_001)
        if len(raw) > 8_000_000 or 'sha256:' + hashlib.sha256(raw).hexdigest() != digest:
            raise RuntimeError('Image object digest mismatch')
        return json.loads(raw)
    manifest = fetch('manifests', IMAGE.split('@')[1])
    if 'manifests' in manifest:
        matches = [m for m in manifest['manifests']
                   if m.get('platform', {}).get('os') == 'linux'
                   and m['platform'].get('architecture') == 'amd64']
        if len(matches) != 1:
            raise RuntimeError('Expected exactly one linux/amd64 image')
        manifest = fetch('manifests', matches[0]['digest'])
    config = fetch('blobs', manifest['config']['digest'])
    if ('WAFERVLM_SOURCE_SHA=' + SOURCE not in config.get('config', {}).get('Env', [])
            or config.get('architecture') != 'amd64' or config.get('os') != 'linux'):
        raise RuntimeError('Image source/platform mismatch')
    return True


def provenance():
    head = git("rev-parse", "HEAD")
    if not re.fullmatch(r"[a-f0-9]{40}", head):
        raise RuntimeError("Invalid source revision")
    # The pinned image runs this exact workload; controller additions may differ.
    changed = git(
        "diff",
        "--name-only",
        SOURCE,
        head,
        "--",
        "src", "scripts", "configs", "uv.lock", "pyproject.toml", "runpod/Dockerfile.confirmation",
    )
    if changed:
        raise RuntimeError("Current workload differs from pinned image; review image/source pin")
    config = json.loads((ROOT / "configs/extension.json").read_text())
    holdout = ROOT / "docs/evidence/final-holdout-v1"
    reservation = json.loads((holdout / "reservation.json").read_text())
    if (hashlib.sha256((holdout / "manifest.jsonl").read_bytes()).hexdigest()
            != config["reserved_holdout_manifest_sha256"]
            or reservation["manifest_sha256"] != config["reserved_holdout_manifest_sha256"]
            or reservation["evaluated"] is not False):
        raise RuntimeError("Unseen holdout reservation is missing or changed")
    return {
        "controller_sha": head,
        "image_source_sha": SOURCE,
        "image": IMAGE,
        "anonymous_manifest_verified": verify_image_manifest(),
        "suite": "seed42-independent-confirmation",
        "config_sha256": hashlib.sha256(
            (ROOT / "configs/seed-confirmation.json").read_bytes()
        ).hexdigest(),
        "worker_sha256": hashlib.sha256((ROOT / "runpod/vlm_worker.py").read_bytes()).hexdigest(),
    }


def preflight(api):
    identity()
    evidence = provenance()
    from runpod.vlm_ledger import reconcile
    evidence["ledger"] = reconcile(identity(), require_current=False, output=OUT)
    pods = api.listing("/pods", "pods")
    if any(p.get("env", {}).get(OWNER) == REPOSITORY for p in pods):
        raise RuntimeError("An owned Pod already exists; cleanup and inspect before launch")
    choices, catalog = [], []
    for name in GPUS:
        minimum_cuda = MIN_CUDA_BY_GPU[name]
        entry = {"gpu": name, "min_cuda_version": minimum_cuda, "rate": None, "regions": []}
        try:
            data = api.call(
                "/catalog/gpus/"
                + urllib.parse.quote(name, safe="")
                + "?"
                + urllib.parse.urlencode(
                    {
                        "include": "AVAILABILITY",
                        "product": "POD",
                        "cloud": "SECURE",
                        "count": 1,
                        "minCudaVersion": minimum_cuda,
                    }
                )
            )
            rate = data.get("price", {}).get("secure")
            entry["rate"] = rate if finite(rate) else None
            if not finite(rate, 0.001) or rate > GPU_CAPS[name]:
                raise ValueError("Per-model approved price exceeded")
            cost = quote(rate)
            regions = sorted(
                d["id"]
                for d in data.get("dataCenters", [])
                if d.get("availability") in {"LOW", "MEDIUM", "HIGH"}
                and re.fullmatch(r"[A-Za-z0-9-]+", d.get("id", ""))
            )
            entry.update(regions=regions, status="available" if regions else "no_capacity")
            if regions:
                choices.append((rate, name, regions, cost))
        except ValueError:
            entry["status"] = "quote_rejected"
        except RuntimeError as exc:
            entry["status"] = "catalog_request_failed"
            # Preserve only a fixed HTTP code, never provider text or credentials.
            match = re.search(r"Remote HTTP ([0-9]{3});", str(exc))
            if match:
                entry["http_status"] = int(match.group(1))
        catalog.append(entry)
    # Failed read-only preflight must leave evidence, too. No admission rule changes.
    evidence.update(
        status="catalog_checked" if choices else "blocked_no_approved_gpu",
        read_only=True,
        catalog=catalog,
        checked_epoch=time.time(),
    )
    save("preflight.json", evidence)
    if not choices:
        raise RuntimeError("No approved GPU with verified price/capacity")
    rate, name, regions, cost = min(choices, key=lambda c: (GPUS.index(c[1]), c[0]))
    evidence.update(
        status="preflight_passed",
        read_only=True,
        catalog=catalog,
        gpu=name,
        min_cuda_version=MIN_CUDA_BY_GPU[name],
        regions=regions,
        cost=cost,
        budget=budget(api, reserve=APPROVED_USD),
        checked_epoch=time.time(),
    )
    save("preflight.json", evidence)
    return evidence


def authorization_guard():
    run = identity()
    if (
        os.environ.get("GITHUB_EVENT_NAME") != "workflow_dispatch"
        or os.environ.get("GITHUB_REF") != "refs/heads/main"
        or os.environ.get("GITHUB_RUN_ATTEMPT") != "1"
        or os.environ.get("CONFIRM_PAID") != "true"
        or os.environ.get("EXPECTED_SHA") != git("rev-parse", "HEAD")
        or time.time() >= AUTHORIZATION_EXPIRES
    ):
        raise RuntimeError("Manual authorization/ref/SHA/attempt/expiry guard failed")
    token = os.environ.get("GH_TOKEN", "")
    if not token:
        raise RuntimeError("Missing read-only Actions token")
    remote = request(f"https://api.github.com/repos/{REPOSITORY}/git/ref/heads/main", token=token)
    if remote.get("object", {}).get("sha") != os.environ["EXPECTED_SHA"]:
        raise RuntimeError("main moved; no paid launch")
    from runpod.vlm_ledger import reconcile
    reconcile(run, require_current=True, output=OUT)


def owned(pod, state):
    env = pod.get("env", {})
    return (
        pod.get("name") == state["name"]
        and env.get(OWNER) == REPOSITORY
        and env.get("WAFER_RUN_ID") == state["run_id"]
        and env.get("WAFER_AUTHORIZATION") == AUTHORIZATION
        and env.get("WAFER_EXPERIMENT") == "vlm-seed42"
        and (not state.get("pod_id") or pod.get("id") == state["pod_id"])
    )


def cleanup(api):
    path = OUT / "provision.json"
    run = identity()
    state = (
        json.loads(path.read_text())
        if path.exists()
        else {"name": AUTHORIZATION + "-vlm42-" + run, "run_id": run, "pod_id": None}
    )
    # Invalidate any earlier success before querying inventory again.
    save("cleanup.json", {"status": "unknown", "name": state["name"]})
    absent_checks = 0
    # Reconcile by exact name even if POST returned an unknown outcome.
    for attempt in range(8):
        pods = api.listing("/pods", "pods")
        matches = [
            p
            for p in pods
            if p.get("name") == state["name"]
            or (state.get("pod_id") and p.get("id") == state["pod_id"])
        ]
        if not matches:
            absent_checks += 1
            # A create intent without an ID gets the full reconciliation window.
            # Absence proves current inventory state, not zero historical billing.
            required = 8 if "started_epoch" in state and not state.get("pod_id") else 3
            if absent_checks >= required:
                evidence = {
                    "status": "verified_absent",
                    "name": state["name"],
                    "checked_epoch": time.time(),
                    "consecutive_absent_checks": absent_checks,
                    "creation_confirmed": bool(state.get("pod_id")),
                }
                if "started_epoch" in state:
                    elapsed = max(0, time.time() - state["started_epoch"])
                    estimate = state["cost"].get("prior_spend_reserve_usd", 0) + elapsed / 3600 * (
                        state["cost"]["gpu_hourly_usd"] + DISK_HOURLY
                    )
                    evidence.update(
                        elapsed_billing_bound_s=elapsed,
                        estimated_total_usd=estimate,
                        within_budget=estimate <= APPROVED_USD,
                        estimated_total_seconds=state["cost"].get("prior_runtime_reserve_seconds", 0) + elapsed,
                        within_runtime=state["cost"].get("prior_runtime_reserve_seconds", 0) + elapsed <= APPROVED_TOTAL_SECONDS,
                        persistent_volumes_created=False,
                        cost_basis="conservative_intent_to_absence_estimate_not_invoice",
                    )
                else:
                    evidence["cost"] = "unverified_no_local_intent"
                save("cleanup.json", evidence)
                if evidence.get("within_budget") is False or evidence.get("within_runtime") is False:
                    raise RuntimeError("Estimated spend or runtime exceeded authorization")
                return
        else:
            absent_checks = 0
            if len(matches) != 1 or not owned(matches[0], state):
                raise RuntimeError("Cleanup ownership mismatch; refusing unrelated deletion")
            pod_id = matches[0].get("id", "")
            if not re.fullmatch(r"[a-zA-Z0-9_-]{1,128}", pod_id):
                raise RuntimeError("Invalid Pod ID")
            if matches[0].get("mounts"):
                raise RuntimeError("Unexpected persistent mount; inspect before deletion")
            if "started_epoch" not in state:
                started = float(matches[0]["env"]["WAFER_STARTED"])
                if not finite(started, 1) or started > time.time():
                    raise RuntimeError("Invalid recovered start time")
                state.update(
                    started_epoch=started,
                    cost=quote(float(matches[0]["env"]["WAFER_RATE"])),
                )
            state["pod_id"] = pod_id
            save("provision.json", state)
            try:
                api.call("/pods/" + pod_id, "DELETE")
            except RuntimeError:
                pass  # DELETE outcome is reconciled by next inventory, never claimed successful.
        if attempt < 7:
            time.sleep(10)
    save("cleanup.json", {"status": "unknown", "name": state["name"]})
    raise RuntimeError("Owned Pod removal not verified; inspect immediately")


def download(url, token, expected, inline=None):
    size, digest = expected.get("size"), expected.get("sha256")
    if (
        type(size) is not int
        or not 0 < size <= MAX_ARCHIVE
        or not re.fullmatch(r"[a-f0-9]{64}", digest or "")
    ):
        raise RuntimeError("Invalid result archive metadata")
    # Use the same explicit client headers as the successful /result requests.
    req = urllib.request.Request(
        url,
        headers={
            "Authorization": "Bearer " + token,
            "Content-Type": "application/json",
            "User-Agent": "wafervlm-actions/0.1",
        },
    )
    if inline is not None:
        if not isinstance(inline, str) or len(inline) > 180000 or size > 128 * 1024:
            raise RuntimeError("Oversized inline diagnostic archive")
        response = io.BytesIO(base64.b64decode(inline, validate=True))
    else:
        response = urllib.request.build_opener(NoRedirect).open(req, timeout=30)
    path = OUT / "experiment.tar.gz"
    hasher, count = hashlib.sha256(), 0
    with response, path.open("wb") as f:
        while block := response.read(1024 * 1024):
            count += len(block)
            if count > size:
                raise RuntimeError("Oversized result archive")
            hasher.update(block)
            f.write(block)
    if count != size or hasher.hexdigest() != digest:
        raise RuntimeError("Archive size/checksum mismatch")


class Deadline(BaseException):
    pass


def interrupt(signum, frame):
    raise Deadline("Work deadline or cancellation; begin cleanup")


def launch(api):
    authorization_guard()
    evidence = preflight(api)
    from runpod.confirmation_contract import launch_plan
    spec = {"seed": 42, "source_sha": SOURCE, "controller_sha": evidence["controller_sha"],
            "image": IMAGE, "gpu": evidence["gpu"],
            "gpu_hourly_usd": evidence["cost"]["gpu_hourly_usd"], "ledger": evidence["ledger"]}
    launch_plan(spec)
    save("launch-spec.json", spec)
    run = identity()
    token = secrets.token_urlsafe(32)
    print("::add-mask::" + token, flush=True)
    started = time.time()
    deadline = started + evidence["cost"]["work_seconds"]
    state = {
        "name": AUTHORIZATION + "-vlm42-" + run,
        "run_id": run,
        "started_epoch": started,
        "deadline_epoch": deadline,
        "cost": evidence["cost"],
        "pod_id": None,
    }
    save("provision.json", state)  # durable intent BEFORE the only create request
    worker = base64.b64encode((ROOT / "runpod/vlm_worker.py").read_bytes()).decode()
    code = (
        "import base64;exec(compile(base64.b64decode("
        + repr(worker)
        + "),'paid_worker.py','exec'))"
    )
    payload = {
        "name": state["name"],
        "image": IMAGE,
        "cloud": "SECURE",
        "disk": DISK_GB,
        "gpu": {
            "id": evidence["gpu"],
            "count": 1,
            "minRamPerGpu": 32,
            "minVcpuCountPerGpu": 4,
            "minCudaVersion": MIN_CUDA_BY_GPU[evidence["gpu"]],
        },
        "dataCenterIds": evidence["regions"],
        "startSsh": False,
        "ports": ["8888/http"],
        "entrypoint": ["/bin/bash", "-lc"],
        "cmd": ["exec /app/.venv/bin/python -u -c " + shlex.quote(code)],
        "env": {
            OWNER: REPOSITORY,
            "WAFER_RUN_ID": run,
            "WAFER_AUTHORIZATION": AUTHORIZATION,
            "WAFER_EXPERIMENT": "vlm-seed42",
            "WAFER_SEED": "42",
            "WAFER_RESULT_TOKEN": token,
            "WAFER_DEADLINE": str(deadline - 90),
            "WAFER_SOURCE_SHA": SOURCE,
            "WAFER_CONFIG_SHA": evidence["config_sha256"],
            "WAFER_CONTROLLER_SHA": evidence["controller_sha"],
            "WAFER_STARTED": str(started),
            "WAFER_RATE": str(evidence["cost"]["gpu_hourly_usd"]),
            "WAFERVLM_IMAGE_DIGEST": IMAGE.split("@")[1],
            "WAFER_DATASET_SHA": json.loads((ROOT / "docs/evidence/wm811k-lot17.json").read_text())["dataset_sha256"],
        },
    }
    handlers = {
        s: signal.signal(s, interrupt) for s in (signal.SIGALRM, signal.SIGTERM, signal.SIGINT)
    }
    signal.alarm(evidence["cost"]["work_seconds"])
    try:
        try:
            pod = api.call("/pods", "POST", payload)  # never retry a create
        except RuntimeError as exc:
            error = {"status": "create_not_confirmed", "checked_epoch": time.time()}
            if isinstance(exc, RemoteHTTPError):
                error.update(http_status=exc.status, categories=exc.categories)
            else:
                error["categories"] = ["transport_or_response_error"]
            # Never serialize the request body, environment, tokens or raw error text.
            save("create-error.json", error)
            raise
        pod_id = pod.get("id", "")
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", pod_id):
            raise RuntimeError("Create returned no valid Pod ID; reconcile by name")
        state["pod_id"] = pod_id
        save("provision.json", state)
        detail = api.call("/pods/" + pod_id)
        if (
            not owned(detail, state)
            or detail.get("gpu", {}).get("id") != evidence["gpu"]
            or detail.get("dataCenterId") not in evidence["regions"]
            or not finite(detail.get("cost"), 0.001)
            or detail["cost"] > evidence["cost"]["gpu_hourly_usd"] + DISK_HOURLY
        ):
            raise RuntimeError("Allocated Pod identity, GPU, placement or rate mismatch")
        save(
            "allocation.json",
            {
                "id": pod_id,
                "gpu": evidence["gpu"],
                "data_center": detail["dataCenterId"],
                "observed_hourly_usd": detail["cost"],
            },
        )
        url = "https://" + pod_id + "-8888.proxy.runpod.net"
        last_phase = None
        last_progress = None
        next_inventory = time.time() + 60
        worker_seen = False
        while time.time() < deadline - 35:
            if time.time() >= next_inventory:
                current = api.call("/pods/" + pod_id)
                if not owned(current, state) or current.get("status") in {"ERROR", "EXITED", "TERMINATED"}:
                    raise RuntimeError("Pod stopped or identity changed before result retrieval")
                print("Pod status: " + str(current.get("status")), flush=True)
                next_inventory = time.time() + 60
            if not worker_seen and time.time() - started > 1200:
                raise RuntimeError("Worker startup exceeded 20 minutes")
            try:
                result = request(url + "/result", token=token)
            except RuntimeError:
                time.sleep(10)
                continue
            worker_seen = True
            phase = result.get("phase")
            if phase != last_phase and phase in {"starting", "data", "training"}:
                print("Worker phase: " + phase, flush=True)
                last_phase = phase
            progress = result.get("progress", {})
            step, total = progress.get("optimizer_step"), progress.get("optimizer_steps")
            if (type(step) is int and type(total) is int and total in {100, 150, 200, 250}
                    and 0 <= step <= total and (step, total) != last_progress):
                print(f"Seed 42 optimizer step {step}/{total}", flush=True)
                last_progress = (step, total)
            if result.get("status") in {"accepted", "rejected", "partial"}:
                # Worker returns only explicit evidence fields, never its environment.
                save("result.json", result)
                if result.get("archive"):
                    download(
                        url + "/archive",
                        token,
                        result["archive"],
                        result.get("archive_inline_base64"),
                    )
                if result.get("status") != "accepted":
                    raise RuntimeError("GPU experiment did not pass all acceptance gates")
                if (
                    result.get("source_sha") != SOURCE
                    or result.get("config_sha256") != evidence["config_sha256"]
                    or result.get("controller_sha") != evidence["controller_sha"]
                    or result.get("profiles") != ["seed-confirmation"]
                    or result.get("seed") != 42
                    or not (OUT / "experiment.tar.gz").is_file()
                ):
                    raise RuntimeError("Result provenance/completeness mismatch")
                from runpod.confirmation_verify import verify_training_archive
                from runpod.vlm_ledger import dataset_metadata
                accepted = verify_training_archive(OUT / "experiment.tar.gz", dataset_metadata(OUT), spec)
                save("acceptance.json", accepted)
                return
            time.sleep(10)
        raise RuntimeError("Deadline reached without complete artifacts")
    finally:
        signal.alarm(0)
        for s, handler in handlers.items():
            signal.signal(s, handler)
        cleanup(api)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("preflight", "run", "cleanup"))
    args = parser.parse_args()
    identity()
    api = Runpod()
    if args.mode == "preflight":
        preflight(api)
    elif args.mode == "cleanup":
        cleanup(api)
    else:
        launch(api)


if __name__ == "__main__":
    main()
