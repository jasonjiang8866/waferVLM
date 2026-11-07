"""Check waferVLM's task against the pinned Broker without credentials or cloud calls."""

import json
from pathlib import Path

from runpod_ci_broker.models import Candidate, Task
from runpod_ci_broker.planner import plan, reasons
from runpod_ci_broker.provider import ReadOnlyRunPod

ROOT = Path(__file__).resolve().parents[1]
task = Task.model_validate_json((ROOT / "runpod/preflight.json").read_text())
assert task.compatibility.min_cuda_version == "12.8"
assert task.compatibility.cuda_versions == []
assert task.limits.total_budget_usd is None
assert task.limits.max_gpu_hourly_usd is None
assert task.volume.kind == "none"  # No existing storage selected/modified.

# Artificial catalog values exercise admission only; they are not supplier quotes.
for version in ("12.7", "12.8", "12.9", "12.10", "13.0"):
    candidate = Candidate(gpu_id=task.gpu_allowlist[0], data_center="TEST",
                          memory_gb=24.0, hourly_usd=1.0, capacity="available",
                          cuda_versions=[version], evidence="mock")
    blocked = reasons(task, candidate)
    assert ("cuda_incompatible" in blocked) == (version == "12.7"), (version, blocked)
    result = plan(task, [candidate], mode="mock")
    assert "budget_inputs_missing" in result["blockers"]
    assert result["selected"] is None and not result["paid_authorized"]

# Verify the actual API adapter sends a minimum floor, not an exact-version set.
calls = []
def get(path, params):
    calls.append((path, params))
    if path == "/catalog/gpus":
        return {"gpus": []}
    if path == "/pods":
        return {"pods": [], "pagination": {"hasNextPage": False}}
    if path == "/catalog/datacenters":
        return {"dataCenters": []}
    raise AssertionError(path)

ReadOnlyRunPod(get=get).snapshot(task)
assert calls[0][1]["minCudaVersion"] == "12.8"
assert "cudaVersions" not in calls[0][1]
assert {p for p, _ in calls} == {"/catalog/gpus", "/pods", "/catalog/datacenters"}
print(json.dumps({"cuda_floor": "12.8", "higher_versions_accepted": ["12.9", "12.10", "13.0"],
                  "budget_missing_blocks_admission": True, "network_calls": 0}))
