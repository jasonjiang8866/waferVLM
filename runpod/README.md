# Bounded RunPod GPU experiment

User authorization (2026-11-07): **US$20 total**, including compute, retries,
temporary disk, transfer and cleanup. This replaces the earlier observation-only
budget state. No paid experiment has completed yet.

## Launch specification

- Workflow: **WaferVLM bounded GPU experiment** (`gpu.yml`), manual main only,
  `confirm_paid=true`, `expected_sha=<exact reviewed current main SHA>`.
- Immutable worker: `ghcr.io/jasonjiang8866/wafervlm@sha256:31c645533047eb8912c00fa5790bfa4f96cdf07cdaae1779bdcd3eb113a5b440`.
- Worker source: `81a2b1a34ddc8cd75aa432d36c44ae0be3174948`. Controller additions
  do not change the image's workload. Preflight rejects any source/config/lockfile
  drift and rechecks anonymous manifest access.
- Model: Qwen/Qwen3.5-0.8B, revision `2fc06364715b967f1860aea9cf38778875588b17`.
- Data: hash-pinned public WM-811K mirror; worker reproduces the audited
  790/160/160 split and compares its dataset identity with committed evidence.
  No factory/private data. Download and data preparation count toward runtime.
- One GPU, minimum 32GB host RAM / 4 vCPUs; host **CUDA >=12.8** for every GPU.
  No exact-version filter. BF16/CUDA is checked on the real allocated worker.
- 60GB ephemeral container disk; no persistent or Global Volume creation.
- 8-step validation smoke followed only on success by a fresh 100-step baseline
  (seed17, accumulation8) and paired test evaluation. An execution pass does not
  assert quality improvement; compare macro-F1, accuracy and invalid output rate.

| Allowed GPU (preference order) | Maximum GPU-only USD/hour |
|---|---:|
| PRO 6000 Blackwell MIG 1g.24gb | 0.69 |
| RTX 4090 | 0.89 |
| RTX 3090 | 0.50 |
| RTX 5090 | 1.19 |

Fresh catalog price and capacity must pass before the only create request.
Startup, data, workloads and result retrieval share a **5,400-second** controller
deadline, plus a **600-second** cleanup allowance. Worker startup is limited to
20 minutes. At the highest allowed rate, the planned GPU + disk bound is
**$1.99822**, leaving over $18 of the total authorization unallocated.
The 60GB disk estimate uses $0.10/GB/month and a conservative 28-day month;
[RunPod Pod pricing](https://docs.runpod.io/pods/pricing) states no ingress/egress
charge. An estimate is not a provider-enforced hard cap: prolonged API/cleanup
failure can overrun. Account credit adequacy and the existing $500 monthly
headroom guard are checked without exporting balances or raw billing history.

## Lifecycle and credential boundaries

`RUNPOD_API_KEY` stays in Actions. The Pod receives only ownership/provenance
markers and a random, masked, run-scoped result-read token. Training subprocesses
receive no account credential or result token. HTTP results use the RunPod HTTPS
proxy, constant-time bearer checking and bounded download/checksum verification.

A 90-day Actions artifact consumes authorization **before** POST /pods. Duplicate
run attempts, another dispatch with an unreconciled marker, non-main execution,
changed main SHA, unconfirmed inputs and expired authorization are rejected.
This initial implementation makes one create request and never retries an
ambiguous create. Any retry needs reconciliation of its marker, cleanup and
spend in code; it does not grant a fresh $20. Authorization expires November 14.

Cleanup uses exact repository, authorization, run ID, name and known Pod ID.
It refuses unrelated Pods and unexpected persistent mounts. `finally` cleanup
and an `always()` inventory check run in the controller workflow. A separate
`workflow_run` recovery workflow runs on completion/cancellation/timeout even if
the original runner is lost. It can also be manually dispatched with
`origin_run_id`. Repeated independent inventory absence is required. These
controls depend on GitHub/RunPod availability; a stopped worker alone does not
prove compute deletion or billing cessation.

Artifacts include sanitized preflight, intent, allocation, worker result,
full checkpoint/adapter/log archive, integrity acceptance and cleanup evidence.
All archive paths/types/sizes, nested checksums, data/model/image/source identity,
step/gradient counts and complete 160-example evaluations are independently
verified before marking the execution accepted. Report quality separately.

## Validation and provenance

```bash
python -m unittest discover -s runpod/tests -v
```

Controller/provider and lifecycle fault tests were adapted from this user's
MIT-licensed `flashattentionanycard` commit
`d53afd9aa6f3ce68ac896db0924558d1fe72ffe7`; the upstream notice is retained in
`UPSTREAM-LICENSE`. REST v2 schema was checked against the
[official create API](https://docs.runpod.io/api-reference-v2/pods/create-a-pod).

The previous `preflight.yml` / `preflight.json` remain explicitly GET-only,
unbudgeted observation fixtures used by the Broker contract tests. They do not
control this paid workflow. The Broker dependency has not been changed and is
not claimed to implement live lifecycle management.

Published image evidence: [42342921932](https://github.com/jasonjiang8866/waferVLM/actions/runs/42342921932).
Earlier inventory evidence: [42342948679](https://github.com/jasonjiang8866/waferVLM/actions/runs/42342948679).
