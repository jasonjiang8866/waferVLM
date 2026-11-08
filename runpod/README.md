# Bounded RunPod GPU experiment

User authorization (2026-11-07): **US$20 total**, including compute, retries,
temporary disk, transfer and cleanup. This replaces the earlier observation-only
budget state. The first attempt passed the 8-step smoke but timed out at
74/100 baseline steps; its Pod is verified deleted. See [results](../docs/gpu-attempt-42346201550.md).
The user subsequently raised the aggregate runtime ceiling to **500 minutes**;
the total US$20 cap and per-GPU price caps remain unchanged.

## Current continuation: checkpoint 100 → 200

The completed baseline reached test accuracy 0.53125 / macro-F1 0.521659 and
validation accuracy 0.64375 / macro-F1 0.640676. User approved **100 additional
steps** on 2026-11-08. This is a true Trainer resume: adapter, AdamW optimizer,
constant scheduler, RNG state and data position are restored, with immutable
parent hashes checked before deserialization. Hyperparameters and dataset stay
unchanged. Real tiny-model CPU CI compares resumed vs uninterrupted adapters.

The build retrieves artifact `13582826145` from run `42385615853`, verifies inner
archive SHA256 `e7e4fad8a89af270278b72529db333f1698d8c027e761a6629a0464cac32633c`,
and embeds only the public-data checkpoint-100 and provenance in the image.
No GitHub or RunPod credential enters an image layer or Pod. This avoids needing
new persistent storage or exposing an account token to download a checkpoint.

- Re-evaluate checkpoint 100 on validation, including answer-token cross-entropy.
- Resume to 150, validate; resume to 200, validate (exactly 50 + 50 new steps).
- Select highest validation macro-F1, breaking ties toward the earlier step.
- Record selection **before** evaluating that selected adapter on test once.
- Retain all checkpoints and report results in **experiment.tar.gz**; no new smoke.

Both earlier Pods are independently verified deleted. Prior reserves now total
**US$2.10 / 180 minutes**, rounded above the observed cumulative estimate. Same
US$20 / 500-minute authorization, same four GPU caps and CUDA >=12.8. One new
attempt is bounded to 180 minutes work + 10 cleanup, thus **370 minutes cumulative**
and at most **US$5.89661 cumulative** at the highest approved rate. A further
attempt remains blocked by its authorization marker until reconciled in code.

The worker's continuation deadline is 160 minutes within the outer controller
limit. Preflight checks that image/source pins match the reviewed workload.
On 2026-11-08 the user explicitly approved publishing the checkpoint state
(adapter, optimizer, scheduler and RNG, without credentials) in this GHCR image.

- Current image: `ghcr.io/jasonjiang8866/wafervlm@sha256:9bf7e57906d19b46c6d1ca8c01c0b9cf871fce4ee760518af71c3101e4f3cf1f`.
- Image source: `d1111efdbe31f9b866610ad7f1ab4e167a7709ef`.
- Publication: [42430153900](https://github.com/jasonjiang8866/waferVLM/actions/runs/42430153900).
- Launch `gpu.yml` on main with `confirm_paid=true` and the exact reviewed main SHA.

## Historical launch specification (completed attempts 1 and 2)

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
Startup, data, workloads and result retrieval share a **180-minute** per-attempt
controller deadline, plus a **10-minute** cleanup allowance. The Actions job has
200 minutes to include setup and artifact upload. Baseline's stage deadline is
120 minutes (formerly 50); smoke retains 30 minutes. Worker startup is limited
to 20 minutes. Numeric optimizer-step telemetry appears in Actions logs.

The first attempt is reconciled by exact run/artifact IDs, with conservative
reserves of **US$0.80 and 70 minutes** (observed estimate $0.774385 / 3988.653 s).
The next attempt therefore plans at most **260 cumulative minutes** and, at the
highest allowed GPU rate, **US$4.59661 cumulative** including disk and cleanup.
The controller clips time to both remaining dollars and the 500-minute aggregate
limit. Any later retry remains blocked until its marker, spend, runtime and
cleanup are reviewed and reconciled; this dispatch does not authorize an
unbounded retry loop. The next baseline starts fresh with the unchanged pinned
image/config; checkpoint-50 is retained, but resume is not yet implemented.
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
