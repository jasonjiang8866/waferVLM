# Bounded RunPod GPU experiment

Seed-confirmation implementation and approval handoff: [fixed training/repeat runners](../docs/seed-confirmation-implementation.md). No new paid workflow has been activated.

**Current authorization:** US$30 / 1,100 minutes cumulative. The [validation-only 200→300 continuation](../docs/continuation-300-results.md) completed; checkpoint 250 was selected. Conservative cumulative usage is **US$5.903039 / 491.911 minutes**. Its Pod was verified absent by primary and independent cleanup; no persistent volume was created. Full states and predictions are preserved in [continuation-300-v1](https://github.com/jasonjiang8866/waferVLM/releases/tag/continuation-300-v1). Earlier diagnostic parity failures remain rejected. The [next seed-confirmation workload](../docs/seed-confirmation-plan.md) is planned but not executed. The current authorization marker remains consumed until reconciled with reviewed next-stage code.

Prefer the already-authorized RTX 4090 after its measured 12.8–13.0s/step probe;
MIG 24GB, 3090 and 5090 remain within their existing price caps. Fresh preflight
must still establish price/capacity and owned-Pod absence. CUDA >=12.8 remains
unchanged. The published extension image below produced the completed run;
historical image pins below describe earlier workloads.
Published extension image: `ghcr.io/jasonjiang8866/wafervlm@sha256:b13a9807ae05698f0813e0d1ab7f25548e782f65729d2b2aec5e66de6de24cd1`.
Image source: `064a37fa828ca0fa4ce106bfce46db3da48d8e5b`.
Publication: [42680747996](https://github.com/jasonjiang8866/waferVLM/actions/runs/42680747996), successful.
The controller pins these values and rejects workload drift; GPU execution still
requires manual main dispatch and the exact reviewed current SHA.


User authorization (2026-11-07): **US$20 total**, including compute, retries,
temporary disk, transfer and cleanup. This replaces the earlier observation-only
budget state. The first attempt passed the 8-step smoke but timed out at
74/100 baseline steps; its Pod is verified deleted. See [results](../docs/gpu-attempt-42346201550.md).
The user subsequently raised the aggregate runtime ceiling to **500 minutes**;
the total US$20 cap and per-GPU price caps remain unchanged.

## Latest completed result

[Run 42682266526](https://github.com/jasonjiang8866/waferVLM/actions/runs/42682266526) completed 200→250→300 at constant LR 5e-5. Step 250 validation accuracy was **78.125%**, macro-F1 **0.773992**, invalid outputs **0/160**. No new test or holdout inference ran. Incremental conservative cost was **US$0.523039 / 34.911 minutes** on RTX 4090. See [results, hashes and recovery](../docs/continuation-300-results.md).

## Historical completed result: checkpoint 200

[Run 42502913001](https://github.com/jasonjiang8866/waferVLM/actions/runs/42502913001)
completed the lower-LR continuation on 2026-11-09 at 03:16 SGT. Validation selected
checkpoint-200; test accuracy is **64.375%**, macro-F1 **0.636059**, invalid output
**0/160**. See the [full report and recovery guide](../docs/checkpoint-200-release.md)
and [preserved Release](https://github.com/jasonjiang8866/waferVLM/releases/tag/checkpoint-200-v1).

The temporary Pod is verified absent by the controller and independent cleanup.
Cumulative conservative usage is **US$5.051749 / 433.21 minutes**, including
prior reserves, within the US$20 / 500-minute cap. No persistent volume was
created. The fourth attempt's marker is reconciled for the bounded diagnostic;
any new run must still pass the ledger and pinned-image checks. The specification below records the
completed run and is not an instruction to launch it again.

## Historical launch specification: lower LR from checkpoint 100

The constant-2e-4 continuation completed at step 200. Checkpoint-100 remained best
by validation macro-F1. See [verified control results](../docs/continuation-control-42431025521.md).
The user approved this experiment on 2026-11-09 (SGT). It ran the same 100 new
steps, from the same checkpoint-100, at **constant LR 5e-5**. Learning-rate decay
is deferred to a separate experiment so this comparison changes one variable.

Trainer restores the complete adapter/AdamW/scheduler/RNG/data position. An
explicit lower-LR flag then changes optimizer LR and constant-scheduler base LR,
without resetting Adam moments or step counters. Every effective optimizer LR
is checked and exported. Tiny-model CI covers restore, save and another resume.

The build retrieves artifact `13582826145` from run `42385615853`, verifies inner
archive SHA256 `e7e4fad8a89af270278b72529db333f1698d8c027e761a6629a0464cac32633c`,
and embeds the same checkpoint-100 state already approved for GHCR publication.
No account credential enters the image or Pod. No persistent volume is required.

- Validate checkpoint 100 with answer-token loss.
- Resume to 150 with the explicit LR intervention, validate.
- Resume to 200 with the lower LR preserved, validate.
- Select highest validation macro-F1 (ties: earlier step), then evaluate test once.
- Export all adapters/checkpoints, actual LR traces and results in `experiment.tar.gz`.

All three previous Pods are verified absent. Reserve **US$3.50 / 300 minutes**
for those attempts (observed conservative cumulative estimate $3.485102 / 298.905
minutes). The same US$20 / 500-minute total cap, GPU price caps and CUDA >=12.8
remain in force. This attempt permits 180 minutes work plus 10 minutes cleanup:
**490 minutes cumulative**, and at most **US$7.29661 cumulative** at the highest
approved GPU rate including ephemeral disk and cleanup. The continuation worker
has a 160-minute stage deadline. Later dispatches remain blocked until this
attempt's ledger, spend and cleanup are reconciled; this is not a retry loop.

The lower-LR image has passed its build and embedded-checkpoint checks and is
pinned below. The completed run used `gpu.yml` on main with `confirm_paid=true`
and controller SHA `4ff0fb4c878decb0afaec0d433215bb7e0ead274`. Preflight verified
anonymous manifest access and rejected workload drift from this image's source.

- Worker image: `ghcr.io/jasonjiang8866/wafervlm@sha256:84c6d4a5734421ecfc2cfc7c1b2d9bce5d855edda7b925e73238009825b34b6e`.
- Worker image source: `ef254921225ded9bbb3aec818de177c9d718db73`.
- Publication: [42502105471](https://github.com/jasonjiang8866/waferVLM/actions/runs/42502105471).

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

