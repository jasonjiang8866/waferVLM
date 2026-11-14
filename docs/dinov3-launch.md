# First DINOv3 RunPod experiment

Status: merge and execution authorized on 2026-11-14; **no DINOv3 GPU has been allocated yet**. The real
[CPU access preflight on 2026-11-14](https://github.com/jasonjiang8866/waferVLM/actions/runs/43552229499)
failed before downloading weights because repository Actions secret `HF_TOKEN`
was not available. The user subsequently configured it and approved **US$10 /
500 minutes cumulative for DINOv3**, including retries, temporary storage,
transfer and cleanup, plus PR #22 merge and execution. This is separate from the
historical Qwen authorization. The first attempt keeps the stricter US$3 /
100-minute bound below; subsequent attempts must reconcile spend and cleanup
against the same US$10 / 500-minute total, not receive a fresh budget.

[Repeated CPU preflight 43562198996](https://github.com/jasonjiang8866/waferVLM/actions/runs/43562198996)
passed on main `529aea9b91a8c07881e7e6523727187e9c940c15`: the approved token
downloaded and fully loaded pinned official ViT-S/16 weights (21,596,544
parameters). This establishes weight access, not GPU quality or performance.

## Reviewable first launch

| Item | Authorized first bounded experiment |
|---|---|
| Workload | Seed 17, linear `{.001,.003,.01}` and Q/V LoRA `{.00005,.0001,.0002}`; six full 10-epoch trials |
| Data | Existing hash-pinned 790 train / 160 validation; historical test integrity audit only; no reserved holdout |
| GPU | One RTX 4090, secure cloud; current catalog GPU rate must be <= US$0.89/hour |
| CUDA | Host capability minimum 12.8 via `minCudaVersion`; newer supported versions accepted |
| Disk | 60 GB ephemeral; no persistent volume |
| Time | 90 minutes total from create intent through retrieval, plus 10 minutes cleanup |
| First-attempt spending ceiling | US$3 including compute, temporary disk, transfer and cleanup, within approved US$10 cumulative |
| Ceiling-rate estimate | US$1.49822 for 100 minutes including disk, leaving > US$1.50 uncertainty reserve; **not a current provider quote or enforced billing cap** |
| Selection | Highest validation macro-F1, earliest epoch ties; lower LR for tied trials |
| Follow-up | Freeze selected LRs before dispatching seeds 42 and 123 within the remaining approved aggregate budget/time |

The first launch is a development sweep, not a three-seed result or an unbiased
holdout comparison. All six trials share one Pod, one budget and one controller
deadline. No automatic Pod-create retry or confirmation-seed launch exists.

## Credential and image prerequisites

1. Obtain access for `facebook/dinov3-vits16-pretrain-lvd1689m` on its official
   Hugging Face page. Put an appropriate read-only token in repository Actions
   secret `HF_TOKEN`; never paste it into a workflow input or commit.
2. Run `dinov3-preflight.yml` on main to verify official checkpoint access/load.
3. Merge this reviewed integration, then publish `dinov3-image.yml` on the exact
   resulting main SHA with `publish=true`. It contains code only, not gated
   weights. Record the resulting immutable GHCR digest.
4. Using the approval above, dispatch `dinov3-gpu.yml` on main:
   `confirm_paid=true`, `expected_sha=<that full main SHA>`, and
   `image=ghcr.io/jasonjiang8866/wafervlm@sha256:<published digest>`.

The paid workflow downloads and loads weights on CPU again **before** allocating
compute. `HF_TOKEN` is scoped only to that step. The controller verifies both
cached file hashes, the immutable image manifest/config and its embedded source
SHA, current main, current GPU quote/capacity, account headroom and owned-Pod
absence. By default the shared project concurrency group serializes Qwen and
DINO launches. The optional narrowly scoped parallel mode is described below.

Only `config.json` and `model.safetensors` are streamed over the RunPod HTTPS
proxy with a masked per-run token. The worker checks their pinned hashes before
opening them, uses an offline HF cache, and receives no RunPod, GitHub or HF
account credentials. Uploads cannot choose arbitrary filenames or overwrite a
received file. Gated base weights are outside the result directory and are never
published in GHCR or uploaded as GitHub artifacts.

## Results and cleanup

The worker preserves six complete histories, head/adapter exports, per-example
probabilities and metrics. Both selected models are reloaded in fresh processes
for validation and timing. The comparison includes the existing Qwen checkpoint
250 predictions, wafer-thumbnail confusion matrices (X=true, Y=predicted), and
paired correct/error examples. No direct speedup claim uses Qwen's incompatible
historical generation timing.

The controller retrieves a bounded checksummed archive, safely extracts it in a
fresh temporary directory, and independently checks data/source/image/base-weight
identities, all six recipes, 7,900 examples and 990 updates per trial, selection,
reloaded weight identities and recomputed prediction metrics. `accepted` means
execution/integrity; it does not assert that LoRA or DINO beats either baseline.

Cleanup runs in the controller's `finally`, in the workflow's `always()` step,
and in `dinov3-cleanup.yml` after completion/cancellation/timeout. Only exact
repository/authorization/run/name ownership may be deleted. Repeated inventory
absence is required; an ambiguous create is never retried. The independent
cleanup also supports manual `origin_run_id`. API failure can delay deletion and
billing; a worker timeout alone is not cleanup evidence.

The first manual GPU dispatch is one-off. A second dispatch or rerun is blocked
until its complete outcome, cost and cleanup have been reviewed and the launch
ledger deliberately reconciled. Workflow history is checked as well as the
persisted upload receipt, so marker expiry/deletion does not silently allow another run.
This also blocks a repeat after a pre-allocation failure; do not use the paid
workflow for credential experiments—use the dedicated free CPU preflight.

Evidence is uploaded as `dinov3-seed17-<run-id>` with a separate cleanup artifact.
Provider rates/inventory, actual CUDA performance, final spending, official
weights and real remote upload/cleanup remain unverified until a live run.

The marker is read by the exact artifact ID returned by its upload step; this avoids the observed GitHub list-visibility race while still verifying run, source, name, expiry and digest.

## Optional parallel execution (activated 2026-11-14)

The historical first dispatch [43563592797](https://github.com/jasonjiang8866/waferVLM/actions/runs/43563592797)
was waiting for VLM seed-42 run
[43562568201](https://github.com/jasonjiang8866/waferVLM/actions/runs/43562568201).
At 2026-11-14 05:56 UTC, the VLM worker was training (29/100 optimizer steps
in its first phase), while DINO had no jobs or paid allocation. A shared mutex
serializes these separately funded experiments even though they use independent
Pods, temporary disks and cleanup ownership.

The `parallel_vlm_run_id` input defaults to empty, preserving serial
execution. Explicitly selecting `43562568201` uses a DINO-specific mutex and
allows at most that one VLM Pod alongside DINO. Both live Actions identity
(workflow, source, branch, event, first attempt, state) and Pod identity
(repository, authorization, experiment, controller source, run and name) must
match. Unknown/duplicate project Pods, stopped Pods, persistent mounts and
rates above the reviewed VLM ceiling block creation. No VLM lifecycle code is
changed. DINO cleanup still targets only its own exact authorization/run/name.

The account guard conservatively reserves **US$33**: the first DINO ceiling of
US$3 plus the VLM experiment's entire separately approved US$30 allowance. This
is headroom verification, not a new charge or a larger DINO budget. DINO remains
within US$10 / 500 aggregate minutes and US$3 / 100 minutes for this first sweep.
Live GPU inventory, price and account checks still run twice before allocation.

To switch the queued task after the parallel change is reviewed and approved:

1. Recheck DINO run `43563592797`. Only if it is still waiting with **zero jobs**
   may it be cancelled and replaced. Do not interrupt or replace a running
   experiment under this procedure. Do not cancel VLM.
2. Verify the old DINO run is terminal `cancelled`, attempt 1, has zero jobs
   across **all attempts**, and zero artifacts. Verify its independent cleanup
   succeeds. This proves no execution began, rather than assuming cancellation
   means zero spend. Live inventory is checked again before the new create.
3. Merge the reviewed change; publish an image for the exact new main SHA and
   verify immutable image/source identity. Do not reuse the old digest with a
   new source SHA.
4. Dispatch once with the normal paid/SHA/image inputs plus
   `parallel_vlm_run_id=43562568201` and
   `replace_queued_run_id=43563592797`.
5. The ledger now allows exactly the current dispatch, this cancelled zero-job
   predecessor, the explicitly reconciled CPU failure, and the first paid
   transfer failure below. Any additional run, changed identity or rerun requires separate
   reconciliation. This does not reset spend/time allowances.

If the queued run begins before switching, preserve it and use the existing
serial path. The unit tests cover ownership, duplicate history, account reserve,
default behavior, cancelled-versus-executed runs and cleanup isolation; the new
parallel path created a separate 4090 in run `43568169684` without interrupting
VLM. The six-trial training/result path still needs a successful live run.

## CPU preflight failure and recovery

Parallel mode was authorized and PR #26 merged as
`f851c88e1dd40f99bb03157c163dda4ec9d564fe`. Dispatch
[43566277450](https://github.com/jasonjiang8866/waferVLM/actions/runs/43566277450)
downloaded and loaded the official weights, then failed the offline tests.
The job-level scheduling inputs leaked into tests that deliberately mock
single-run history and empty inventory. They activated the parallel path inside
those tests; one fake-token request consequently received HTTP 401. Provider
preflight, authorization marker and paid create/training steps were all skipped.

Scheduling inputs are now scoped to the two controller steps. Lifecycle tests
also isolate these variables, and CI exercises the tests with the real
scheduling input values present. This fixes environment leakage without
disabling a test or loosening provider checks.

`runpod/dinov3-precreate.json` records the exact failed run, job, source, required
skipped steps and two immutable cleanup artifact receipts. Both cleanup records
verified 12 consecutive inventory absences. The result archive contains only
CPU weight fingerprints and cleanup; no provision intent or allocation exists.
The reconciled GPU cost/runtime are therefore zero. The next dispatch verifies
the exact live run/job history, both artifact identities/digests and archive
contents before accepting this exception. A paid step, rerun, extra job,
changed cleanup or unexpected resource-intent file still blocks launch.

The first sweep remains within US$3 / 100 minutes and the approved aggregate
US$10 / 500 minutes. VLM training and its independent budget are unchanged.


## First paid attempt: transfer failure and bounded recovery

Run [43568169684](https://github.com/jasonjiang8866/waferVLM/actions/runs/43568169684)
passed official CPU weight loading, offline tests, image provenance, inventory,
price and account checks. It allocated one RTX 4090 in US-NC-1 at US$0.89/hour.
The worker reached `awaiting_weights`; the controller then received an HTTP
error during weight delivery. No training result was produced. The old log
preserved only the exception type, so the exact response status and underlying
cause cannot be recovered; a proxy size/timeout limit is not established.

The main receipt verifies three consecutive inventory absences, with an
intent-to-absence bound of 177.835 seconds and estimated US$0.044406 including
60 GB ephemeral disk. Independent cleanup run
[43568497764](https://github.com/jasonjiang8866/waferVLM/actions/runs/43568497764)
verified twelve additional absences. No persistent volume was created. These
are conservative estimates, not provider invoices.

`runpod/dinov3-paid-history.json` pins both artifacts, their complete JSON
contents, source, run and job. The next dispatch revalidates those bytes and
live run history, charges **US$0.05 / 180 seconds** to DINO's aggregate ledger,
and allows exactly one further paid attempt. Unknown runs/reruns still block.
The remaining aggregate allowance is US$9.95 / 497 minutes; the recovery still
uses 90 minutes of work plus 10 minutes of cleanup and a cumulative US$3 ceiling.

Weight delivery now uses authenticated chunks of at most 4 MiB. Each chunk and
the completed file are SHA256-checked; duplicate acknowledgements can be safely
replayed, but content cannot be overwritten. Only transient transport/5xx,
408 and 429 failures get at most three requests for the same chunk. There is
still exactly one Pod-create request. `transfer.json` preserves the filename,
byte range, attempt and HTTP status without tokens, URLs or response bodies.
Local HTTP tests cover multi-chunk delivery, a lost acknowledgement after bytes
are committed, invalid ranges/hashes and sanitized terminal errors. These tests
are not a claim that the live RunPod proxy has already been verified.


## Second paid attempt: first 743-byte upload receives HTTP 403

Recovery run [43569912370](https://github.com/jasonjiang8866/waferVLM/actions/runs/43569912370)
again reached `awaiting_weights`. Its sanitized transfer receipt identifies
`config.json`, offset 0, size/total 743 bytes, attempt 1, HTTP **403**. No model
weight or training result was delivered. This is not a large-payload failure.
The main cleanup receipt verifies three inventory absences and estimates
167.819 seconds / US$0.041905 including disk. Across the two paid attempts the
measured intent-to-absence estimates sum to **345.654 seconds / US$0.086311**;
rounding each upward reserves **US$0.10 / 360 seconds** for future accounting.
These remain estimates rather than provider invoices.

The successful `/result` requests already send `User-Agent: waferVLM-actions/0.1`.
Upload and archive download had omitted that header, falling back to Python's
default. Both now use the same honest project client identifier. No credential,
permission, TLS, redirect, endpoint or authentication rule is changed. The local
HTTP regression requires that header and still verifies chunk replay; archive
download checks the same header.

A free Actions diagnostic compares two anonymous GETs to this now-deleted Pod's
proxy route, with the Python default and the project client respectively. It
requires no RunPod or HF secret and creates no resource. This can discriminate
an HTTP client-dependent proxy response; it cannot establish successful weight
delivery or CUDA training. The observed response and root-cause conclusion must
come from the diagnostic/live evidence, not from a passing synthetic test.

No third paid dispatch is enabled here: the existing four-run history guard
rejects another attempt until the second run's spending/cleanup are explicitly
reconciled and an additional paid attempt is authorized.
