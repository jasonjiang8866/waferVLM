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
absence. A shared project concurrency group serializes Qwen and DINO launches.

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
