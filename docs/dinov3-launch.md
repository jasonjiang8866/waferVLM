# First DINOv3 RunPod experiment

Status: launch integration prepared; **no DINOv3 GPU has been allocated**. The real
[CPU access preflight on 2026-11-14](https://github.com/jasonjiang8866/waferVLM/actions/runs/43552229499)
failed before downloading weights because repository Actions secret `HF_TOKEN`
was not available. This is separate from the historical Qwen authorization.

## Reviewable first launch

| Item | Proposed bounded experiment |
|---|---|
| Workload | Seed 17, linear `{.001,.003,.01}` and Q/V LoRA `{.00005,.0001,.0002}`; six full 10-epoch trials |
| Data | Existing hash-pinned 790 train / 160 validation; historical test integrity audit only; no reserved holdout |
| GPU | One RTX 4090, secure cloud; current catalog GPU rate must be <= US$0.89/hour |
| CUDA | Host capability minimum 12.8 via `minCudaVersion`; newer supported versions accepted |
| Disk | 60 GB ephemeral; no persistent volume |
| Time | 90 minutes total from create intent through retrieval, plus 10 minutes cleanup |
| Proposed total spending ceiling | US$3 including compute, temporary disk, transfer and cleanup; user approval still required |
| Ceiling-rate estimate | US$1.49822 for 100 minutes including disk, leaving > US$1.50 uncertainty reserve; **not a current provider quote or enforced billing cap** |
| Selection | Highest validation macro-F1, earliest epoch ties; lower LR for tied trials |
| Follow-up | Freeze selected LRs before separately authorizing/dispatching seeds 42 and 123 |

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
4. After approving the first sweep's budget, dispatch `dinov3-gpu.yml` on main:
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
persisted marker, so marker expiry/deletion does not silently allow another run.
This also blocks a repeat after a pre-allocation failure; do not use the paid
workflow for credential experiments—use the dedicated free CPU preflight.

Evidence is uploaded as `dinov3-seed17-<run-id>` with a separate cleanup artifact.
Provider rates/inventory, actual CUDA performance, final spending, official
weights and real remote upload/cleanup remain unverified until a live run.
