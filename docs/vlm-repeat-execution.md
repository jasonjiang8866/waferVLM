# Frozen six-checkpoint validation repeats

Prepared 2026-11-15 SGT. [Seed 42](vlm-seed42-results.md) and
[seed 123](vlm-seed123-results.md) completed fresh 0-to-250 training. Seed 17
reuses the two pinned released full checkpoints. The [six-endpoint matrix](evidence/vlm-six-endpoints.json)
was reconstructed from all four original ZIPs, including optimizer/scheduler/RNG
states, before starting any new inference. Its SHA256 is
`81c048fbb25396a9c2d937a94d9c055989b032d76d51446cc104f2641fd25a62`.

**2026-11-17 update:** the GPU run completed 14/18 passes before timeout; the
results were retrieved and cleanup verified, but the study was not accepted.
See [verified continuation and current budget](vlm-repeat-resume.md). The launch
allowance below describes the original attempt, not the four-pass continuation.

Launch update: the first dispatch failed at the provider's create API (HTTP 500),
with both cleanup paths verifying absence and no inference performed. See the
[recovery receipt and reduced retry allowance](vlm-repeat-create-recovery.md).

## Workload and interpretation

The existing fixed runner performs **3 seeds × 2 checkpoints × 3 fresh processes
= 18 validation passes**, preserving all 160 ordered predictions per pass.
Report each endpoint's first pass as its headline, every repeat score and changed
prediction, F1 range, and paired step-250-minus-step-200 differences. Distinguish
the exploratory seed 17 from confirmation seeds 42/123. No best-repeat selection,
new training, test scoring or final-holdout inference occurs in this workflow.

The immutable image remains
`ghcr.io/jasonjiang8866/wafervlm@sha256:a6d31b4f12c2c3ee64c59d08b1e67c89e9bbf9b46cbdb4179417f1d1efa3fa79`.
The source inside it is `11d2e5f088b350dd9391864dbe2869417fff5af7`.
The launch guard checks that inference source, scripts, config and package lock
still match that image. Only reviewed orchestration and input receipts are injected;
CPU thread counts, model kernels, generation and analysis recipe are unchanged.

## Input and result verification

Before allocation, CPU CI and preflight re-download both fresh-seed archives and
independent cleanup receipts, reconcile cumulative spend, and reconstruct all six
full checkpoints from hash-pinned ZIPs. Missing, altered or reordered endpoints
block allocation. The worker requests a fresh artifact-scoped URL immediately
before each of the four downloads, never a GitHub/RunPod account token. Delivery
is authenticated and bound to the requested archive and nonce. URLs are masked,
not persisted in artifacts or environment variables, and not inherited by
inference subprocesses. It rechecks each ZIP, training
acceptance, released-state hashes and the same committed six-endpoint matrix.

After retrieval, the controller validates the entire output tree and all 18
distinct process identities, binds every pass to its frozen adapter hash, checks
dataset/sample order and locked inference settings, reparses raw responses and
recomputes every metric and the final summary. Failed/partial output is preserved
without acceptance. Acceptance means a valid study, not model promotion.

## Launch and budget

Manual entry: **WaferVLM fixed-weight validation repeats**,
`.github/workflows/vlm-repeats.yml`, branch `main`, `expected_sha` equal to the
reviewed current main commit. `confirm_paid=false` is read-only preflight;
`true` can create one Pod after all checks. Reruns and unreconciled prior paid
repeat attempts are blocked. An ambiguous create is reconciled, never retried.

The starting ledger is **US$8.132282 / 640.704 minutes** within the existing
**US$30 / 1100-minute** authorization, expiring 2026-11-17 08:00 SGT.
The attempt allows **60 minutes total work** (including startup, input transfer,
data preparation, all 18 passes and result retrieval), plus **10 minutes cleanup**.
At a verified RTX 4090 quote of US$0.89/hour, the full 70-minute estimate adds
US$1.04875 including temporary disk; cumulative US$9.181032 / 710.704 minutes.
The live quote is checked at launch; this estimate is not a provider-enforced cap.
The original protocol's evaluation allowance is shared with the later final study,
so its remaining allocation must be reviewed after actual repeat consumption.

Retain 4090 preference, authorized fallback/rate ceilings and CUDA >=12.8.
No persistent volume is created. Paid workflows share the existing mutex, and
the VLM preflight blocks any existing owned project Pod. The independent cleanup
workflow verifies absence three times and only deletes the exact originating
`vlm-repeats` Pod, never seed-training or DINOv3 resources. DINOv3's implementation
and separate authorization are unchanged.
