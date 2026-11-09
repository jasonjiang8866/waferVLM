# Validation-only continuation from checkpoint 200

Prepared 2026-11-10 SGT. Code/configuration preparation is complete; execution
requires a new published/pinned worker image and an exact-main-SHA manual
dispatch. A completed run is distinct from a quality gain.

## Why proceed

Released validation macro-F1 rose 0.640676 →0.686280 →0.717461 at steps
100/150/200, while answer-token loss fell 0.270583 →0.220790 →0.211074.
The [diagnostics](checkpoint-200-diagnostics.md) confirmed actual CUDA execution
and obtained phase timings. Their unchanged numerical-parity gates failed;
variation also occurred between unprofiled repeats. No profiler, fused kernel,
thread change, new dependency or other speed modification is adopted. Continue
the existing unprofiled training recipe and report its single-seed limits.

## Fixed experiment

- Parent: full checkpoint-200 from `checkpoint-200-v1`, original run 42502913001.
  `prepare_profile_parent.py` downloads the immutable Release asset, verifies
  ZIP/inner/member hashes and stages adapter, Adam moments/steps, scheduler,
  RNG, Trainer state and metadata. Exact hashes are in `configs/extension.json`.
- Same Qwen3.5-0.8B revision, audited 790/160/160 WM-811K development dataset,
  prompt/rendering, answer-only loss, frozen vision and language-only LoRA.
  Seed17, rank8, accumulation8, constant LR5e-5, max_length1024.
- Re-evaluate checkpoint-200 on all 160 validation images on the allocated GPU.
  Execute 50 updates to 250; validate. Restore that complete state, execute 50
  more updates to 300; validate. No LR intervention or scheduler reset.
- Select highest validation macro-F1 among 200/250/300, ties to the earlier step.
  Retain 200 if neither continuation improves. Report accuracy, loss, invalid
  rate, per-class counts and actual runtime alongside F1. This is development
  selection, not independent evidence of final generalization.
- No test inference, new holdout inference or diagnostic-weight promotion.
  The new runner has no test-evaluation call. Dataset integrity auditing still
  verifies the existing split manifests/images without model evaluation.

## Frozen final holdout

`docs/evidence/final-holdout-v1/` records 148 source identities and image hashes:
20 per class except near-full, for which only **8 eligible unseen maps** remain.
All eight classes are represented, but the rare-class estimate will be imprecise.

The reservation excludes every duplicate-connected lot group containing any of
the 1,110 development images, quarantines the same six conflicting rows, then
uses a fixed salted ID ordering and rejects raw-map/rendered-image collisions.
Selection uses no model prediction or confidence. No holdout image was visually
reviewed or evaluated with a model. The manifest hash is
`f08050ef4fcaca3b9c35ad29905a63957a375af7a024b69e0281968d06f4564f`.
This establishes non-overlap with the recorded waferVLM experiments, not unknown
external use or factory production coverage. Future data expansion must exclude
these reserved lot groups. Keep evaluation deferred until recipes are frozen.

Reproduce the reservation from the same hash-pinned trusted source:

```bash
uv run --frozen python scripts/reserve_holdout.py \
  --source data/source/LSWMD.pkl --dataset data/wm811k-lot17 \
  --output outputs/final-holdout-v1 --trust-pickle
```

## Launch specification and cost

Repository: `jasonjiang8866/waferVLM`; workflow `gpu.yml`, manual `main` only;
`confirm_paid=true`, `expected_sha` equal to the reviewed current main commit.
Publish the image from the approved workload source via `image.yml`, then pin its
immutable digest/source before launch. Preflight rejects workload/image drift.
No account credential enters the Pod; `RUNPOD_API_KEY` stays in Actions Secrets.

| Item | Bound / decision |
|---|---|
| Total authorization | US$30 /1,100 cumulative minutes |
| Reconciled history reserve | US$5.38 /457 minutes |
| Worker continuation stage | 160 minutes |
| Controller work / cleanup | 185 /10 minutes |
| Actions timeout | 205 minutes including setup/upload |
| Maximum planned cumulative runtime | 652 minutes |
| Maximum planned cumulative cost | US$9.276518, including ephemeral disk |
| Storage | 60GB ephemeral; no persistent volumes |
| GPU preference | RTX 4090, MIG24GB, RTX3090, RTX5090 |
| Existing GPU-only hourly caps | 0.89 /0.69 /0.50 /1.19 USD in that order |
| Host selection | CUDA >=12.8, higher supported versions allowed |

The 4090 short probes measured 12.82/12.98s per new step; MIG measured 44.28s.
100 updates alone imply about 21.5 minutes on the measured 4090 environment or
73.8 minutes on the measured MIG environment. These are rough planning estimates,
not end-to-end guarantees. Loading, data preparation, all three full validation
passes, export/download and cleanup are included in the separate lifecycle cap.
Per-run quotes, capacity, account guards and owned-Pod inventory are refreshed
before the single create attempt. Unknown creates are reconciled, not retried.

## Acceptance and recovery

The controller independently verifies archive paths/limits/checksums, image/
source/data/model identity, full parent and 250→300 adapter/optimizer/scheduler/
RNG chains, exactly 50+50 finite nonzero-gradient updates and every effective LR.
It requires complete resumable 250/300 states, exact validation-manifest identity,
160 unique matching sample IDs/labels/image hashes per evaluation, recomputed
classification metrics and the declared selection rule. Test outputs are rejected.
Separate controller and workflow-run cleanup require three absence observations.

Before relying on a result, retrieve/hash the archive, inspect validation and
training curves, reconcile its authorization marker/spend/runtime, verify both
cleanup paths and preserve full evidence in a Release. Retain checkpoint-200 if
quality does not improve. No automatic retry or unbounded tuning loop is enabled.
