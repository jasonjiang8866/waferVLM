# DINOv3 cross-seed confirmation and prespecified holdout comparison

The seed-17 development sweep selected frozen-backbone linear LR **0.003** and
vision Q/V LoRA LR **0.0001**, with head LR **0.001**. Freeze those settings before
training seeds **42 and 123**. Reuse the two verified seed-17 exports, including
their original predictions as historical development evidence. Do not rerun its
LR sweep or count inference repetitions as training seeds.

## Training and primary comparison

The original `configs/dinov3.json` remains unchanged: 790 training maps,
batch size 8, 10 epochs / 990 updates / 7,900 presentations per trial, AdamW,
FP32 parameters with BF16 autocast, rank-8 Q/V LoRA (alpha 16, dropout 0),
backbone eval mode in both arms. Preserve complete 256px maps, ImageNet
normalization, no resize/crop/augmentation, CLS + mean patch pooling excluding
registers, and eight-way argmax. The original per-trial rule selects highest
validation macro-F1, ties to the earliest epoch. Seed 17 selected linear epoch 10
and LoRA epoch 5; later seeds keep the 10-epoch budget and selection rule, rather
than copying either epoch or applying Qwen's step-200/250 protocol.

The primary DINO pairing is LoRA minus linear within each training seed. Report
all three seeds and identify seed 17 as exploratory; seeds 42/123 are confirmation.
No best-seed headline. Qwen comparisons include different architecture,
pretraining, training exposure, adaptation location and generative output.

## Validation: 18 fresh processes

For each of six fixed exports, run three independent processes on the exact
160-map validation manifest. Batch size **8** is fixed for all 18 processes,
matching the historical selected outputs. Report pass 1 as primary and give the
number and IDs of label changes for passes 2 and 3. The historical batch-1 reloads
remain separate and are not counted toward 18. Preserve all 2,880 predictions,
eight probabilities in canonical order, selected tensor hashes, training
histories, software/image/overlay identity, timings and peak allocated memory.

No DINO holdout image or prediction is accessed in this phase. The validation
worker overlay does not include the holdout manifest or holdout evaluator.

## Freeze before holdout

After verifying validation results and independent cleanup, commit
`docs/evidence/dinov3-six-endpoints.json` containing the six exact export hashes,
all training-export file hashes, selected epochs, original source/image identities,
and the immutable result archive ID/digests. Record the already-frozen plan hash.
The holdout dispatch refuses to run without this file. Reconcile the completed
validation run into the separate DINO ledger before another allocation.

The holdout is **PR #53's same 148 WM-811K maps / 143 lot groups**, manifest
SHA-256 `f08050ef4fcaca3b9c35ad29905a63957a375af7a024b69e0281968d06f4564f`.
Qwen's results are already public: this is an additional prespecified model
comparison on that same holdout, not a wholly unseen study design. Do not select
learning rate, epoch, seed or headline checkpoint using these results.

Each of six DINO models is evaluated once, yielding **888 predictions**. Report
accuracy, macro-F1, each class recall, confusion counts, total and per-map
inference time, forward-only time, and peak allocated GPU memory. DINO's inclusive
timer includes image verification/decode/preprocessing/transfer/forward, excludes
model loading, uses synchronized CUDA boundaries and no warmup. The existing Qwen
timer is generation-only; no cross-model speedup claim is permitted.

Within each seed, use the same lot-group paired bootstrap as Qwen: **10,000**
draws, NumPy RNG seed **20261110**, shared lot multiplicities for both models,
fixed eight classes, undefined F1 = 0, percentile 95% intervals for LoRA − linear
accuracy and macro-F1. Also pair DINO seed-17 LoRA with the published Qwen
seed-17 step-250 predictions, with exact per-image identity checks. Report all
seeds and both development and holdout conclusions, including disagreements.

## Execution, budget and cleanup

Reuse the verified DINO image
`ghcr.io/jasonjiang8866/wafervlm@sha256:d64ef98a018ce25d41add6a8c3a65225ab83f22addb16ab47959e0c5b01ab329`.
Its original training/data/recipe/dependency files must remain byte-identical to
source `c255bec677ba4b525e4997af60d9020592e15a46`. Only reviewed study orchestration,
evaluation modules and manifests are supplied in a compressed SHA-256-verified
overlay; both source identities and the overlay digest are recorded. Official
backbone weights remain embedded, hash-verified and loaded offline. A short-lived
artifact URL is delivered by authenticated POST when the worker requests it;
the Pod never receives GitHub/HF/RunPod account credentials.

The original DINO allowance remains **US$10 / 500 minutes cumulative**, separate
from Qwen. Before this stage, verified estimated usage was **US$0.648122 / 29.4332
minutes**. Every subsequent dispatch is fail-closed on unreviewed history and
must add its actual intent-to-absence bound and independent cleanup receipt.
The controller rechecks live inventory, image provenance, global catalog and
price before one create request. RTX 4090 is preferred; RTX 5090 is the approved
fallback. Host CUDA uses **>=12.8**. No persistent volume is created.

Each attempt allows 45 minutes of work plus 10 minutes cleanup, with GPU caps
US$0.89/hour (4090) and US$1.39/hour (5090), temporary disk included and a US$1
uncertainty reserve against the cumulative cap. At the 5090 cap the maximum
estimate is US$1.2824. This is a conservative estimate, not a provider invoice.
An unknown create result is reconciled by owned Pod name before any retry.
Controller `finally`, an always-run cleanup step, and a separate `workflow_run`
cleanup verify consecutive inventory absence; workflow success alone is not
accepted as cleanup or experiment completion.

Entrypoints: `dinov3-study.yml` (explicit main-only manual dispatch with exact SHA)
and `dinov3-study-cleanup.yml` (independent cleanup). This document and code are
preparation; only verified run artifacts establish completed counts.
