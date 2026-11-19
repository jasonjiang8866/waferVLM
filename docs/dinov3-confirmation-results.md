# DINOv3 cross-seed development confirmation

The [2026-11-19 run](https://github.com/jasonjiang8866/waferVLM/actions/runs/44537796019) completed the four new training jobs, reused the two verified seed-17 exports, and produced **18/18 independent-process validation passes / 2,880 predictions**. **DINO holdout completion is still 0/888**. The archive was downloaded, its outer and inner SHA-256 verified, and all predictions and training exports independently checked on CPU. [Independent cleanup](https://github.com/jasonjiang8866/waferVLM/actions/runs/44540357834) also succeeded.

## What the complete 160-map validation set supports

| Training seed | Linear accuracy | LoRA accuracy | LoRA − linear (pp) | Linear macro-F1 | LoRA macro-F1 | Difference |
|---|---:|---:|---:|---:|---:|---:|
| 17 | 83.750% | 87.500% | +3.750 | 0.836449 | 0.873125 | +0.036676 |
| 42 | 81.875% | 86.875% | +5.000 | 0.810348 | 0.867664 | +0.057316 |
| 123 | 81.250% | 86.250% | +5.000 | 0.808981 | 0.861604 | +0.052623 |

These are each fixed model's **first** new batch-size-8 process. All twelve subsequent reproducibility passes have **zero changed labels and empty changed-ID lists** relative to their own first pass. This is three training seeds, not eighteen. Seed 17 is exploratory; seeds 42 and 123 confirm the already-frozen learning rates and 10-epoch selection protocol. Every training export verifies 990 updates / 7,900 presentations and the highest-validation-macro-F1, earliest-tie selection rule. Selected epochs are linear 10/9/10 and LoRA 5/5/3 for seeds 17/42/123.

The two confirmation seeds both reproduce LoRA's overall gain. Their loc recall rises from **35% to 65%** and **45% to 70%**. Seed 17's loc recall falls from **70% to 65%**, so the overall gain is not a gain on every class. Training seed 17 used RTX 5090; seeds 42/123 used RTX 4090. Each within-seed pair shares a training GPU type, but between-seed variation does not isolate training randomness from that hardware difference. All eighteen new validation processes ran on the same RTX 4090.

[All six metrics, class recalls, four-way paired counts, timing and memory](showcase/dinov3-confirmation-validation/README.md) · [Six confusion matrices](showcase/dinov3-confirmation-validation/confusion-matrices.png) · [All 160 paired rows](showcase/dinov3-confirmation-validation/paired-per-image.csv) · [Full predictions and eight-class probabilities](showcase/dinov3-confirmation-validation/predictions)

## Qwen's common errors and the difficult loc maps

The six Qwen first-pass diagnostic files are pinned to PR #54 commit `ae2cd56f7047c3015a34711324072f90f70dac2e`, while the overall reference remains the original published Qwen seed-17 step-250 predictions. IDs, image SHA-256, labels, completeness and source-file hashes are checked before comparison.

Of the sixteen maps missed by all six Qwen checkpoints, the three linear models rescue **7, 6, 5**; the three LoRA models each rescue **the same six maps**, including **the same three loc maps**: `0dba1dc3f2c5cf63eda306ba`, `38dd30111d9a15f20ad272cc`, `dd855c3b619057c30af18ade`. The other LoRA rescues are one scratch, one edge-loc and one donut. These selected cases support a diagnostic observation; overall performance uses all 160 maps.

Four loc maps remain wrong for **all six DINO models**:

| ID | Visible morphology from the original image review | DINO predictions across all three seeds |
|---|---|---|
| `50773a965c423a8290e617f2` | Large solid center-right failure patch; passing crescent on the left | Both arms: near-full |
| `aac6e9a0b3b5aa660ce62121` | Dense left/lower-edge failures reaching the wafer boundary | Both arms: edge-loc |
| `be8bdd8417b86a07da6a61f2` | Narrow horizontal failure band extending inward from the left edge | Both arms: scratch |
| `c882bea412b6513af0935266` | Elongated diagonal upper-right cluster near the edge | Linear: edge-loc; LoRA: scratch |

These descriptions suggest overlapping visual patterns; they neither prove what features caused a decision nor justify relabeling the source data. [CPU-only original prediction comparison and real ten-map gallery, PR #56](https://github.com/jasonjiang8866/waferVLM/pull/56) · [All sixteen cases with six Qwen and six DINO predictions](showcase/dinov3-confirmation-validation/common-errors-16.csv)

## Preserve the historical primary result

The original seed-17 selected outputs remain **134/160 linear and 139/160 LoRA** in the historical report and CPU-only PR #56. The new first-pass LoRA score is 140/160: `e400f55a09848e7ba9bed36c` (true scratch) changes from edge-loc to scratch, and `f5ca2de48af4e569a3ce1c2b` (true edge-ring) changes from scratch to edge-loc, still wrong. Thus **two predictions change, with a net gain of one correct map**. Linear has no historical-to-new changes.

The historical batch-size-1 reload is separate from the eighteen same-configuration passes. Historical inference used RTX 5090 and this run uses RTX 4090; the comparison does not identify a single cause of the differences. No favorable repeat, new weight, seed or endpoint is chosen using these observations.

## Freeze before unseen-lot inference

[Six endpoint freeze](evidence/dinov3-six-endpoints.json) records every selected weight SHA-256, training checksums, the immutable input artifact, exact manifests, preprocessing, inference configuration and statistical method. It is committed before any DINO prediction on the 148-map / 143-lot holdout. All six models are prespecified; each will be evaluated once, for 888 rows. This is an additional prespecified comparison on the same holdout whose Qwen results were already public, not an entirely unseen study design.

The primary holdout contrast is LoRA minus linear **within each seed**. The paired lot-group bootstrap uses 10,000 resamples, RNG seed 20261110, fixed eight classes and zero for undefined F1. A separate fixed comparison uses seed-17 DINO LoRA versus published Qwen seed-17 step 250. DINO's 10-epoch classification protocol and Qwen's 250-step text-generation protocol are not compute matched. See the [frozen plan](dinov3-confirmation-plan.md).

## Cost, cleanup and reproducibility

This successful RTX 4090 run (EUR-IS-1, US$0.89/hour) used a conservative **843.887009-second / US$0.2107206** intent-to-verified-absence bound. Including the original DINO experiments and all five unsuccessful confirmation attempts, cumulative usage is **US$1.0627674 / 57.1092 minutes** under the existing **US$10 / 500-minute DINO-only** allowance. Remaining allowance is about **US$8.9372 / 442.8908 minutes**; estimates are not invoices.

The originating workflow recorded three consecutive absences for Pod `kvgd8runvm4b7l`; independent cleanup recorded twelve. No persistent volume was created. [Exact receipt, artifact hashes and cleanup evidence](evidence/dinov3-validation-complete.json) · [Cumulative ledger](../runpod/dinov3-study-ledger.json) · [Earlier attempt diagnosis](dinov3-study-recovery.md)

The selected weights and complete raw run are in [artifact 14538331200](https://github.com/jasonjiang8866/waferVLM/actions/runs/44537796019/artifacts/14538331200). Git retains all eighteen prediction files and eight probabilities per row. To regenerate from the verified archive:

```bash
uv run --frozen python -m analysis.dinov3_confirmation_report \
  /path/to/unpacked/results validation /path/to/launch-spec.json /new/output/directory
```
