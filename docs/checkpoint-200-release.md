# Checkpoint-200: lower-LR continuation and preserved release

The lower-LR continuation completed on **2026-11-09 at 03:16 SGT** in
[run 42502913001](https://github.com/jasonjiang8866/waferVLM/actions/runs/42502913001).
It restored the original checkpoint-100, ran 100 additional optimizer steps at
constant LR **5e-5**, and selected checkpoint-200 by validation macro-F1 before
evaluating the test split. This is an exploratory, single-seed experiment.

## Results

| Same 160 test images | Original model | Step 100 | Selected step 200 |
|---|---:|---:|---:|
| Accuracy | 11.875% | 53.125% | **64.375%** |
| Macro-F1 | 0.028107 | 0.521659 | **0.636059** |
| Invalid outputs | 11/160 | 0/160 | **0/160** |

Step 200 correctly classified **103/160**, versus **85/160** at step 100:
18 more correct predictions and **+11.25 percentage points** of accuracy.
The original-model and step-100 test scores are historical results from
[run 42385615853](https://github.com/jasonjiang8866/waferVLM/actions/runs/42385615853);
the selected-test predictions in this release are the step-200 results.

| Validation (160 images) | Step 100 | Step 150 | Step 200 |
|---|---:|---:|---:|
| Accuracy | 64.375% | 69.375% | 72.500% |
| Macro-F1 | 0.640676 | 0.686280 | 0.717461 |
| Answer-token loss | 0.270583 | 0.220790 | 0.211074 |
| Invalid outputs | 0/160 | 0/160 | 0/160 |

Segment training losses were 0.253191 (steps 101–150) and 0.205033 (151–200).
All 100 new effective optimizer LRs were 5e-5; all 100 gradient checks were
finite and nonzero. AdamW moments/step counts, scheduler, RNG and data position
were restored. The [constant-2e-4 control](continuation-control-42431025521.md)
had selected checkpoint-100; its step-200 test performance was not measured.

## Download and verify

[Release checkpoint-200-v1](https://github.com/jasonjiang8866/waferVLM/releases/tag/checkpoint-200-v1)
preserves the results outside Actions retention. It is marked **pre-release**
because this is an experimental checkpoint, not a production qualification.
The original Actions artifact expires **2026-12-09 03:15:51 SGT**
(`2026-12-08T19:15:51Z`); that expiry does not apply to the Release assets.

| Release asset | Purpose |
|---|---|
| `wafervlm-qwen3.5-0.8b-step200-adapter.tar.gz` | Inference adapter, processor/tokenizer files, run metadata and checksums |
| `wafervlm-run-42502913001.zip` | Unmodified complete Actions result package, including resumable step-150/200 checkpoints, predictions, logs and cleanup evidence |
| `release-manifest.json` | Source/model/data identities, exact metrics, asset sizes and SHA256 digests |
| `MODEL_CARD.md` | Standalone usage, results and limitations |
| `SHA256SUMS` | SHA256 checks for the four files above |

Download all five assets into one directory and run:

```bash
sha256sum -c SHA256SUMS
tar -xzf wafervlm-qwen3.5-0.8b-step200-adapter.tar.gz
```

The inference package preserves every original adapter file plus `run.json` and
`training.json` byte-for-byte. Its `source-checksums.json` is the original
step-200 manifest; its new `checksums.json` covers only the packaged files, so
the evaluator can verify it without requiring omitted optimizer files.
The full result ZIP is unchanged. Base-model weights and WM-811K images/raw
data are not included; obtain them separately using the pinned recipe.

## Use the adapter

From a checkout of this repository, install the locked training dependencies and
prepare the audited dataset as described in the root README. Assuming the
inference archive was extracted in the checkout:

```bash
uv sync --frozen --extra train
uv run --frozen --extra train wafervlm evaluate \
  --dataset data/wm811k-lot17 \
  --adapter checkpoint-200/adapter \
  --split validation --device cuda \
  --output outputs/released-200-validation
```

The evaluator downloads the pinned base model, verifies the adapter's model/data
identity and checksums, and produces predictions and metrics. Use a new output
directory. The default above uses validation; repeated test evaluation should
not become a tuning loop.

## Recover the full training state

```bash
mkdir recovered-run-42502913001
unzip wafervlm-run-42502913001.zip -d recovered-run-42502913001
tar -xzf recovered-run-42502913001/experiment.tar.gz \
  -C recovered-run-42502913001
```

The resumable checkpoint is
`recovered-run-42502913001/results/continuation/step200/checkpoints/checkpoint-200`.
It retains the adapter, `optimizer.pt`, `scheduler.pt`, `rng_state.pth`,
`trainer_state.json` (global step 200), and `training_args.bin`. Keep the full
`step200` directory together because resume also verifies its parent metadata
and checksums. Load serialized training state only from this trusted release.

For a separately planned continuation, this command restores step 200 and targets
**300 total optimizer steps**, retaining LR 5e-5:

```bash
uv run --frozen --extra train wafervlm train \
  --dataset data/wm811k-lot17 --output outputs/resume-300 \
  --resume-from recovered-run-42502913001/results/continuation/step200/checkpoints/checkpoint-200 \
  --steps 300 --seed 17 --accumulation 8 --rank 8 \
  --learning-rate 0.00005 --max-length 1024 --device cuda
```

This is a recovery example, not a new RunPod dispatch or experiment authorization.
The inference-only archive is insufficient for optimizer-state resume.

## Reproducibility and validation

- Release tag/controller commit: `4ff0fb4c878decb0afaec0d433215bb7e0ead274`.
- Worker source: `ef254921225ded9bbb3aec818de177c9d718db73`.
- Image: `ghcr.io/jasonjiang8866/wafervlm@sha256:84c6d4a5734421ecfc2cfc7c1b2d9bce5d855edda7b925e73238009825b34b6e`.
- Base: `Qwen/Qwen3.5-0.8B`, revision `2fc06364715b967f1860aea9cf38778875588b17`.
- Dataset SHA256: `c588b3ef59d9dba57cfc745ec20db903e6ff2dcc75504112348f8b9f1ad3ab68`.
- Original artifact ID: `13630761886` (`wafervlm-gpu-42502913001-1`).
- Original ZIP SHA256: `b039ca64d37b2d343a0f94da5e26e741e3b463ee48af2ab5877659c555841490`.
- Inner `experiment.tar.gz` SHA256: `eb83e251e3a27a59ae643d89f7f98a7c620b6b225fd7e9926a4190c6f7226508`.
- Step-200 adapter weights SHA256: `8cda3bcfcacdb62a4f6649d17fa415749af01aedd663986322fa7550ed328c55`.
- BF16 language-only LoRA: r8, alpha16, frozen vision tower, answer-token loss,
  seed17, batch1 × accumulation8, maximum length1024.
- Torch2.10.0/CUDA12.8, Transformers5.3.0, PEFT0.18.1, Accelerate1.12.0.
  Optional DeltaNet fast kernels were not installed.
- GPU: NVIDIA RTX PRO 6000 Blackwell Server Edition MIG 1g.24gb, US-PA-1.

The outer digest matches GitHub's artifact digest. The repository verifier accepts
all 66 files (236,129,647 unpacked bytes), including nested hashes, provenance,
resume chains, LR/gradient traces and evaluation completeness. All four sets of
classification metrics were also recomputed from their per-image predictions.
The inference bundle was separately reopened and every packaged hash checked.
This preservation task does not claim a new GPU inference or resume execution.

The worker's execution acceptance and measured quality are separate: the generic
`quality_accepted` field remains false; this report describes the observed scores,
not a production acceptance decision.

## Runtime, cost and cleanup

Continuation workload elapsed 129.55 minutes. The Pod lifecycle billing bound was
133.21 minutes. The cumulative conservative estimate is **US$5.051749 / 433.21
minutes**, including prior-attempt reserves, within the US$20 / 500-minute cap.
This is a controller estimate, not an invoiced charge.

Pod `mg6kq4xbekfmlf` passed three consecutive absence checks; an
[independent cleanup run](https://github.com/jasonjiang8866/waferVLM/actions/runs/42514159527)
also verified absence. No persistent volumes were created. Subsequent paid
dispatches remain blocked until this attempt's authorization marker and ledger
are reconciled; publishing this release changes no compute settings.

## Scope and model card

The task is closed-set generation of one of eight labels from a rendered WM-811K
map and a fixed question. Greedy generation uses at most 16 new tokens with
thinking disabled. Exact normalized labels count as valid; explanatory text,
unknown labels and token-limit truncation fail. Zero invalid outputs here does
not imply general instruction-following perfection or accurate classification.

There are 790 training, 160 validation and 160 test images, with lot separation
and duplicate/contradictory-map handling. Normal and unlabeled samples are
excluded. This does not measure normal-wafer false alarms, unknown defects,
SEM inspection, or process root-cause diagnosis. The vision tower is frozen and
the model is a generative VLM with LoRA, not a trained sequence-classification head.

One seed and a small, repeatedly observed development split cannot establish
production generalization. Further tuning should use validation; a final claim
needs a fresh, untouched holdout and replication. These scores are not directly
comparable to NVIDIA's different demonstration split.

Repository code is MIT. The upstream Qwen model retains Apache-2.0 terms;
WM-811K retains its own data terms. The repository license does not relicense
upstream materials. This adapter is an experimental research artifact.
