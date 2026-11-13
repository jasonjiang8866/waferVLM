# DINOv3 visual baseline

Status: implementation and CPU compatibility tests; **no pretrained DINOv3 experiment has run**. No DINOv3 accuracy, cost, GPU speed or final-holdout result is available yet. This is separate from the Qwen seed-confirmation work.

## Question and fixed arms

Does adapting a compact vision backbone improve eight-class wafer-map recognition, and how does the resulting classifier compare with Qwen3.5-0.8B language-side LoRA?

| Arm | Trainable weights | Purpose |
|---|---|---|
| DINOv3 linear | Eight-class linear head only | Measure frozen feature transfer |
| DINOv3 LoRA | Attention Q/V LoRA plus the same head architecture | Measure vision-adaptation gain over the linear baseline |
| Existing Qwen checkpoint 250 | Language-side LoRA; frozen vision | Compare complete classification systems |

Use official `facebook/dinov3-vits16-pretrain-lvd1689m`, pinned to revision `114c1379950215c8b35dfcd4e90a5c251dde0d32`. [Meta recommends testing frozen features first](https://github.com/facebookresearch/dinov3/blob/main/MODEL_CARD.md). The [official Hugging Face weights](https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m) require approved access. Random tiny models in CI establish code compatibility, not pretrained-model quality.

The LoRA-minus-linear paired difference measures adaptation within DINOv3. DINOv3-minus-Qwen includes architecture, pretraining, model size, adaptation location and output-head differences. It is not an isolated comparison of LoRA algorithms. An eight-way argmax has zero invalid labels by construction; that is not evidence of better visual understanding.

## Data, transforms and selection

- Reuse the exact audited **790 train / 160 validation / 160 historical test** dataset and fixed split seed 17. Identity hashes are in [the recipe](../configs/dinov3.json). Audit historical test file integrity without evaluating it. No holdout entrypoint exists in this implementation.
- Preserve the complete 256x256 RGB rendering. Normalize with the declared ImageNet mean/std; no resize, crop, rotation, color jitter or changes to die semantics. This size is divisible by the 16px patch size.
- Concatenate the CLS token and mean of patch tokens, explicitly excluding the four register tokens. Train an eight-class linear layer with unweighted cross entropy.
- Keep the backbone in `eval()` in **both** arms to disable positional jitter, stochastic depth and dropout. Gradients still flow into the LoRA weights. Head initialization is identical across arms at the same seed; base weights are always the same pinned pretrained checkpoint.
- LoRA rank 8, alpha 16, dropout 0; exact attention Q/V targets only. AdamW, constant LR, zero weight decay, gradient clipping 1.0; FP32 parameters with BF16 autocast on CUDA. The existing locked Torch/Transformers/PEFT versions remain unchanged.
- Batch size 8, 10 complete epochs, no dropped final batch: **7,900 examples / 990 optimizer updates** per complete run. This exposure exceeds Qwen's 250-step history; disclose it, plus actual training time and trainable parameter counts, rather than claiming compute matching.
- Evaluate all 160 validation examples at each epoch. Select highest macro-F1, with earliest epoch winning ties. Preserve every epoch's loss/metrics/exposure and the selected head/adapter tensors. Runs cannot overwrite prior outputs. Checkpoints are inference exports, not resumable optimizer snapshots.

### Bounded development and confirmation

Use seed **17** for a three-LR sweep per arm: linear `{0.001, 0.003, 0.01}`; LoRA `{0.00005, 0.0001, 0.0002}` with head LR 0.001. Freeze each arm's LR after that sweep; ties choose the lower LR. Then run seeds **42 and 123** at that frozen LR and epoch budget. This is at most ten training runs, not ten new spending allowances. Do not tune each confirmation seed separately.

Keep the existing Qwen independent-seed study distinct. Its historical tuning effort differs from the DINO sweep, so the Qwen comparison is exploratory. The six-run DINO summary requires all three seeds for both arms and an identical recipe within each arm; it reports all seeds, sample standard deviations and paired LoRA-minus-linear differences, never the best seed alone. Learning-rate selection is a recorded analysis decision before confirmation dispatch; the CLI does not autonomously select or launch trials.

The reserved 148-image holdout remains untouched. Any extension of the existing seven-model final study to DINOv3 must be declared before opening that holdout, with frozen endpoints and a separate complete final-analysis plan.

## Run locally or inside a prepared GPU worker

```bash
uv sync --frozen --extra train

# Run on CPU BEFORE allocating paid compute. Uses the standard HF credential store;
# do not put tokens in command arguments, images, logs, or repository files.
uv run --frozen --extra train python scripts/run_dinov3.py preflight \
  --output results/dino-weights-preflight

# These commands train real models when executed; each process is time-bounded.
uv run --frozen --extra train python scripts/run_dinov3.py train \
  --dataset data/wm811k-lot17 --mode linear --seed 17 --learning-rate 0.001 \
  --output results/dino-linear-17-lr001 --timeout-seconds 3600

uv run --frozen --extra train python scripts/run_dinov3.py train \
  --dataset data/wm811k-lot17 --mode lora --seed 17 --learning-rate 0.00005 \
  --output results/dino-lora-17-lr00005 --timeout-seconds 3600

# Fresh process, reloaded selected tensors, batch-size-one validation/timing.
uv run --frozen --extra train python scripts/run_dinov3.py evaluate \
  --dataset data/wm811k-lot17 --trained results/dino-lora-17-lr00005 \
  --output results/dino-lora-17-eval --timeout-seconds 3600

# CPU-only paired images/counts, using the released Qwen checkpoint-250 predictions.
uv run --frozen python scripts/compare_dinov3.py compare \
  --dataset data/wm811k-lot17 --dinov3-result results/dino-lora-17-lr00005 \
  --output results/qwen-dino-17

# Replace these six directories with the frozen-recipe run outputs.
uv run --frozen python scripts/compare_dinov3.py summarize \
  --dataset data/wm811k-lot17 --results \
  results/linear-17 results/linear-42 results/linear-123 \
  results/lora-17 results/lora-42 results/lora-123 --output results/dino-three-seed
```

Training emits `run.json`, `history.json`, `training.json`, `status.json`, `selected.safetensors`, selected `predictions.jsonl`/`metrics.json` and complete checksums. Every prediction retains ID, ground truth, image hash, predicted class and probabilities in the canonical label order. CUDA runs additionally fingerprint the cached base weights. A failure or timeout is not an accepted result.

The comparison generates two wafer-thumbnail confusion matrices (**X=true, Y=predicted**, with an invalid row) and four paired-example categories: both correct, only Qwen correct, only DINOv3 correct, both wrong. Every example contributes to counts; thumbnails use lowest IDs. No fabricated inference graphics or accuracy placeholders are published in README.

Batch-size-one timing includes image verification/decode/preprocessing/transfer and forward execution, with synchronized CUDA boundaries and no warmup. The first sample includes cold effects. Historical Qwen timing measures generation only, so these numbers must not be used for a speedup claim. Rerun both models on the same host with matching timing boundaries before comparing deployment latency. Training peak memory and wall time are recorded independently.

## GPU execution boundary

Prefer **RTX 4090** and select host CUDA capability **>=12.8** through the existing numeric lower-bound logic. The worker image reuses a pinned CUDA environment; accepting a newer host does not change the container runtime. The CPU-tested image workflow builds on PRs and publishes only through explicit `workflow_dispatch` on main; it never creates a Pod.

Before paid execution: complete official-weight access/preflight, freeze source and image digest, reconcile the experiment budget/current provider quote and active Pods, and connect this workload to the reviewed RunPod controller with aggregate accounting and independently verified cleanup. The current GPU workflow still launches Qwen workloads; **do not dispatch it expecting DINOv3**. A local process timeout does not terminate a RunPod Pod or enforce the full experiment's spending cap. No existing authorization marker is reset by this change.

## Verification

```bash
uv run --frozen --extra train python -m unittest discover -s tests -p 'test_dinov3*.py' -v
```

Tests cover actual tiny DINOv3/PEFT gradients, frozen weights, register-token exclusion, border preservation, head initialization, training/selection/reload, failed-result preservation, checksum tampering, complete prediction joins, invalid labels, six-run recipe consistency, and data-derived PNG rendering. Pretrained weights, real CUDA quality/speed, and RunPod lifecycle integration require separate verification.
