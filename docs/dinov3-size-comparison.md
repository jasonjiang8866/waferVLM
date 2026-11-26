# DINOv3 Small and Base on wafer maps

The accepted [Small results](dinov3-holdout-results.md) cover three seeds and two adaptation methods. This extension compares the two official LVD-1689M backbones at the same 16-pixel patch size. The [registry](../src/wafervlm/dinov3_backbones.py) pins each Hub revision, the SHA-256 of its configuration and safetensors files and its parameter count.

| Backbone | Parameters | Feature width | Classification head input |
|---|---:|---:|---:|
| ViT-S/16 | 21,596,544 | 384 | 768 |
| ViT-B/16 | 85,660,416 | 768 | 1,536 |

## CPU acceptance

With approved Hugging Face access and the training dependencies installed, run the preflight in a fresh output directory:

```sh
python scripts/run_dinov3.py preflight --backbone-size base --output results/base-preflight
```

The image check stages the license and pinned weights with a build-only token. A separate offline step loads each backbone, runs a synthetic 256×256 input through each classifier, exports its trainable parameters and compares logits after loading a fresh model. The local compatibility test uses generated fixtures with both official feature widths. Neither check measures wafer accuracy.

Training accepts the same size flag. The default training backbone is Small. Evaluation reads the identity from the sealed training export and checks any explicit size against it. The original Small replay workflow uses its pinned image sources.

## Development comparison

Use the frozen 790/160 split, full 256×256 images, batch 8, ten epochs (990 updates, 7,900 presentations), earliest epoch at maximum validation macro-F1. Pass the frozen linear LR 0.003 and LoRA adapter LR 0.0001 explicitly. LoRA uses head LR 0.001, rank 8, alpha 16 and Q/V in each layer.

1. On RTX 4090, run both sizes and both adaptation methods at seed 17. The historical seed-17 Small training used a 5090.
2. Run Base at seeds 42 and 123 for both methods. Pair with the corresponding frozen 4090 Small models. Report all seeds. Same update budget allows the larger model to consume more compute.
3. Compare Base-LoRA minus Small-LoRA macro-F1 within seed. Report Base-linear minus Small-linear, accuracy, per-class recall, rescued and newly missed loc IDs, trainable parameters, elapsed training time, throughput, peak allocation and lifecycle cost.
4. Freeze endpoint hashes and analysis before scoring on the published 148 maps / 143 lots. Use 10,000 paired lot bootstrap resamples with seed 20261110 and eight fixed classes, scoring undefined F1 as zero. Independent confirmation of a size benefit requires a fresh lot-disjoint set checked for duplicate images.

This change covers CPU compatibility. A GPU launch and size-comparison accuracy results are subsequent work.

## Seed-17 GPU validation

On the DINO study workflow select `size17`, the exact main SHA and `NVIDIA GeForce RTX 4090`. The existing DINO ledger, inventory checks, one-create receipt, independent cleanup and 45-minute work / 10-minute cleanup bounds apply. Small comes from the pinned historical image. Actions stages and hashes Base before allocation, then sends the two pinned files to the temporary Pod through the authenticated chunk endpoint. The credential stays on the runner. The overlay digest binds the new training source separately from the underlying image.

All four arms share one Pod. Primary metrics use the selected epoch's batch-8 development predictions. Acceptance rechecks the 160 sample identities, all eight probabilities, 990 updates, 7,900 presentations, earliest best epoch and selected weight hashes. The archive retains the four exports. The report includes both paired accuracy tables, macro-F1 deltas, loc IDs in each outcome, timing and peak training allocation.
