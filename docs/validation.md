# Validation record — 2026-11-07

Initial target: Qwen3.5-0.8B wafer-map SFT baseline.

## Local checks

- Python 3.12.14; Torch 2.10.0+cu128; Transformers 5.3.0; PEFT 0.18.1; Accelerate 1.12.0.
- 15 unit/integration tests passed on CPU. Includes a real randomly initialized tiny Qwen3.5 with one DeltaNet and one full-attention layer, image tensors, nonzero LoRA gradients, frozen vision, image sensitivity, and adapter save/reload equality.
- Data tests cover whole-lot splits, duplicate-connected lots, conflicting labels, rendering palette/aspect ratio, source digest checks, strict parsing, invalid-output metrics, tampering and paired-comparison identity.
- Real pinned pretrained processor/tokenizer: all eight labels passed exact answer-only masking; image tensors and the explicit non-thinking prefix verified. No pretrained model weights were needed for this check.
- Real Trainer integration passed two optimizer steps with gradient accumulation, gradient checkpointing, finite/nonzero adapter gradients, checkpoint state and adapter export. Uses the official tokenizer/processor with a tiny random model and explicitly synthetic fixtures.

These are compatibility/protocol tests. Random fixtures are not WM-811K, and the tiny randomly initialized model is not the pretrained 0.8B checkpoint. No wafer accuracy is inferred from them.

## Real-data and preflight integration

- The public Kaggle WM-811K mirror was downloaded and SHA256-pinned. pandas 2.3.3 loaded the historical pickle without a compatibility shim.
- Strict preparation detected contradictory target labels on 3 identical maps (6 rows). An explicit exclusion policy now quarantines all copies, while the default generic importer still fails on conflicts. Checksummed quarantine membership is audited against selected splits.
- Full real-data preparation and PNG/manifest audit passed: **790 train / 160 validation / 160 test**, all eight classes, disjoint lots/raw maps/rendered images. See [data audit](data-audit.md) and its committed machine-readable evidence.
- **18 protocol tests passed locally** after the changes, including conflict opt-in, quarantine enforcement, cached archive tampering, incomplete-download cleanup and provenance in a container without Git. The two model integration tests were not rerun locally in this stage; CI runs them with the locked training dependencies.
- Container review found that provenance previously attempted to execute `git` unconditionally. The slim image has no Git binary; provenance now correctly falls back to the build's `WAFERVLM_SOURCE_SHA`.
- The pinned Broker contract passed locally with no network calls: numerical CUDA floor 12.8 accepts 12.9/12.10/13.0; the actual adapter emits `minCudaVersion`, and missing budget prevents selection. No live RunPod request was made locally.
- PR checks on integration commit `d92e5c0c9e3cfe7e55bc5db45ff28a1cd5c86f7b` passed: [real-data reproduction](https://github.com/jasonjiang8866/waferVLM/actions/runs/42339253021), [Docker build](https://github.com/jasonjiang8866/waferVLM/actions/runs/42339253261), [model/Trainer compatibility](https://github.com/jasonjiang8866/waferVLM/actions/runs/42339252671), and [offline Broker contract](https://github.com/jasonjiang8866/waferVLM/actions/runs/42339252670). The data workflow independently downloaded the pinned mirror and matched the committed evidence exactly. The subsequent missing-Git provenance fix has its own local regression test and triggers fresh CI; check the latest commit before merging.

## Not yet verified

- Pretrained 0.8B CUDA forward/backward, memory/time, full training, or quality improvement.
- GHCR publication and a live provision/monitor/collect/delete RunPod controller. Docker validation is delegated to the PR image build because the local environment has no Docker executable.
- Secret existence, RunPod inventory, capacity, prices, experiment spend or owned-resource cleanup evidence.

No paid resources were created. Public read-only downloads for model configuration and processor files are separate from GPU execution.
