# Validation record — 2026-11-07

Initial target: Qwen3.5-0.8B wafer-map SFT baseline.

## Local checks

- Python 3.12.14; Torch 2.10.0+cu128; Transformers 5.3.0; PEFT 0.18.1; Accelerate 1.12.0.
- 15 unit/integration tests passed on CPU. Includes a real randomly initialized tiny Qwen3.5 with one DeltaNet and one full-attention layer, image tensors, nonzero LoRA gradients, frozen vision, image sensitivity, and adapter save/reload equality.
- Data tests cover whole-lot splits, duplicate-connected lots, conflicting labels, rendering palette/aspect ratio, source digest checks, strict parsing, invalid-output metrics, tampering and paired-comparison identity.
- Real pinned pretrained processor/tokenizer: all eight labels passed exact answer-only masking; image tensors and the explicit non-thinking prefix verified. No pretrained model weights were needed for this check.
- Real Trainer integration passed two optimizer steps with gradient accumulation, gradient checkpointing, finite/nonzero adapter gradients, checkpoint state and adapter export. Uses the official tokenizer/processor with a tiny random model and explicitly synthetic fixtures.

These are compatibility/protocol tests. Random fixtures are not WM-811K, and the tiny randomly initialized model is not the pretrained 0.8B checkpoint. No wafer accuracy is inferred from them.

## Not yet verified

- Trusted WM-811K acquisition and actual per-class lot-isolated counts. MIR Lab returned unavailable/error responses during this session.
- Pretrained 0.8B CUDA forward/backward, memory/time, full training, or quality improvement.
- Docker build (no Docker executable in the local environment), GHCR publication and RunPod controller.
- Secret existence, RunPod inventory, capacity, prices, experiment spend or owned-resource cleanup evidence.

No paid resources were created. Public read-only downloads for model configuration and processor files are separate from GPU execution.
