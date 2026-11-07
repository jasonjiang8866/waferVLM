# Initial experiment protocol

## Question and fixed configuration

Does language-side LoRA improve Qwen3.5-0.8B wafer-map classification over the same model without adaptation?

| Setting | Initial value |
|---|---|
| Model | Qwen/Qwen3.5-0.8B |
| Revision | 2fc06364715b967f1860aea9cf38778875588b17 |
| Labels | Eight WM-811K defect labels; no `none` |
| Input | 256×256 image and the fixed prompt in `core.py` |
| Precision | BF16 CUDA; FP32 only for CPU debugging |
| Training | Language linear layers only, LoRA rank 8, alpha 16, dropout 0 |
| Frozen | Vision encoder, vision merger, base model weights, token embeddings and LM head |
| Optimizer | AdamW, lr 2e-4, weight decay 0, clip norm 1 |
| Training budget | 100 optimizer steps, microbatch 1, accumulation 8 |
| Seed | 17 |
| Generation | Greedy, thinking disabled, at most 16 new tokens |
| Primary metric | Macro-F1 across all eight classes, invalid output counted as incorrect |

These are initial experimental choices, not tuned recommendations. Language projections include both standard attention and Gated DeltaNet projections, plus MLP projections. Using only `q_proj`/`v_proj` would miss most hybrid-attention layers. Exact adapted module names are recorded.

## Data identity and isolation

1. Verify the trusted pickle's SHA256 before deserialization.
2. Preserve source row index and lot identity. Reject missing lots and malformed maps.
3. Reject conflicting labels on identical maps; deduplicate exact maps.
4. Join lots linked by a duplicate map, then hash the resulting group with the seed into 70% train / 15% validation / 15% test buckets.
5. Sample at most 100/20/20 images per class from the already isolated splits. Never duplicate rare examples to fill quotas. Actual sample counts are authoritative.
6. Render with nearest-neighbor interpolation, preserve aspect ratio, and add black padding. No filenames containing labels, plot axes, legends or annotation text enter the model.
7. Check IDs, lots, raw-map hashes and rendered-image hashes across splits; verify all file checksums before every run.

This checks exact duplicates, not arbitrary rotated/near-duplicate morphology or hidden manufacturing relations across lots. Cross-fab, product, time and inspection-point-map generalization remain separate studies.

The official WM-811K `trianTestLabel` field (including its spelling) is not used: this protocol uses explicitly declared lot-group isolation. Missing classes are a preparation failure. Changing the split seed must be declared before any model evaluation; do not search for a seed based on accuracy.

## Loss and evaluation

The full training token sequence must start with exactly the inference prompt sequence, including image tokens and the non-thinking assistant prefix. A prefix mismatch raises an error. Prompt tokens are masked with -100; only the label and assistant terminator are supervised. Examples exceeding the sequence limit fail instead of silently losing labels or image tokens.

The initial dataset contains one direct classification question per image, not four correlated QA variants. NVIDIA's question expansion is optional future work, and must happen after splitting by original wafer.

Smoke testing uses validation only. The full experiment fixes hyperparameters before test evaluation. If further tuning is needed, use validation; test is for the final report. Paired comparisons require the same model revision, dataset identity, prompt, decoding settings and exact ordered image identities. No accuracy threshold is invented before measuring the baseline. Report failures and degraded results too.

## Acceptance gates

- Technical: finite loss/gradients, at least one nonzero adapter-gradient step, frozen vision, exact answer masking, adapter export/reload, complete checksummed artifacts.
- Data: all eight classes, no detected cross-split leakage, known source hash and actual sample counts.
- Scientific: report macro-F1, accuracy, per-class recall, invalid rate, generation latency and memory. One seed is exploratory; confirm promising gains across seeds before making stronger claims.
- Operations: obtain per-experiment budget, pin the built image digest, collect outputs before deleting the temporary Pod, and independently verify deletion. Worker success alone cannot establish this gate.

## Next stages

After a real-data smoke: obtain a paired baseline, then consider additional seeds, a 2B model, a small image-classifier baseline, normal/unknown classes, and inspection-coordinate transfer. Vision adaptation is a separate ablation; do not silently unfreeze it in this baseline.
