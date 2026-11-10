# Fixed-recipe seed and untouched-lot confirmation

Prepared 2026-11-11 SGT. This plan has not been executed and does not launch compute.

The [continuation study](continuation-300-results.md) selected checkpoint 250 by validation macro-F1. Its gain over the re-evaluated checkpoint 200 is comparable to the observed cross-run difference at unchanged step-200 weights. Freeze the comparison now; adding more steps on the same seed would not establish repeatability or lot generalization.

## Training comparison

- Seeds **17, 42, 123**; seed 17 reuses released full checkpoints 200/250. Train seeds 42 and 123 independently from the same pinned Qwen3.5-0.8B revision.
- Keep the audited 790/160/160 split, renderer, fixed prompt, answer-only loss, language-only LoRA rank 8 / alpha 16, frozen vision, effective batch 8 and max length 1024.
- Use constant LR 2e-4 through step 100, then the existing explicit LR transition to constant 5e-5 with complete AdamW/scheduler/RNG/Trainer restoration. Save complete checkpoints at 100, 200 and 250. Do not retune LR, steps, prompts or kernels using these results.
- Primary comparison: **step 250 minus step 200** within each seed. Report all seeds individually, mean and dispersion. Seed 17 was used for development; distinguish its exploratory result from the two new confirmation seeds.
- Validate artifact hashes, actual LR and finite/nonzero gradients, frozen parameters, full resume chains and CUDA/BF16 execution. Keep the original parity failures visible; do not adopt unvalidated acceleration changes.

## Fixed-weight repeatability

For each of the six seed/step endpoints, run three fresh-process greedy validation evaluations on the same GPU model and locked environment where feasible. Keep image order, prompt, generation settings and seed fixed. Report the **first pass** as the headline score, plus all repeat scores, ranges and changed predictions; never choose the best repeat. Record host/GPU, driver, image and model hashes, tokenizer/processor identities, generation time and peak allocated memory. If hardware changes, report that limitation and avoid attributing differences to hardware alone.

## One predeclared final evaluation

Keep the reserved **148 images / 143 duplicate-connected lot groups** untouched until the six trained endpoints and analysis code are fixed. The manifest hash is `f08050ef4fcaca3b9c35ad29905a63957a375af7a024b69e0281968d06f4564f`; it has 20 images per class except near-full (8).

Evaluate the original model plus all six frozen endpoints once each: **seven models** in the single final study. Do not use holdout results to choose a seed, checkpoint, prompt or retry. Publish every prediction and failure, accuracy, macro-F1, class recall, invalid rate, inference timing and memory. The historical test split remains development evidence and is not reused for tuning.

Use 10,000 paired bootstrap resamples of `lot_group`, analysis seed **20261110**, to estimate 95% percentile intervals for paired step-250-minus-step-200 accuracy and macro-F1 within each seed. Each resample includes all selected images in the sampled groups, preserving pairing. Record how resamples missing a class are handled before evaluation; use a fixed eight-class macro-F1 with zero-division set to zero. Report independent-seed dispersion separately; do not treat 480 development predictions or bootstrap replicas as independent training seeds. Small class counts, especially near-full, limit precision.

## Compute envelope and completion

| Workload | Work cap | Cleanup cap |
|---|---:|---:|
| New seed 42 through step 250 | 185 min | 10 min |
| New seed 123 through step 250 | 185 min | 10 min |
| Fixed-weight repeats and predeclared final evaluation | 60 min | 10 min |
| Total additional bound | 430 min | 30 min |

Planned cumulative bound: **951.911 minutes** from the current conservative 491.911 minutes. At the highest approved GPU-only rate (US$1.19/hour) plus the existing 60 GB ephemeral-disk estimate, planned cumulative cost is approximately **US$15.095**, below the US$30 / 1,100-minute total authorization. Include startup, acquisition, inference, retrieval, retries and cleanup in this same ledger. These estimates are not provider-enforced caps.

Retain existing GPU choices and hourly ceilings, CUDA **>=12.8**, pinned source/image/model, no persistent volume, exact owned-Pod cleanup and independent three-observation absence checks. Reconcile the consumed authorization marker, implement bounded seed and evaluation workloads, pass CI, publish/pin images and obtain fresh capacity/price evidence before dispatch. If the evaluation cap is insufficient, stop and revise the remaining aggregate plan before further paid work; do not silently reduce the declared study or relax gates.

Completion requires full result preservation and a final report separating validation selection, repeatability, seed variation and untouched-lot results. This plan does not address normal/unknown defects, SEM images or causal process diagnosis.
