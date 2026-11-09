# waferVLM TODO — after checkpoint 200

Updated 2026-11-09 (Singapore time). This is the next-stage plan; unchecked items
are not completed experiments or new paid-launch authorization.

## Completed baseline

- [x] Train Qwen3.5-0.8B with language-only LoRA on audited WM-811K
  (790 train / 160 validation / 160 test images).
- [x] Compare the constant-2e-4 continuation with the lower-LR arm from the same
  checkpoint-100. At LR 5e-5, validation selected checkpoint-200.
- [x] Record checkpoint-200 test accuracy **64.375%**, macro-F1 **0.636059**,
  and invalid outputs **0/160**; validation macro-F1 **0.717461**.
- [x] Preserve the full result archive, resumable training state, inference
  adapter and SHA256 manifests in
  [checkpoint-200-v1](https://github.com/jasonjiang8866/waferVLM/releases/tag/checkpoint-200-v1).
- [x] Update README/results/recovery instructions and verify temporary Pod
  deletion. See [the completed-run report](docs/checkpoint-200-release.md).

## P1 — diagnose performance before more training

### Prepare instrumentation and reconcile the run ledger

- [ ] Reconcile run `42502913001`: authorization marker, actual lifecycle bound,
  conservative spend, verified cleanup and previously consumed runtime.
  Preserve the existing duplicate-launch and budget guards.
- [ ] Add explicit phase/timing records for environment setup, model/data load,
  image decoding/processing, tokenization/collation, host-to-device transfer,
  forward, backward, optimizer update, evaluation, checkpoint save and export.
- [ ] Separate startup, warm-up and steady-state measurements. Use CUDA-aware
  timing/profiling where appropriate; report instrumentation overhead and an
  unprofiled comparison rather than treating profiler overhead as training cost.
- [ ] Record CPU utilization, data-loader settings, GPU memory, tensor/model
  devices, actual kernel path and optimizer-step progress beside those timings.
  Existing CUDA/BF16 checks already fail without a suitable GPU: preserve that
  behavior and make failures visible; never silently fall back to CPU in RunPod.

### Run a bounded diagnostic

- [ ] Prepare a diagnostic workload capped at **20 minutes**, using a short,
  fixed training/validation subset and a disposable copy of checkpoint-200.
  Include cold start, artifact retrieval and cleanup in a separate lifecycle cap
  that fits the remaining aggregate allowance. Do not overwrite the release or
  promote diagnostic steps into the continuation checkpoint.
- [ ] Identify whether CPU 100% / low reported GPU utilization comes from setup,
  preprocessing, data transfer, the DeltaNet fallback, evaluation, or monitoring.
  Treat RunPod's utilization gauge as one signal; verify actual CUDA execution.
- [ ] Compare candidate changes one at a time, only after profiling identifies
  a bottleneck. Caching/preprocessing, loader changes or optional fast kernels
  are candidates, not assumed fixes.
- [ ] Before adopting a speed change, check fixed-input outputs/loss, finite
  gradients, frozen parameters and a short save/resume trajectory. Record any
  numerical differences and pin the resulting dependency/image versions.

**Done when:** a checked-in report attributes elapsed time by phase, names the
observed bottleneck, reports steady-state seconds per optimizer step, peak GPU
memory and evaluation time, and recommends a measured change or explicitly
retains the current path. Export evidence and verify owned-Pod cleanup.

## P1 — inspect errors using existing evidence

- [ ] Compare checkpoint-100/150/200 validation confusion matrices, per-class
  precision/recall/F1, prediction counts and invalid outputs.
- [ ] Build a validation error gallery, with particular attention to `loc` and
  `scratch`, checking ambiguous morphology, map resolution and label quality.
  Record hypotheses separately from confirmed annotation problems.
- [ ] Plot actual training loss, validation answer-token loss and validation
  macro-F1 against **global optimizer step**, marking the LR change at step 100.
  Keep the constant-LR control and lower-LR arm distinguishable.
- [ ] Describe already observed test errors only as historical evidence.
  Use validation evidence for new training, prompt or preprocessing decisions;
  do not silently relabel samples or alter the split in response to model errors.

**Done when:** every score and example links to an original prediction/sample ID,
the major confusions are documented, and any proposed intervention has a stated
hypothesis and a separate experiment specification.

## P2 — controlled continuation from 200 to 300

Start only after performance diagnosis, checkpoint recovery verification and a
revised lifecycle estimate. Keep the current 0.8B generative VLM, frozen vision
tower and language-only LoRA as the experimental baseline.

- [ ] Prepare a new continuation configuration using the released complete
  checkpoint-200; remove dependence on the expiring Actions download URL.
  Verify the release digest, dataset/model identity and checkpoint global step.
- [ ] Restore adapter, AdamW moments/step counts, scheduler, RNG and data
  position. Keep **constant LR 5e-5**, seed17, rank8, accumulation8, prompt,
  answer mask, data split and rendering fixed. Do not add LR decay or unfreeze
  vision in this same comparison.
- [ ] Validate checkpoint-200 as the starting reference, then execute **50 + 50
  new optimizer steps** and validate at **250 and 300**. Check every actual LR
  and gradient trace; verify the second resume retains the intended state.
- [ ] Select the highest **validation macro-F1** among 200/250/300; break ties
  in favor of the earlier checkpoint. Record validation accuracy, answer-token
  loss and invalid rate alongside it. If neither new checkpoint improves,
  retain checkpoint-200 and report the negative result.
- [ ] Keep this development run validation-only. Do not repeatedly rerun the
  existing test split to choose steps, learning rates, prompts or kernels.
- [ ] Export full resumable states, predictions, timings and provenance;
  validate hashes, independently verify cleanup, and preserve the run in a
  Release before relying on it for subsequent work.

**Done when:** the run has a complete resume chain and reproducible selection
record, and the report states whether the additional 100 steps improved
validation quality enough to justify their measured cost.

## P3 — reserve an untouched lot-level final evaluation

- [ ] Before another tuning cycle, identify and freeze a new holdout from lots
  never used in any previous training, validation or test subset. Exclude whole
  duplicate-connected lot groups; audit raw-map and rendered-image collisions
  against all development data. Declare the selection rule and actual class
  counts without selecting lots based on model scores.
- [ ] Keep the new holdout out of profiling, error-driven tuning and checkpoint
  selection. Do not treat a new split seed over previously seen images as an
  untouched test set. Record any limits in available lot/metadata coverage.
- [ ] Once the training recipe is fixed, plan paired independent-seed runs with
  the same data protocol and comparable optimizer-step budgets; estimate their
  compute separately. Existing single-seed gains remain exploratory meanwhile.
- [ ] Freeze the final models/comparisons, then evaluate the new holdout once
  per predeclared model/seed. Report accuracy, macro-F1, per-class recall,
  invalid rate, uncertainty across lots/seeds, inference time and peak memory.

**Done when:** a separate final report distinguishes development results from
untouched-lot generalization and states what the data cannot establish.
Normal/unknown detection, SEM images and process root-cause diagnosis remain
outside this eight-label experiment.

## Budget and execution boundaries

User extension on 2026-11-09 (SGT): **+US$10 / +600 minutes**.
The [next diagnostic specification](docs/profile-checkpoint200-plan.md) reserves
US$5.10 / 435 minutes for reconciled history and limits this attempt to 45 minutes
work plus 10 minutes cleanup. The historical measured estimates remain below.

| Item | Recorded state after run 42502913001 |
|---|---:|
| Authorized aggregate spend | US$30 |
| Conservative cumulative estimate, including prior reserves | US$5.051749 |
| Arithmetic spend headroom before new work | US$24.948251 |
| Authorized aggregate runtime | 1,100 minutes |
| Conservative cumulative runtime, including prior reserves | 433.21 minutes |
| Arithmetic runtime headroom before new work | **666.79 minutes** |

The previous 100-step continuation took about 129.55 minutes of workload / 133.21
minutes of Pod lifecycle. The extension provides room for diagnostics and a
controlled continuation; re-estimate from observed profiling results before the
next paid run. Include startup, data, retries, retrieval and cleanup in the same
aggregate ledger. These are conservative estimates, not provider invoices or
new allowances for each dispatch.

- Keep GPU choices and price ceilings within existing authorization; obtain
  fresh capacity/price evidence at launch. Prefer the previously used MIG 24GB
  when available and appropriate.
- Keep the host requirement **CUDA >= 12.8**, accepting higher supported
  versions rather than demanding exact equality.
- Use GitHub Actions and the repository secret `RUNPOD_API_KEY`; pin source,
  worker image and model revision, and preserve independent cleanup.
- A TODO, documentation merge or successful CI run does not launch compute.

Recommended order: **performance + validation diagnostics → revised runtime
plan → controlled 200→300 continuation → untouched-lot/seed confirmation**.
Reserve the new holdout before further tuning, even though its evaluation comes
last. Larger models, vision adaptation and changed data/prompt recipes remain
separate later experiments rather than additions to the immediate continuation.
