# Seed confirmation implementation and review handoff

Prepared 2026-11-11 SGT against main `4bb2e8415002336759e54c6c28aacc0ee5bad4a2`.
This implements the existing [seed-confirmation-plan](seed-confirmation-plan.md);
it does not change that protocol. Status: **code prepared locally; no new GPU run,
external branch, PR, image publication or holdout evaluation**.

## Current evidence

- Live GitHub: PRs 17 and 18 are merged; no open PRs at inspection. Main CPU CI
  run 42912636332 passed. The latest GPU run remains 42682266526.
- Both released full-state ZIPs were downloaded again, checked against their
  pinned outer/inner SHA256 values and inspected. All 18 files in each selected
  step directory match its original manifest. Restricted CPU tensor loading
  verified 372 AdamW states and 372 finite language-LoRA tensors / 5,411,328
  parameters per endpoint; Adam/scheduler positions are 200/250, LR is 5e-5.
- Current conservative cumulative usage is **US$5.903039 / 491.911 minutes**;
  remaining authorization is **US$24.096961 / 608.089 minutes**. This is the
  recorded lifecycle estimate, not a supplier invoice or fresh live billing.
- Run 42682266526 authorization artifact **13703281873**, SHA256
  `b6c494dc8ca303e275c49a6a3956ecc650fd530e9852dcc6a9665a91bb44d53d`, is still
  consumed. This patch does not add it to `RECONCILED_MARKERS`.
- Historical primary and independent cleanup verified the old Pod absent.
  Current account inventory, capacity and live quote have **not** been queried.

## Implemented workloads

`configs/seed-confirmation.json` fixes the model recipe, dataset identity,
training seeds 42/123, split seed 17, endpoints 200/250 and inference seed 17.
The existing `train` implementation restores adapter, AdamW, scheduler, RNG and
Trainer data position. The new runner uses separate training processes for
0→100→150→200→250, retaining the historical step-150 restart boundary as well as
all required 100/200/250 checkpoints. Constant LR changes from 2e-4 to 5e-5 only
at the full step-100 restore. No optimizer reset, prompt, kernel or split change.

A recovery may start from a complete hash-checked step 50/100/150/200 state of
that same seed. It copies the parent into the new result package and refuses to
overwrite the original. Failure/timeout preserves available checkpoints and a
failed status. A newly sealed partial directory is evidence, not an accepted
training result; incomplete checkpoint members are rejected on restoration.
Recovery time counts against the same aggregate budget; the local CLI does not
create a new authorization or implement a global spending cap.

The six full checkpoints must be hash-bound before any repeat starts. Each
endpoint gets exactly three fresh-process greedy validation passes. The first
pass is the headline, even when a later pass scores better. All raw responses,
changed labels/responses, complete eight-class metrics (zero division = zero),
per-pass timings/memory, F1 range and paired 250−200 deltas are preserved.
Seed 17 is labelled exploratory; statistics for seeds 42/123 are also reported
separately. Standard deviations describe seed dispersion, not confidence in
independent-lot generalization.

The evaluation records source/image identity, base-weight and adapter hashes,
processor/tokenizer/template hashes, actual GenerationConfig, seed, process
identity, host, GPU, driver and package versions. Model/processor/software drift
is rejected. Hardware/host changes are reported as a limitation, not assigned a
causal effect. Analysis rechecks every artifact hash and sample order, parses
raw responses again and recomputes metrics rather than trusting metrics.json.

The final 148-image holdout has no inference entrypoint in this runner. The
existing development dataset audit still checks train/validation/test image
integrity; only validation is passed to evaluation. No test score is produced.

## Exact container/local entrypoints

Use the pinned environment (`uv run --frozen --extra train` locally; `python`
inside the confirmation image). Prepare data once with the unchanged
`scripts/prepare_public_data.py`; never replace its split seed with training seeds.
Paths below refer to retained inputs and separate, new output directories.

```bash
python scripts/run_seed_confirmation.py train \
  --dataset /workspace/wafer-data --seed 42 --output /workspace/seed42 \
  --timeout-seconds 11100
python scripts/run_seed_confirmation.py train \
  --dataset /workspace/wafer-data --seed 123 --output /workspace/seed123 \
  --timeout-seconds 11100

# Example recovery, ONLY after accounting for the previous attempt and cleanup:
python scripts/run_seed_confirmation.py train \
  --dataset /workspace/wafer-data --seed 42 --output /workspace/seed42-recovery \
  --resume-from /workspace/seed42/step100/checkpoints/checkpoint-100 \
  --timeout-seconds 11100

# Reconstruct the released seed-17 full states, with all published hash checks:
python scripts/prepare_confirmation_inputs.py --download --output /workspace/released

python scripts/run_seed_confirmation.py freeze \
  --dataset /workspace/wafer-data --output /workspace/endpoints.json \
  --seed17-step200 /workspace/released/seed17-step200/checkpoints/checkpoint-200 \
  --seed17-step250 /workspace/released/seed17-step250/checkpoints/checkpoint-250 \
  --seed42-step200 /workspace/seed42/step200/checkpoints/checkpoint-200 \
  --seed42-step250 /workspace/seed42/step250/checkpoints/checkpoint-250 \
  --seed123-step200 /workspace/seed123/step200/checkpoints/checkpoint-200 \
  --seed123-step250 /workspace/seed123/step250/checkpoints/checkpoint-250

python scripts/run_seed_confirmation.py repeat \
  --dataset /workspace/wafer-data --endpoints /workspace/endpoints.json \
  --output /workspace/repeats --timeout-seconds 3600

# CPU-only independent analysis after retrieving the complete package:
python scripts/run_seed_confirmation.py analyze \
  --results /workspace/repeats --output /workspace/recomputed-summary.json
python -m runpod.confirmation_budget
```

## Runtime and price envelope

The original caps remain unchanged; **these are bounds, not measured durations**.
Startup, acquisition, inference, retrieval and any retry share the same envelope.
The supervisor must give the runner only the remaining phase time after startup
and acquisition, reserving retrieval and cleanup. Local runner flags are maximum
work limits, not permission to exceed the aggregate lifecycle allocation.

| Planned phase | Work bound | Cleanup reserve | Lifecycle bound |
|---|---:|---:|---:|
| Seed 42, full fresh training | 185 min | 10 min | 195 min |
| Seed 123, full fresh training | 185 min | 10 min | 195 min |
| Repeatability / later predeclared final-study allowance | 60 min | 10 min | 70 min |
| Total original additional envelope | 430 min | 30 min | **460 min** |

This patch runs **validation repeats only** in the evaluation phase. The final
study in the original protocol remains deferred, with no claim that its runtime
has been measured or that it fits the unused portion of those 60 minutes. If
repeats exhaust that allowance, stop and review the remaining aggregate plan;
do not omit passes, silently add time, or access holdout.

Using 60 GB ephemeral disk at the existing estimate of $0.10/GB/month (28-day
month), no persistent volume, and the full 460-minute bound:

| Assumed GPU-only rate | Added compute + disk | Cumulative estimate | Remaining after plan |
|---|---:|---:|---:|
| RTX 4090 ceiling $0.89/h | $6.891786 | $12.794825 | $17.205175 |
| Highest existing ceiling $1.19/h | $9.191786 | $15.094825 | $14.905175 |

Planned time: **951.911 / 1,100 minutes**; unallocated time: **148.089 minutes**.
The quoted rates are authorized ceilings/historical assumptions, **not fresh
supplier offers**. Keep RTX 4090 first; existing MIG/3090/5090 choices and their
individual ceilings remain unchanged; CUDA capability must be **>=12.8**.
No new monthly-budget or current account-balance check has been performed.

## Image, transport and launch boundary

`runpod/Dockerfile.confirmation` inherits the exact existing worker environment:

`ghcr.io/jasonjiang8866/wafervlm@sha256:b13a9807ae05698f0813e0d1ab7f25548e782f65729d2b2aec5e66de6de24cd1`

It overlays only project source/config/tests and runs the new offline tests.
It neither re-resolves dependencies nor includes a base model, raw dataset or
account secret. The new `.github/workflows/confirmation-image.yml` builds on PRs;
publication requires an explicit manual main dispatch with `publish=true`.
The resulting new immutable digest is **pending**. No Docker daemon was available
in the local review environment, so the container build itself is not claimed
as verified. Do not use the old digest as if it contained this implementation.

The existing paid `gpu.yml`, worker and controller still target the completed
extension study; they are intentionally **not activated for confirmation** in
this implementation PR. The new CLI/container entrypoints are executable, but
are not yet a dispatch-ready RunPod lifecycle integration. After code approval
and image publication, a separately reviewable activation change must route
seed42/seed123/repeats through the existing bounded controller, pin that exact
source/digest, add independent archive acceptance and reconcile each consumed
marker sequentially. Do not dispatch the old workflow to run these new commands.
Fresh capacity/price/owned-Pod checks and explicit user approval are required
before the first paid action; the user's current request authorizes preparation.

For each eventual Pod, retain the complete seed output (100/150/200/250 full
states, recovery parent when present, logs, run/training/config/status files and
checksums). Retrieve and verify an experiment.tar.gz plus controller allocation,
acceptance, ledger and primary/independent cleanup evidence before ending the
attempt. Use 30-day Actions artifacts for transport; propose a versioned
`seed-confirmation-v1` prerelease for durable full states and all 18 prediction
sets after review. No Release is created by this patch. Re-stage verified seed
42/123 outputs plus the pinned seed-17 releases for the repeat Pod, regenerate
the path-only endpoint matrix at the new location and recheck all weight hashes.
No persistent volume is needed. Preserve raw failed attempts as well as successes.

## Review request

Proposed branch: `feat/seed-confirmation-validation`.
Proposed PR title: **Implement frozen independent-seed training and validation repeatability**.
Base: current `main`, rechecked before eventual submission.
Scope: new runners/configuration, full-state hash checks, provenance and independent
analysis, offline CPU tests, image build definition, budget estimate and this handoff.

Approval requested now is **create this code PR only**. It does not include merge,
GHCR publication, Release publication, lifecycle activation or paid GPU dispatch.
The later GPU request must show the actual reviewed source/digest, workflow inputs,
fresh quote, remaining ledger, result route and exact cleanup scope.

## Local validation evidence

- 47 project tests passed, including both new seeds with real tiny Qwen3.5,
  Trainer, AdamW and scheduler/RNG restoration; no model/data downloads are
  needed for this suite.
- 41 RunPod contract/lifecycle/budget tests passed. These use mocks and do not
  demonstrate current provider availability or actual owned-Pod absence.
- `scripts/check_training.py` passed with the pinned official tokenizer and
  processor, tiny random model weights, answer-mask checks, uninterrupted vs
  restored training comparison, LR intervention/save/second restore and
  diagnostic instrumentation parity. This compatibility check downloaded
  processor/config files only; no base-model training or GPU execution occurred.
- Both published full checkpoints passed the new staging command with their
  original manifests, plus restricted CPU inspection of saved tensor state.
- Workflow YAML parsing and `git diff --check` passed.
- Local environment: Python 3.12.14, Torch 2.10.0+cpu / torchvision 0.25.0+cpu,
  Transformers 5.3.0, PEFT 0.18.1 and Accelerate 1.12.0. Other dependencies came
  from the unchanged lockfile. The CPU Torch variant is an offline test choice;
  it does not replace or certify the pinned CUDA image. No lockfile was changed.
