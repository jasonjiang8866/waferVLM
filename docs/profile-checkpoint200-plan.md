# Checkpoint-200 performance diagnostic

The user added **US$10 and 600 minutes** on 2026-11-09 (SGT), raising the same
experiment's aggregate ceilings from US$20 / 500 minutes to **US$30 / 1,100
minutes**. Original GPU choices and hourly caps remain unchanged. The existing
authorization ID is retained for ownership and ledger continuity; its historical
`usd20` spelling is not the current spending ceiling.

Run `42502913001` is reconciled with authorization artifact `13626700093`
(SHA256 `2f13170ad3fff0f8546ee987db5cd63818ddf1cebc90b9061b1d996e5d3e7b2f`).
The preserved result ZIP and independent cleanup run `42514159527` confirm
completion and Pod absence. Conservatively reserve **US$5.10 / 435 minutes**
for the four previous attempts, rounding up their US$5.051749 / 433.211-minute
cumulative estimate. Ledger checks still reject any additional unreconciled run.

## This launch

- One temporary GPU Pod; existing MIG24GB / 4090 / 3090 / 5090 caps and
  **CUDA >=12.8** selection remain unchanged. Fresh price, capacity, existing-Pod
  inventory and account headroom checks run before allocation.
- Diagnostic workload limit **20 minutes**. The whole paid attempt has **45
  minutes** for startup, data, diagnostics and result retrieval, plus **10
  minutes** cleanup. At the highest allowed GPU rate, the conservative planned
  cumulative estimate is below **US$6.21 / 490 minutes** including ephemeral disk.
  This is not a provider-enforced billing limit.
- Download the already published `checkpoint-200-v1` result ZIP during image
  build, verify outer/inner SHA256 and every staged checkpoint/evidence file.
  It does not depend on the expiring Actions artifact. No account key enters
  the image or worker.
- Start two disposable arms from the same complete step-200 state: **four
  optimizer steps each**, effective batch8, seed17, LR5e-5, frozen vision and r8.
  The first arm is unprofiled; the second adds synchronized phase timings and
  one-microbatch CPU/CUDA operator capture after one optimizer-step warm-up.
- Compare final adapter tensors within atol1e-6/rtol1e-5. Record instrumentation
  overhead, model load order, CPU process time, actual parameter devices,
  DeltaNet function selection, GPU peak memory and actual CUDA events.
- Time generation on eight preselected validation images (one per class) using
  the original released step-200 adapter. These timings are not a quality score.
- Recompute all existing 100/150/200 validation metrics; export confusion tables,
  per-class metrics, learning curves and a deterministically selected error gallery.
  No test evaluation or checkpoint selection is performed in this diagnostic.
- Discard disposable diagnostic weights after trajectory comparison; preserve
  their hashes and complete diagnostic logs/metadata. Checkpoint-200 remains
  the released model. Future 200→300 training must restore that original state.

## Acceptance and follow-through

The independent archive verifier requires source/image/data/model identity,
the correct step-200 adapter and optimizer hashes, restored Adam step 200,
exactly four nonzero-gradient steps and the constant LR trace in each arm,
finite timings, actual CUDA events, parameter placement, eight-image validation
timing coverage, and the 160-image historical validation analysis.

Only after artifact verification and independent cleanup can the diagnostic be
called complete. Interpret nested timing scopes without adding parent and child
times; separate warm-up, profiler capture, normal microbatches and cold loading.
The serial instrumentation run is not itself an optimized throughput result.

Use measured bottlenecks to decide whether to retain the current runtime or
prepare a separately validated speed change. Then size the planned 200→250→300
validation-only continuation within the remaining US$30 / 1,100-minute allowance.
No vision unfreezing, LR decay or larger model is bundled into this comparison.

Published worker: `ghcr.io/jasonjiang8866/wafervlm@sha256:82c5e3a06de7f38d36a67f12d56cf2e21b1a3c615c0bf912ef42764b4fc62ab5`,
built from source `4e0d4c7d59ae4a3b7e07553c60b36c0e640d39f5` by
[image run 42652571130](https://github.com/jasonjiang8866/waferVLM/actions/runs/42652571130).
The launch provenance check also requires the reviewed current main SHA and
rejects any subsequent workload change not represented by this image.


## Repeatability-control diagnostic after run 42655151230

The first diagnostic is **rejected**, not an accepted performance comparison.
Both arms restored Adam step 200, executed steps 201–204 with nonzero finite
gradients, and stayed at LR5e-5. However, their adapter maximum absolute
difference was 1.5239231e-5 and the predeclared atol1e-6/rtol1e-5 comparison failed.
The two runs alone cannot distinguish instrumentation effects from ordinary
CUDA/BF16 nondeterminism. The original checkpoint-200 release is unchanged.

The preserved trace establishes CUDA execution (all model parameters on cuda:0;
54,521 CUDA events in the captured microbatch) and the torch_chunk_gated_delta_rule
fallback. The unprofiled training loop took 177.1 seconds for four optimizer
steps (44.28 s/step); its whole subprocess took 192.660 seconds. This short
measurement is exploratory, and profiler timings are nested and instrumented.
The eight-image inference probe was not reached in that rejected run.

A bounded retry adds a second **unprofiled** arm from the same complete parent,
then runs the profiled arm: three independent four-step processes in the same
20-minute workload /45+10-minute lifecycle bounds. Record all three pairwise
adapter comparisons at the original atol1e-6/rtol1e-5. No tolerance is relaxed.
The original checkpoint-200 eight-image validation timing probe runs before the
final comparison gate, so useful inference measurements survive parity failure.
A failed comparison remains a rejected experiment; do not promote those weights
or call it a verified speed change. A single repeat measures observed baseline
variation; it cannot establish statistical equivalence or prove its cause.

Strict full determinism was considered but is not used: the pinned Transformers
5.3.0 `torch_chunk_gated_delta_rule` calls floating-point `g.cumsum`, and the
installed PyTorch2.10.0 `use_deterministic_algorithms` documentation lists CUDA
floating-point cumsum among operations that raise in strict mode. Avoiding that
operation would change the algorithm being diagnosed. This retry preserves the
original standard CUDA/BF16 execution mode and records its repeatability limits.

Run 42655151230 authorization marker 13690724238 has SHA256
`a5732ce500f12391e281bffae6a2b1480c81862b2287b3003c49364ca9b76188`.
Its result ZIP (artifact 13693196380) has SHA256
`2bc71c0df5bab219a2fded799f7308bf8e804f0ab0b734807bc661aac0c1affb`.
Pod `pesnvzd6sxdu40` was confirmed absent in three consecutive checks;
[independent cleanup 42657156272](https://github.com/jasonjiang8866/waferVLM/actions/runs/42657156272)
also succeeded. Its intent-to-absence bound was 690.767 seconds and incremental
compute plus disk estimate US$0.134111 (not an invoice). Reconcile that attempt
before retry; round cumulative reserves up to **US$5.24 / 447 minutes**.
The unchanged 45+10-minute launch cap yields a maximum planned cumulative estimate
below **US$6.35 / 502 minutes** at the highest allowed hourly price, within the
existing US$30 /1,100-minute authorization. A newly published and pinned worker
image is required before this changed workload can launch.

Repeat-control worker published by [run 42659522956](https://github.com/jasonjiang8866/waferVLM/actions/runs/42659522956):
`ghcr.io/jasonjiang8866/wafervlm@sha256:b8b9f2998c571b13f471c229ac2451f31a9dd977bd25367da48a9b841df3dd93`,
source `1243e899dca62f607b770cac27818fe7655a399e`. This pin supersedes the first-attempt image above.
