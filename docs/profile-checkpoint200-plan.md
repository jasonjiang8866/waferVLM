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
the correct step-200 adapter and optimizer hashes, restored Adam step200,
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
