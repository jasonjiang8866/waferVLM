# Checkpoint-200 diagnostics — measured findings and pending recovery

Updated 2026-11-10 SGT. **Evidence recovery complete; both diagnostic numerical
comparisons remain rejected.** Both raw archives, internal checksum manifests,
parent/optimizer identities, actual steps/LRs, comparison tensors and cleanup
records were inspected. The repeat confirms variation between two unprofiled
runs as well as between profiled/unprofiled runs. It does not prove numerical
or statistical equivalence. No diagnostic weights are promoted.

[Permanent diagnostic Release](https://github.com/jasonjiang8866/waferVLM/releases/tag/checkpoint-200-diagnostics-v1) — both original ZIPs, metadata/figures, report and SHA256SUMS. All five uploaded asset sizes/hashes were checked against the local files.

## What is already established

The first run executed on CUDA: every model parameter was on `cuda:0`, peak
training allocation was **2,295,145,472 bytes (2.138 GiB)**, and one profiled
microbatch contained **54,521 CUDA events**. A low RunPod utilization gauge does
not establish CPU fallback. Execution used `torch_chunk_gated_delta_rule` with
`fast_path_available=false`.

[Run 42655151230](https://github.com/jasonjiang8866/waferVLM/actions/runs/42655151230)
used PRO 6000 Blackwell MIG 1g.24gb in US-WA-1, Torch 2.10.0+cu128,
Transformers 5.3.0 and the pinned Qwen3.5-0.8B revision. Both arms restored the
released step-200 adapter, Adam moments/step counts and RNG/data position, then
executed exactly four optimizer updates at LR5e-5, seed17 and accumulation8.

| First-run measurement | Observed result |
|---|---:|
| Unprofiled Trainer loop | 177.1125 seconds for 4 new steps |
| Unprofiled seconds per new optimizer step | **44.2781** |
| Unprofiled whole subprocess | 192.6597 seconds |
| Profiled Trainer loop | 220.0048 seconds for 4 new steps |
| Profiled whole subprocess | 231.5499 seconds |
| Observed whole-process ratio | 1.2019 |
| Profiled peak host RSS | 3.862 GiB |
| CUDA kernel-launch API calls in one capture | 49,768 |
| Largest absolute adapter difference | **1.5239231e-5** |
| Original tensor tolerance | atol1e-6 / rtol1e-5; **failed** |

Trainer's raw resumed `train_steps_per_second` counts historical global steps.
The table uses four **executed** steps divided by this run's training time.
The short serial probes include warm-up and ordering effects; their ratio is
not an isolated profiler-overhead estimate or a production throughput guarantee.

| Profiled scope | Calls | Wall seconds |
|---|---:|---:|
| Model/processor loading | 1 | 6.308 |
| Image verification/decoding | 33 | 0.132 |
| Image preprocessing | 66 | 14.273 |
| Tokenization | 66 | 0.044 |
| Host-to-device transfer | 32 | 0.096 |
| Forward and loss | 32 | 91.414 |
| Backward | 32 | 86.850 |
| Optimizer update | 4 | 0.046 |
| Checkpoint save | 1 | 0.197 |
| Adapter export | 1 | 0.850 |

These synchronized scopes are **nested**; do not add parent and child times.
The kernel capture was microstep9, following one eight-microbatch warm-up step.
The final two steps averaged 47.72 seconds in summed training-microbatch scopes,
excluding collation, optimizer work and saving. This is not full step latency.
Data acquisition took 27.62 seconds; validation analysis/gallery generation 0.164.

Forward/backward dominate these measured scopes. The numerous short kernels,
CPU launch/copy work and fallback implementation support a host-dispatch/fallback
bottleneck hypothesis. They do not isolate thread contention or validate the
dashboard's sampling. PyTorch reported 64 intra-op and 64 inter-op threads and
zero DataLoader workers. Record actual cgroup CPU quota/affinity before testing
thread limits. Fused DeltaNet kernels remain a separately validated experiment;
none were installed or adopted here.

## Why there is a repeat control

Two differing trained adapters do not identify whether instrumentation or
ordinary CUDA/BF16 variation caused the difference. PR #12 therefore added a
second unprofiled arm before the profiled arm, with all three starting from the
same full checkpoint. It retains the original numerical tolerance and rejects
any failed pair. The original step-200 eight-image validation timing probe now
runs before final parity rejection, so that measurement can survive a mismatch.

Strict full determinism was considered and removed before any GPU launch:
the pinned DeltaNet fallback uses floating-point CUDA cumsum, which the
installed PyTorch documentation lists as unsupported in strict deterministic
mode. Changing that operation would change the implementation being measured.
No numerical algorithm, model, dependency or training recipe was changed.

[Repeat run 42661065788](https://github.com/jasonjiang8866/waferVLM/actions/runs/42661065788)
used **RTX 4090 in US-IL-1 at US$0.89/hour**. All three arms restored the same
step-200 adapter/Adam/scheduler/RNG identities and completed steps 201–204 with
four finite, nonzero gradient checks and constant LR5e-5. The actual adapter
comparisons were recomputed independently from the saved safetensors.

| Repeat comparison | Maximum absolute adapter difference | Original allclose gate |
|---|---:|---|
| Unprofiled vs unprofiled repeat | 2.2049207e-5 | failed |
| Unprofiled vs profiled | 3.0776457e-5 | failed |
| Unprofiled repeat vs profiled | 1.2092525e-5 | failed |

The unchanged atol1e-6 /rtol1e-5 gate therefore rejected all three pairs.
Variation exists without instrumentation; this does not identify its exact
CUDA/BF16 mechanism or show that the profiler has no numerical effect.
Strict deterministic mode and relaxed tolerances were not adopted.

| Repeat timing | Unprofiled | Unprofiled repeat | Profiled |
|---|---:|---:|---:|
| Trainer loop, four new steps (s) | 51.2614 | 51.9300 | 104.6189 |
| Seconds per new optimizer step | 12.8154 | 12.9825 | 26.1547 |
| Whole subprocess (s) | 76.3318 | 64.5925 | 117.5528 |

The profiled scopes recorded forward/loss 22.100s, backward 36.434s,
image preprocessing 0.601s and host-to-device 0.0039s. One microbatch contained
55,259 CUDA events and 50,396 kernel launches. The final two training-microbatch
scope sums were 16.264s and 13.992s, excluding collation, optimizer and saving.
Peak training allocation remained 2.138 GiB; host peak RSS was 3.550 GiB.
The original checkpoint-200 timing probe completed on eight validation images:
model loading 4.689s, generation total 5.001s, first image 1.106s, subsequent
seven-image mean 0.556s, peak allocation 1.692 GiB. This is a small latency probe,
not a new quality evaluation or a representative tail-latency benchmark.

The 4090 run was faster than the previous MIG run, but host allocation, hardware
and process timing changed together; **this is not an isolated GPU speedup**.
Both used the same fallback path. The 4090 reported 64 intra-op /128 inter-op
threads; CPU quota/affinity were not captured, so CPU saturation is not fully
attributed. Keep the original unprofiled training implementation for continuation.
Prefer the already-authorized 4090 when available; assess any thread-limit,
preprocessing or fused-kernel change in a separate controlled experiment.

PR #14 fixed progress retaining the previous arm's 204/204. Its 37 lifecycle/
integrity tests, model compatibility checks and image build passed before merge.
That telemetry fix was not part of either measured GPU run and requires a new
published/pinned worker image for future runs.

## Validation quality evidence

These metrics were recomputed from the same 160 released validation predictions,
checking identical sample IDs, labels and image hashes. They are **validation**,
not new test-set scores.

| Validation metric | Step 100 | Step 150 | Step 200 |
|---|---:|---:|---:|
| Accuracy | 64.375% | 69.375% | 72.500% |
| Macro-F1 | 0.640676 | 0.686280 | 0.717461 |
| Answer-token loss | 0.270583 | 0.220790 | 0.211074 |
| Invalid output rate | 0% | 0% | 0% |

Mean training loss fell from 0.253191 over steps 101–150 to 0.205033 over 151–200.
The lower validation loss and rising F1 support a controlled continuation, but
do not guarantee further gains. Each validation class has only 20 examples.
Step 200 recall is 100% for near-full, 35% for loc and 50% for scratch; center recall
fell from 90% at step 150 to 65% at step 200 even as aggregate F1 increased.

Six of 20 scratch examples were predicted loc. The reviewed gallery includes
`27eadb2b864e3ee3edfdda41` and `5f0dc094af5b30aae2baddf1`, with visible thin
diagonal/vertical traces but loc predictions. These are morphology-confusion
observations, not proven label errors. No labels, images, prompt or split changed.

The first artifact contains `validation-analysis/` with full confusion counts,
per-class metrics, sample IDs, learning-curve source data/SVG and an HTML error
gallery (up to four errors per class, sorted by ID). These outputs can also be
regenerated with the committed analysis script and the immutable parent Release.
The reviewed learning-curve and loc/scratch error figures are included in the
diagnostic evidence bundle. These figures use existing validation evidence. The official model remains
[checkpoint-200-v1](https://github.com/jasonjiang8866/waferVLM/releases/tag/checkpoint-200-v1)
with 64.375% test accuracy and 0.636059 test macro-F1; no new test evaluation occurred.

## Provenance, artifacts and cleanup

| Item | First attempt | Repeat control |
|---|---|---|
| Controller SHA | `09e32cd7b6fcdfa5760f8c896de3d1042269bdf5` | `731755fccd2ba8fea63bdb16c657d791f062d732` |
| Image source SHA | `4e0d4c7d59ae4a3b7e07553c60b36c0e640d39f5` | `1243e899dca62f607b770cac27818fe7655a399e` |
| Worker digest | `sha256:82c5e3a06de7f38d36a67f12d56cf2e21b1a3c615c0bf912ef42764b4fc62ab5` | `sha256:b8b9f2998c571b13f471c229ac2451f31a9dd977bd25367da48a9b841df3dd93` |
| Result artifact ID | 13693196380 | 13693684721 |
| ZIP bytes | 168312935 | 252441790 |
| ZIP SHA256 | `2bc71c0df5bab219a2fded799f7308bf8e804f0ab0b734807bc661aac0c1affb` | `3aa535a29c9647f547bfe3ecca5ec1a592bf032655be2170c58fd0ab9c834a32` |
| Independent cleanup | [42657156272](https://github.com/jasonjiang8866/waferVLM/actions/runs/42657156272), success | [42662666105](https://github.com/jasonjiang8866/waferVLM/actions/runs/42662666105), success |

First inner tar SHA256:
`a7b055b7921f270639177ae4d8a67c325f8e6f9151860e54c75c74ff0641f7f2`.
Its 58 regular files and 84 checksum-manifest entries were checked. This establishes
artifact integrity, not numerical acceptance. Pod `pesnvzd6sxdu40` was absent
in three consecutive controller checks and again in the independent cleanup.

Repeat inner tar SHA256:
`a7c5faa09b12a4792f511281fbbdc57755193e0e499e2bc8cc3bb3c43825483d`.
Its **77 regular files /119 manifest entries** were independently verified.
Pod `54pv0bke2e16ev` was absent in three consecutive controller checks and again
in the independent cleanup. Cleanup artifact 13693809665, SHA256
`6242e251a9cd697b5cbd05d2ae09e22f1d4a72ead05feae462c2bfee5fac2fc7`,
was downloaded, hashed and read; it records verified_absent with three checks.

The original Actions artifacts expire 2026-12-10 00:40:20 SGT (first) and
01:14:13 SGT (repeat). The diagnostic Release preserves separate copies of both
raw archives, a metadata/figure bundle and SHA256SUMS. Raw diagnostic archives
contain disposable step-204 weights solely as rejected-experiment evidence;
they are not deployable model releases. The official model remains checkpoint-200.

## Cost and continuation decision

The user extension adds US$10 /600 minutes, yielding **US$30 /1,100 minutes**
aggregate. The first diagnostic added an estimated US$0.134111 and 11.513 minutes;
its conservative cumulative ledger was US$5.234111 /446.513 minutes.
The repeat started with rounded-up history reserves US$5.24 /447 minutes.
Its intent-to-verified-absence interval was **552.109 seconds (9.202 minutes)**,
adding estimated GPU+ephemeral-disk cost **US$0.137863**. The resulting
conservative total is **US$5.377863 /456.201817 minutes**, not a provider invoice.
No persistent volumes were created. Reserve **US$5.38 /457 minutes** before
any next attempt; marker 13694661007 belongs to the reconciled repeat run.

Proceed with a separate validation-only 200→250→300 continuation from the
unchanged released full step-200 checkpoint. Retain LR5e-5, seed17, rank8,
accumulation8, frozen vision and the existing fallback implementation. The
failed profiler-equivalence experiment remains failed; no optimized kernel or
instrumented trajectory is being adopted. Select by validation macro-F1, ties
to the earlier checkpoint, retaining 200 if neither later candidate improves.
Do not evaluate the existing test split again during this development cycle.

Freeze any available unseen, duplicate-connected lot groups before this run;
report limited class coverage rather than reusing development lots. Re-estimate
and bound the complete lifecycle using the measured timings, with startup,
validation, retrieval and cleanup included. A next-stage configuration/CI merge
alone is not evidence that paid training has launched or improved quality.
