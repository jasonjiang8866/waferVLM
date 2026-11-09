# Checkpoint-200 diagnostics — measured findings and pending recovery

Updated 2026-11-10 SGT. **Status: partial.** The first diagnostic was retrieved
and independently inspected. Its numerical-parity gate failed. A three-arm
repeat-control run also failed final acceptance and uploaded its result archive,
but its internal comparison, inference timings and final cost have **not yet
been inspected**: the execution workspace and browser became unavailable during
artifact retrieval. Both originating runs' independent cleanup workflows passed.
Do not describe the repeat as accepted or assume its failure has the same cause.

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
has uploaded evidence but remains **uninspected beyond Actions metadata/logs**.
Do not infer its internal parity result, probe completion, GPU allocation or
actual cost until its archive is read. The displayed progress also retained the
maximum from earlier arms; the accompanying fix resets progress to the current
arm and labels each group. That fix was locally tested with 37 lifecycle/integrity
tests before the workspace disconnected; it was not part of either GPU run.

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
The polished local figures and combined diagnostic Release were not uploaded
before the workspace outage. The official model remains
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

For the repeat, the cleanup job explicitly used ORIGIN_RUN_ID 42661065788 and its
owner-checked deletion/inventory step completed successfully. Its 297-byte cleanup
artifact 13693809665 has SHA256
`6242e251a9cd697b5cbd05d2ae09e22f1d4a72ead05feae462c2bfee5fac2fc7`.
Its JSON and result archive still need direct inspection; no Pod ID is guessed.

The first result expires **2026-12-10 00:40:20 SGT**; the repeat result expires
**2026-12-10 01:14:13 SGT**. They have not yet been copied to a diagnostic Release.
The existing checkpoint-200 model Release is already preserved independently.

## Cost and exact remaining work

The user extension adds US$10 / 600 minutes, yielding **US$30 /1,100 minutes**
aggregate. First diagnostic intent-to-absence time was 690.767 seconds
(11.513 minutes), with estimated incremental GPU+ephemeral-disk cost
**US$0.134111**. The conservative cumulative ledger then stood at
US$5.234111 /446.513 minutes, including prior reserves. No persistent volume
was created.

The repeat reserved US$5.24 /447 minutes for history and retained the 20-minute
diagnostic and 45+10-minute lifecycle limits. Its highest-rate planned cumulative
bound was below US$6.35 /502 minutes. This is a conservative plan, not an invoice;
the repeat's actual cost and resource details remain uninspected.

1. Download repeat artifact 13693684721; verify its ZIP hash, inner result hash,
   all member manifests, parent checkpoint/optimizer identity and every arm's
   actual steps, LR trace, gradients and CUDA metadata.
2. Read all pairwise adapter comparisons and the eight-image inference probe;
   distinguish baseline variation from instrumented-run variation without
   asserting statistical equivalence from one repeat.
3. Read cleanup artifact 13693809665, reconcile marker 13694661007 (SHA256
   `6a9f7d955eb2d5f7547b1e8125281b4bef7a9b07b55a883199cd4257aee5352b`)
   and the actual conservative cost/runtime. Do not unlock another paid run
   from guessed totals.
4. Publish the diagnostic evidence/figures in a Release, clearly excluding
   disposable diagnostic weights from model promotion, and finish this report.
5. Only then finalize the 200→250→300 validation-only continuation estimate.
   Keep LR5e-5 and the original step-200 parent; reserve an untouched lot-level
   holdout before further tuning and keep seed confirmation separate.

No 200→300 quality-training run was started in this phase.
