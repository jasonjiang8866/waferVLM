# Completed frozen validation repeats

Verified 2026-11-18 (Singapore time). The study is **18/18 complete**, with
2,880 prediction rows from six frozen endpoints evaluated three times each.
[Run 44124342191](https://github.com/jasonjiang8866/waferVLM/actions/runs/44124342191)
completed the last two processes, preserving all 16 previously completed pass
trees byte-for-byte. Independent local verification reproduced the controller's
acceptance receipt, every score and the saved summary.

**Execution and integrity are accepted. Model quality/generalization are not.**
The reserved 148-image holdout remains untouched. The first pass is the headline,
as declared before this study; later repeats are never used to select a score.

## Results

All scores below are validation macro-F1, except the labeled accuracy column.
Seed 17 is exploratory; seeds 42 and 123 are the new confirmation seeds.

| Seed | Step | First-pass accuracy | Repeat 1 | Repeat 2 | Repeat 3 | Changed labels vs repeat 1 (R2 / R3) |
|---|---:|---:|---:|---:|---:|---:|
| 17 | 200 | 75.000% | 0.744294 | 0.744294 | 0.744294 | 0 / 0 |
| 17 | 250 | 78.125% | 0.773992 | 0.773992 | 0.773992 | 0 / 0 |
| 42 | 200 | 71.250% | 0.708102 | 0.708102 | 0.708102 | 0 / 0 |
| 42 | 250 | 63.750% | 0.622862 | 0.622862 | 0.622862 | 0 / 0 |
| 123 | 200 | 80.000% | 0.798185 | 0.798185 | 0.785522 | 0 / 6 |
| 123 | 250 | 75.000% | 0.744042 | 0.757192 | 0.757192 | 9 / 9 |

| Seed | Role | Step 250 minus 200 accuracy | Step 250 minus 200 macro-F1 |
|---|---|---:|---:|
| 17 | Exploratory | +3.125 percentage points | +2.970 percentage points |
| 42 | Confirmation | -7.500 percentage points | -8.524 percentage points |
| 123 | Confirmation | -5.000 percentage points | -5.414 percentage points |

Across the two confirmation seeds, mean first-pass macro-F1 change is **-0.069691**
(sample SD 0.021989); mean accuracy change is -0.062500 (sample SD 0.017678).
Thus the exploratory seed-17 improvement from 200 to 250 did **not** replicate
in either confirmation seed on this validation set. Two confirmation seeds and
a reused development split do not establish generalization or statistical certainty.

## Repeatability and hardware limits

The original 14 passes used RTX 4090. Two continuation passes used RTX PRO 6000
Blackwell Server Edition 1g.24gb MIG; the final two used RTX 4090 again. This is
a mixed-hardware study. Seed-123 step-200 repeat 3 changed six labels and its
step-250 repeats 2/3 changed nine relative to repeat 1. Macro-F1 ranges are
0.012662 and 0.013150 respectively. The other four endpoints had identical label
and raw-response sequences across their three repeats.

Frozen weights, evaluator source, image digest, processor, dependency versions,
sample order, prompt and generation settings were verified across all passes.
Hardware/host/driver differences were retained in provenance. The comparison
does not isolate a hardware effect, and identical observed repeats do not prove
universal determinism. No kernel, batching or model change was introduced here.

## Successful final continuation

Allocation: **RTX 4090, EU-RO-1, US$0.89/hour GPU quote**, Pod `9y3z9qv4fjvg2e`.
The controller considered the approved global GPU/region candidates; this was
the allocation actually returned and checked. The previous PRO 6000 startup
failure remains in the cost ledger; its root cause was not established by this run.

| Observed phase | Wall time |
|---|---:|
| Data preparation | 29.759 s |
| Frozen inputs and prior-result restoration | 145.469 s |
| Last two fresh-process evaluations | 89.296 s |
| Full intent-to-confirmed-absence bound | 556.888 s |

Visible CPU quota was **13.6 cores**, with 32 logical/affinity CPUs. Over the
worker lifetime, 174 of 2,286 observed scheduling periods were throttled (7.61%).
These are cgroup scheduling counters, not a percentage of wall time, and ancestor
limits were not independently visible. The preceding MIG continuation recorded
6.8 quota CPUs, 256 visible logical CPUs and approximately 95% throttled periods.
Both hardware and host limits changed, so the large timing difference is not a
controlled CPU-only or GPU-only speedup measurement.

The two new passes averaged 0.109897 and 0.110985 seconds per generated answer;
each recorded 2,129,835,008 bytes of peak allocated CUDA memory and no invalid
labels. Generation timing excludes other evaluation work, including answer loss.

Incremental conservative compute/disk estimate: **US$0.139056**. Cumulative
campaign estimate: **US$10.610880 / 771.549163 minutes**, within the existing
US$30 / 1,100-minute authorization. These are intent-to-absence estimates, not
the provider invoice. No persistent volume was created.

Primary cleanup recorded three consecutive absence checks at **00:31:43 SGT**.
[Independent cleanup 44126820592](https://github.com/jasonjiang8866/waferVLM/actions/runs/44126820592)
separately recorded three at **00:32:33 SGT**, both on 2026-11-18.

## Evidence and verification

[Machine-readable receipt](evidence/vlm-repeat-complete.json) records artifact
hashes, allocation, cleanup, first-pass deltas and every repeat score.

- Controller: `9ce2311b436ea8536ab231f80ad27070da93edb8` (merged PR #48).
- Frozen evaluator: `11d2e5f088b350dd9391864dbe2869417fff5af7`.
- Image: `ghcr.io/jasonjiang8866/wafervlm@sha256:a6d31b4f12c2c3ee64c59d08b1e67c89e9bbf9b46cbdb4179417f1d1efa3fa79`.
- [Result artifact 14347647183](https://github.com/jasonjiang8866/waferVLM/actions/runs/44124342191/artifacts/14347647183): 120,831,263 bytes; SHA256 `dcdd7e6e44a842ad8f1b89c1817f8f01f5c4e27c57f5443050839c184bcd3c2c`.
- Inner `experiment.tar.gz`: 120,772,120 bytes; SHA256 `9211a60744e518300a83b27540a89bb388cb0af3d0533e24f90938e80af61d52`.
- Authorization artifact 14346133661: SHA256 `3f494d44642888eb9866d828ccc7745c98e30ff0f3fa6c681650663ebcacfcb0`.
- Independent cleanup artifact 14348955243: SHA256 `24114131593ed193026718fa6d845a83ddb2dc1ffa2c4a2c6b71bf6b9734b1f6`.

The outer artifact contains the original result archive, allocation, launch spec,
budget chain, cleanup and acceptance records. The inner archive preserves six
adapter weights, all prediction rows, run metadata, timings and checksums.
Actions reports result expiry at **2026-12-18 00:31 SGT**; expiry does not invalidate
an independently retained copy matching the pinned hash.

Verification used `runpod.vlm_repeat_verify.verify_repeat_archive` with the
retrieved launch spec, requiring all 18 unique processes, all 2,880 rows, the
six frozen adapter hashes, all 16 reused output hashes, exact validation identity,
recomputed metrics and preserved mixed-hardware provenance. The original ZIP,
authorization and cleanup hashes were separately verified. PR #48 passed 188
local lifecycle/contract tests and all eight applicable CI checks before merge.

## Next development step

Keep the [predeclared final study](seed-confirmation-plan.md) unchanged: original
model plus all six frozen endpoints, once each on 148 images / 143 untouched lot
groups. Do not replace it with only the best seed or checkpoint after seeing this
table. Implement its bounded worker/controller and offline verification before
opening the holdout; retain every prediction and failure, eight-class macro-F1,
class recall, invalid rate, timings and memory. Freeze paired lot-group bootstrap
analysis (10,000 resamples, seed 20261110, fixed eight labels and zero division 0)
before inference.

The next paid admission must reconcile this successful run's marker, actual
consumption and independent cleanup, then obtain fresh capacity/pricing. The
completed repeat workflow must not be rerun to spend the budget again. The
holdout has not been evaluated or authorized by this documentation update.
