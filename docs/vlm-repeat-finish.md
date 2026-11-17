# Finish the last two validation repeats

Prepared 2026-11-17 after the measured MIG timeout. This is one new bounded
allocation within the existing US$30 / 1100-minute campaign, after closing and
reconciling the previous two-attempt continuation slice.

## Verified state: 16/18, still incomplete

[Run 44047184828](https://github.com/jasonjiang8866/waferVLM/actions/runs/44047184828)
reused the original 14 byte-identical pass trees and completed two fresh CUDA
processes on RTX PRO 6000 Blackwell Server Edition MIG 1g.24gb:
`seed123-step200-repeat3` and `seed123-step250-repeat1`. The following process
wrote 45/160 predictions before timeout. Those 45 predictions are retained as
failure evidence and are **not** a completed pass or reusable inference input.

Data preparation took 33.586 seconds, input transfer/preparation 62.107 seconds,
and validation 2074.143 seconds. Actions observed the two completed passes about
14.2 and 15.9 minutes apart. The completed second process's log shows fast
cached-weight loading. Its mean generation time was 2.733 seconds per map;
that measure excludes preprocessing, answer-loss evaluation and model loading.

The container exposed 256 logical/affinity CPUs but a cgroup quota of 6.8 CPUs.
Between start and end, 20,365 of 21,383 observed CPU scheduling periods were
throttled (95.24%). This establishes heavy CPU throttling, but does not isolate
its causal contribution from the different GPU, host and fallback kernels.
No thread, kernel, processor or model change is adopted in this continuation.

The retrieved ZIP and inner archive were independently hash-verified. All 16
complete pass trees, raw predictions, metrics, frozen weights and validation
identities were rechecked; the original 14 checksum manifests remain identical.
See the [16-pass receipt](evidence/vlm-repeat-finish.json) and
[public artifact pins](evidence/vlm-repeat-finish-pin.json).
The [independent cleanup](https://github.com/jasonjiang8866/waferVLM/actions/runs/44053623260)
confirmed three consecutive absence checks after the primary cleanup. No
persistent volume was created.

## One allocation, two missing processes

- Copy all 16 verified output directories, then run only
  `seed123-step250-repeat2` and `seed123-step250-repeat3` in fresh processes.
- Prefer RTX 4090; allow the already authorized RTX 5090, then RTX 3090 when
  their catalog, CUDA >=12.8 and per-model price checks pass. Exclude MIG from
  this slice because its measured throughput exhausted the previous envelope.
- Make at most one create request. An ambiguous create or another paid attempt
  blocks further allocation until its inventory, artifacts, cost and cleanup
  have been reconciled.
- Preserve the same image digest, evaluator source, software, six adapters,
  validation order, prompt, greedy generation and first-pass headline rule.
  The two completed MIG passes remain in the report; the study has mixed
  hardware. Do not attribute score changes to hardware alone or describe it as
  homogeneous-hardware repeatability.
- Require all 18 passes / 2880 predictions, recomputed metrics, every reused
  checksum, result retrieval and primary plus independent Pod absence checks.
  A further timeout is still a failed study. The 148-image holdout stays untouched.

## Time and cost

The MIG run's conservative intent-to-absence bound is **2355.088 seconds /
US$0.457233**. Including all reconciled prior attempts, cumulative usage is
**US$9.472522 / 738.893 minutes**, leaving **US$20.527478 / 361.107 minutes**.

Reserve **40 work minutes + 10 cleanup minutes** for this one allocation. At
the highest permitted choice, RTX 5090 at US$1.19/hour plus the existing 60 GB
temporary-disk estimate, this adds at most approximately **US$0.999108**, taking
the planned cumulative estimate to **US$10.471629 / 788.893 minutes**. These
are conservative estimates, not provider invoices or provider-enforced caps.
The existing continuation window ends 2026-11-20 08:00 SGT.

This change excludes the separate monitoring patch in PR #43. It changes only
continuation orchestration and receipt verification; it does not modify the
in-flight/finished experiment or DINOv3's code and budget.
