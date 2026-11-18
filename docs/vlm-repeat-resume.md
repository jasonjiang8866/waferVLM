# Complete the timed-out 14/18 VLM repeat study

**2026-11-18 completion:** [all 18 passes are independently verified](vlm-repeat-results.md), with 2,880 predictions and both cleanup paths confirmed. The incomplete-run descriptions below are retained as history. Quality/generalization remain unaccepted; the holdout is untouched.

**2026-11-17 outcome:** the bounded fallback reached 16/18 and timed out again.
Results were retrieved and both cleanup checks passed. The [measured failure and
final-two-pass plan](vlm-repeat-finish.md) supersede the prospective budget below.

Prepared 2026-11-17. This continuation is validation only; it does not train a
model, change the frozen recipe, select the best repeat, or access the holdout.

**Capacity recovery update:** [continuation dispatch 44043355369](https://github.com/jasonjiang8866/waferVLM/actions/runs/44043355369)
passed preflight but the RTX 4090 create request returned HTTP 400 `capacity`.
No allocation or new inference was confirmed. Primary cleanup verified absence
eight times; [independent cleanup](https://github.com/jasonjiang8866/waferVLM/actions/runs/44044322434)
also passed. The bounded fallback below supersedes the initial same-GPU plan.

The next dispatch skips that rejected GPU type and tries the already approved
MIG 24GB, then RTX 3090 if the catalog has no acceptable MIG candidate. Each
dispatch makes at most one create request; ambiguous creates are never retried.
This is the second and final allocation attempt in this continuation slice.
Unknown subsequent attempts still block new allocations.

The complete report will retain the original 14 RTX 4090 passes and identify
the four new passes' actual GPU/host. The verifier binds each pass to its own
origin's GPU. A mixed-GPU result is a completed validation study with environment
variation, **not a homogeneous-hardware repeatability result**; checkpoint/seed
comparisons spanning GPUs must be interpreted with that confound. No model is
promoted on this evidence alone.

The rejected create adds a conservative 144.352 seconds / US$0.036045 to the
ledger despite no confirmed allocation. Current totals are **US$9.015289 /
699.642 minutes**. For the potentially slower fallback GPU, reallocate 15 more minutes from the
existing campaign allowance. Deducting the rejected attempt leaves **2555 work
seconds plus 600 cleanup seconds**. The shared continuation slice, including the
rejection, is bounded by 55 minutes and approximately US$0.65 at the MIG ceiling.
MIG is capped at US$0.69/hour; RTX 3090 at US$0.50/hour, plus the same temporary
disk. The cumulative US$30 / 1100-minute limits remain unchanged. Actual invoices
may be lower than this conservative intent-to-absence accounting.

## Verified failure and reusable evidence

[Run 43681461001](https://github.com/jasonjiang8866/waferVLM/actions/runs/43681461001)
performed real RTX 4090 inference but failed the study's acceptance gate.
Its validation subprocess ran for 2931.59 seconds and timed out during
`seed123-step200-repeat3`. Exactly 14 complete passes (2240 predictions) were
retrieved. Both the primary and
[independent cleanup](https://github.com/jasonjiang8866/waferVLM/actions/runs/43685646665)
confirmed the Pod absent. No persistent volume was created.

The original 60-minute evaluation allowance included provisioning, data staging,
four input archive transfers, all 18 fresh inference processes, and retrieval.
It also deducted the preceding failed-create lifecycle. This left insufficient
inference time, and the original runner always started a new 18-pass directory.
Increasing an Actions job timeout alone would not fix either limitation.

The result artifact is pinned to SHA256
`5634554e305d686b51632d4d03ea1212ee58aac6ad421f09813c34cf2029a678`;
its inner archive is pinned to
`821e28fb4d9724b3bfec9272640837e71b90ce51747c08a081f1efbe3e0f7c3a`.
The [reuse receipt](evidence/vlm-repeat-resume.json) records each completed
directory's checksum manifest and prediction hash. The unfinished 15th directory
is discarded; it is never counted as a completed pass.

## Continuation contract

The same manual workflow now resumes this exact verified source:

- Reverify original artifact bytes, endpoint weights, validation identities,
  completed process identities, raw predictions, metrics and both cleanup paths
  before allocating compute. Unknown prior attempts still block allocation.
- Request fresh scoped URLs only when the worker is ready. No account token is
  transferred to the worker. The partial result is a fifth hash-pinned input.
- Use the original immutable image and unchanged `repeat_validation` function.
  Intercept only its subprocess boundary: copy the 14 verified output directories
  byte for byte; launch the other four original evaluation commands normally.
- Run `seed123-step200-repeat3` and `seed123-step250-repeat1..3` in fresh processes.
  The initial attempt requested RTX 4090; the recovery above permits the approved
  fallback with explicit per-pass hardware provenance.
- Recompute the full 18-pass / 2880-prediction report with the original verifier.
  Require the reused bytes to match their committed hashes. Preserve both hosts'
  provenance; do not claim all passes came from the same environment.
- A second timeout remains failed with partial results retained. Acceptance still
  requires every pass, every integrity check and verified temporary Pod removal.

## Cost and time

Including the timed-out allocation, conservative cumulative consumption is
**US$8.979244 / 697.236 minutes**, leaving **US$21.020756 / 402.764 minutes**
under the existing **US$30 / 1100-minute** campaign authorization.

The user's 2026-11-17 continuation request uses that remaining campaign budget.
This dispatch reserves **30 work minutes plus 10 cleanup minutes**; it does not
reset historical consumption or increase either campaign cap. At the approved
RTX 4090 ceiling of US$0.89/hour plus the 60 GB temporary disk, its full lifecycle
estimate is **US$0.599286**, taking the cumulative estimate to **US$9.578529 /
737.236 minutes**. The new continuation window closes 2026-11-20 08:00 SGT;
training retains its previous expiry. An unknown or additional paid dispatch
must be reconciled before another allocation. Estimates are not provider hard
caps or invoices.

The 30-minute allowance provides roughly twice the observed time needed for four
passes, including preparation and retrieval. DINOv3's budget and workflow remain
separate. No model-quality or completed-study claim is made until GPU acceptance.
