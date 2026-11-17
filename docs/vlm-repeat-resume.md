# Complete the timed-out 14/18 VLM repeat study

Prepared 2026-11-17. This continuation is validation only; it does not train a
model, change the frozen recipe, select the best repeat, or access the holdout.

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
  Restrict this continuation to RTX 4090 to match the original study's GPU type.
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
