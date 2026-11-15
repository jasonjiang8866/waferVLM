# First repeat launch: provider create failure

On 2026-11-15, PR #35 passed all five CI workflows and merged at
`56af40fa13e17f20957c9d8855841d70bffc561a`. The first repeat dispatch,
[43679597770](https://github.com/jasonjiang8866/waferVLM/actions/runs/43679597770),
verified all six real checkpoints, cumulative history, inventory and a 4090
quote of US$0.89/hour. Its single `POST /pods` returned HTTP 500 before a Pod ID
was confirmed. No evaluation pass ran. The provider response does not establish
whether request size, transient capacity or another server issue caused it.

The main cleanup checked absence eight times after the ambiguous creation;
[independent cleanup 43680016685](https://github.com/jasonjiang8866/waferVLM/actions/runs/43680016685)
checked absence three more times. Both exact archives and the authorization
marker were downloaded and SHA256 verified. The committed
[receipt](../runpod/vlm-repeat-attempts.json) retains the original failed status,
both cleanup records, launch identity and conservative billing estimate.

Count 145.149327 seconds / US$0.036244 for this attempt, despite never confirming
a Pod. The cumulative estimate is **US$8.168526 / 643.124 minutes**, not a provider
invoice or a zero-spend claim. The retry's work cap is **3,454 seconds
(57 minutes 34 seconds)**, deducting the failed attempt rounded up from the
original 60-minute evaluation allowance, with the same 10-minute cleanup cap.
The aggregate evaluation envelope has not increased. All 18 passes are still
required, and the final holdout remains sealed.

The retry compresses the reviewed bootstrap from approximately 75 KB to about 22 KB.
A roundtrip test verifies every embedded file is byte-identical after decompression.
This reduces request size; it is not a confirmed explanation for the HTTP 500.
Model source, immutable image, checkpoint matrix, inference settings, GPU order
and price ceilings are unchanged. There is still one create request per dispatch.
Any additional failed paid attempt blocks further allocation until explicitly
reconciled; no automatic retry loop is added.

A separate pre-launch review found that [GitHub documents artifact redirect
links as valid for only one minute](https://docs.github.com/en/rest/actions/artifacts#download-an-artifact).
Links are now minted individually after the worker requests each archive,
after startup and data preparation. The authenticated worker accepts only a
bounded URL message matching the current archive key and random request nonce.
Links never enter Pod environment variables, logs, result artifacts or model
subprocesses. HTTP loopback tests cover delivery, authentication, nonce, replay,
destination and size checks. This addresses a separate expiry risk, not the
observed provider create error.
