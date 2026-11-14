# Independent Qwen VLM seed 42

This is the first paid execution of the frozen [seed-confirmation protocol](seed-confirmation-plan.md).
DINOv3 remains a separate experiment. Model, dataset, training split, rank, accumulation,
learning rates and step schedule are unchanged: fresh seed 42, 0 → 100 → 150 → 200 → 250.
Step 100 is the only learning-rate reduction; each later stage restores full optimizer,
scheduler and RNG state. Test and final holdout inference are not invoked.

The confirmation image was published by [run 43552594559](https://github.com/jasonjiang8866/waferVLM/actions/runs/43552594559):

- Image source: `11d2e5f088b350dd9391864dbe2869417fff5af7`.
- Image: `ghcr.io/jasonjiang8866/wafervlm@sha256:a6d31b4f12c2c3ee64c59d08b1e67c89e9bbf9b46cbdb4179417f1d1efa3fa79`.
- Controller/worker code comes from the reviewed workflow commit. The controller checks
  that workload files still match the image source, verifies anonymous image pull and
  the image's embedded source SHA, and injects only its transport/telemetry worker.

## Execution

On main, manually dispatch **WaferVLM seed 42 confirmation** (`vlm-seed42.yml`) with
the exact current reviewed main SHA. `confirm_paid=false` performs GET-only preflight.
`confirm_paid=true` additionally uploads a durable intent and may create exactly one Pod.
Pushes, pull requests and workflow reruns never allocate GPUs.

The existing total remains **US$30 / 1100 minutes**, including past attempts, disk,
startup, failures and cleanup. The seven reconciled attempts consume a conservative
US$5.903039 / 491.911 minutes. At the preferred RTX 4090 ceiling of US$0.89/hour,
this attempt reserves 185 work minutes + 10 cleanup minutes, including 60GB temporary
disk: about **US$2.92 additional**, **US$8.82 cumulative**. At the highest allowed
rate, the additional bound is about US$3.90. These are estimates, not a provider hard cap.
The bounded continuation window ends 2026-11-17 00:00 UTC; it does not reset consumption.

Before create, the controller verifies all seven historical marker ZIP hashes and IDs,
the last immutable result ZIP's cumulative cleanup record, a complete live marker
inventory, current owned-Pod absence, price/capacity, CUDA >=12.8 and account guardrails.
Dispatch history also blocks a prior paid attempt whose marker was deleted or expired.
An unknown marker from any experiment or any owned Pod blocks allocation. The workflow
shares the existing `wafervlm-paid-usd20` concurrency group. No RunPod or GitHub account
credential enters the Pod. Failed or ambiguous creation is never retried automatically.

## Retrieval and acceptance

The `wafervlm-seed42-<run>-1` Actions artifact contains the launch spec, ledger,
preflight/allocation/provision records, the worker archive and cleanup evidence.
`acceptance.json` independently verifies archive paths, complete file hash coverage,
all four training stages, LR/update counts, full-state parent chains, source/image
provenance and both endpoint identities. It means execution/integrity acceptance,
not evidence of improved accuracy. Incomplete runs preserve available outputs and
require explicit recovery reconciliation before another allocation.

Cleanup runs in the controller's finally block, in an always-run workflow step, and in
the separate **WaferVLM seed 42 independent cleanup** workflow. Deletion requires the
exact experiment/run ownership tags; repeated inventory absence is required for success.
The independent cleanup has no dependency on image verification or authorization expiry.

After seed 42 is accepted and its actual consumption and cleanup are reconciled, use
the remaining budget for independent seed 123, then the frozen 18 fresh-process
validation passes. This workflow intentionally cannot repeat seed 42 or launch seed 123
against a stale ledger. The final 148-image lot holdout remains sealed.

## First live dispatch and receipt fix

[Run 43553982790](https://github.com/jasonjiang8866/waferVLM/actions/runs/43553982790)
passed anonymous image provenance, historical budget, owned-Pod absence, live
capacity/rates and account checks. RTX 4090 was available in five regions at
US$0.89/hour. It then failed inside authorization_guard, before launch-spec,
resource intent or POST /pods: the newly uploaded marker had not appeared in
the repository artifact listing. Only ledger, preflight and cleanup files were
produced. Both cleanup paths independently observed absence. No training ran.

The fix passes the exact upload receipt ID to the controller. If the list lags,
it verifies that specific artifact's repository/run/SHA/digest and ZIP contents;
only a 404 on this GET is retried (at most 55 seconds), never Pod creation.
Missing historical markers and unknown attempts remain blocking. The exact failed
run and its two immutable evidence ZIPs are reconciled in
[vlm-precreate.json](../runpod/vlm-precreate.json); a rerun, changed source, resource
intent or changed cleanup evidence invalidates that zero-increment accounting.
No second paid dispatch was submitted during this repair.
