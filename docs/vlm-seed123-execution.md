# Independent Qwen VLM seed 123

This prepares the second fresh seed of the frozen
[confirmation protocol](seed-confirmation-plan.md). The first seed-42 GPU run is
[43562568201](https://github.com/jasonjiang8866/waferVLM/actions/runs/43562568201),
launched on 2026-11-14 from `537e59bd34e1415bf5091b90bb270fcc3ac08f22`.
Live logs have confirmed optimizer updates. Completion, checkpoint integrity,
final cost and cleanup are still pending at the time of this preparation.

## Admission gate

`runpod/vlm-completed.json` is intentionally empty. Do not fill it with estimated
success, synthetic test fixtures or a workflow's green status alone. Before
dispatching seed 123:

1. Download the seed-42 authorization marker, result ZIP and independent-cleanup
   ZIP. Record their exact artifact IDs and SHA256 values, run ID and controller SHA.
2. Independently verify `experiment.tar.gz` with `verify_training_archive`, using
   the original `launch-spec.json` and pinned dataset metadata. Copy the exact
   `acceptance.json`, `cleanup.json` and independent `cleanup.json` into the receipt.
3. Review and commit that receipt to `vlm-completed.json`. Each record requires
   `seed`, `run_id`, `controller_sha`, `marker_artifact_id`, `marker_sha256`,
   `result_artifact_id`, `result_sha256`, `independent_cleanup_artifact_id`,
   `independent_cleanup_sha256`, `acceptance`, `cleanup`, `independent_cleanup`.
4. Review main and manually dispatch **WaferVLM seed 123 confirmation** with its
   exact SHA. `confirm_paid=false` runs GET-only preflight; `true` may allocate one
   Pod only after all gates pass. Reruns cannot allocate.

The controller rechecks immutable ZIP bytes, all four training stages and both
endpoint states, original run identity, cumulative consumption, and both cleanup
receipts. Missing acceptance, altered history, any prior unreviewed paid attempt,
or insufficient remaining budget blocks allocation. It never treats quality as
accepted: improved accuracy still requires the frozen validation experiment.

## Budget and isolation

The original **US$30 / 1100-minute VLM total** is unchanged. Seed 123 starts from
seed 42's measured conservative intent-to-absence billing bound, including the
earlier US$5.903039 / 491.911-minute history. It reserves the entire 185 work minutes
plus 10 cleanup minutes or refuses to launch. It cannot silently shorten the
training recipe. Quote and cleanup recovery both carry the updated cumulative
USD and time counters; they do not reset to the seven-run baseline.

DINOv3 has a separate US$10 / 500-minute authorization. Only its exact
`runpod-authorization-wafervlm-dinov3-seed17-v1` marker is excluded from the VLM
ledger, after checking the marker bytes and originating main/manual workflow.
Unknown markers still block. Both experiments retain the shared paid-workflow
mutex and global owned-Pod absence check, so this exception permits no concurrent
allocation and makes no claim that DINOv3 was free or completed.

Seed 123 has distinct Pod ownership tags, result artifacts and an independent
cleanup workflow. Its cleanup cannot delete seed 42 or DINOv3 Pods. The pinned
confirmation image and workload remain unchanged; the injected worker receives
only the selected seed and transport metadata, never account credentials.

## Scientific boundary

Fresh seed 123 follows **0 → 100 → 150 → 200 → 250**, with the same dataset split,
LoRA rank, accumulation and model revision. Only the step-100 transition lowers
the learning rate; subsequent stages restore optimizer, scheduler and RNG state.
It does not evaluate the test set or final 148-image lot holdout.

After seeds 17, 42 and 123 have frozen step-200/250 endpoints, the next experiment
is the planned 18 fresh-process validation passes. This PR does not dispatch that
evaluation, select a winning checkpoint or unlock the final holdout.
