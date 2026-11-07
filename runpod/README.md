# RunPod worker readiness

The initial repository contains a runnable training/evaluation **worker**, not a paid-resource controller. No current workflow reads `RUNPOD_API_KEY`, creates a Pod, retries capacity or deletes account resources. The user has reported configuring that repository secret; secret existence and account inventory have not been independently verified.

## Current entry points

- `CPU protocol and model compatibility`: PR/main CPU tests; no paid RunPod calls.
- `Build worker image`: manual build; `publish=false` by default. With explicit image-publication authorization, `publish=true` publishes a source-SHA tag and prints the immutable digest. This does not create a GPU Pod.
- `scripts/run_experiment.py`: `smoke` (8 steps, validation) or `baseline` (100 steps, paired test), bounded by an explicit worker timeout.

## Container example after image publication

```bash
# Replace the image digest with the verified build output.
docker run --rm --gpus all \
  -v /path/to/prepared-data:/data:ro \
  -v /path/to/outputs:/outputs \
  -v /path/to/hf-cache:/workspace/hf-cache \
  -e WAFERVLM_IMAGE_DIGEST=sha256:VERIFIED_DIGEST \
  ghcr.io/jasonjiang8866/wafervlm@sha256:VERIFIED_DIGEST \
  --dataset /data --output /outputs/smoke-17 \
  --profile smoke --timeout-seconds 1800
```

The model revision is fixed in source, including processor/tokenizer loading. The image records the source SHA from its build argument. The initial Docker base tag is mutable; the final built **worker image digest** must be pinned before launch. No weights, data, account keys or local cache are baked into the image.

## Required before any paid launch

1. Obtain trusted WM-811K bytes and terms, prepare/audit manifests and stage them on an approved volume or verified download route.
2. Verify current source CI and build/publish the image; record source SHA and worker digest.
3. Integrate an inspected lifecycle controller (potentially `runpod-ci-broker`) in a separate reviewed change. Read live GPU price/capacity and existing project-owned resources. Confirm region/storage compatibility.
4. Obtain this experiment's total budget and allowed GPUs. A possible workload target is a single BF16-capable 24 GB GPU, but fit, availability and price are **unverified**. Prior experiments' budgets do not apply.
5. Record maximum aggregate runtime, retries, temporary storage, transfer and cleanup reserve inside the cap. A worker timeout stops the workload; it does **not** delete the Pod or enforce a billing hard cap.
6. Keep `RUNPOD_API_KEY` only in the Actions controller environment. Never inject it into the worker, logs or artifacts. Reconcile resources before retrying ambiguous creation.
7. Collect and verify all output checksums, including incremental failure artifacts. Delete only owned temporary compute and independently confirm deletion. Preserve persistent volumes.

The controller must provide cancellation-independent cleanup/reconciliation; GitHub `always()` alone does not cover every cancellation or runner-loss scenario. A scheduled reaper is a fallback, not deletion evidence.

## Launch record for this stage

| Field | State |
|---|---|
| Repository | jasonjiang8866/waferVLM |
| Workload | Qwen3.5-0.8B BF16 language LoRA, 8-step smoke |
| Model revision | 2fc06364715b967f1860aea9cf38778875588b17 |
| Dataset | Not yet acquired/verified |
| Worker image digest | Not yet built/published |
| GPU, region and quote | Unverified |
| Budget | Not authorized for this experiment |
| Paid execution | Not run |
| Cleanup | No resources created by this stage; account inventory unverified |
