# RunPod worker readiness

The repository contains a runnable training/evaluation **worker** and a pinned Broker **GET-only observation** workflow. No workflow creates a Pod, retries allocation or deletes account resources. The user has reported configuring `RUNPOD_API_KEY`; secret existence and live account inventory have not yet been independently verified.

## Current entry points

- `CPU protocol and model compatibility`: PR/main CPU tests; no paid RunPod calls.
- `Pinned public WM-811K data audit`: real public-data download, hash verification, conflict quarantine, split audit and comparison with committed evidence. No credentials/GPU.
- `Build worker image`: builds without publishing on relevant PRs; manual `publish=false` by default. `publish=true` publishes a source-SHA tag and prints the immutable digest. This does not create a GPU Pod.
- `RunPod observation and offline contract`: PRs run credential-free Broker contract checks. A manual dispatch on **main only** also reads catalog prices/capacity and Pod counts through RunPod REST GET endpoints.
- `scripts/run_experiment.py`: `smoke` (8 steps, validation) or `baseline` (100 steps, paired test), bounded by an explicit worker timeout.

## CUDA compatibility and read-only preflight

The inspected Broker dependency is pinned to [`dfd8c3c7250ac5a73c4c703932e9fb93d07dcb45`](https://github.com/jasonjiang8866/runpod-ci-broker/tree/dfd8c3c7250ac5a73c4c703932e9fb93d07dcb45). That version has no live create/delete execution implementation. Both workflow jobs check out this exact commit and install its locked dependencies in a separate environment.

`preflight.json` declares **`min_cuda_version: "12.8"`**. The actual REST adapter sends **`minCudaVersion=12.8`**, with no exact `cudaVersions` filter. Contract checks accept 12.8, 12.9, 12.10 and 13.0 and reject 12.7. Versions are compared as integer components, not floats. This is host catalog compatibility; it does not change the worker's locked Torch/CUDA runtime, prove BF16/architecture support, or replace a real CUDA smoke test.

The observation candidates are PRO 6000 Blackwell MIG 24GB, RTX 4090, RTX 3090 and RTX 5090 (one GPU, at least 24GB VRAM). This is an observation list, not paid authorization or proof of memory fit. GPU price cap, total budget and storage/transfer reserves are deliberately **unset**; budgets from other repositories are not reused. The all-zero image digest is an explicit unbuilt-image placeholder, and no volume/staging route has been selected. The workflow updates the observed source SHA to the actual workflow commit.

Consequently, authenticated preflight is expected to report `budget_inputs_missing` and finish nonzero while still uploading its observation report. A red run with this blocker is not proof of missing capacity. Inspect per-GPU observations and query/inventory statuses separately. A successful offline contract job is not a live RunPod check. Do not use this template for a paid launch: replace the placeholder image, bind dataset hashes/staging and supply an approved budget in a reviewed deployment configuration first.

`RUNPOD_API_KEY` exists only in the main-only observation step's environment. It is not passed to the worker, PR jobs, dependency installation, task JSON or artifacts. The Broker returns sanitized observations; raw Pod environment values are not uploaded.

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

1. Reproduce the [verified data subset](../docs/data-audit.md) and stage it on an approved volume or verified download route. Local preparation is complete; remote staging is not.
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
| Dataset | Real mirror verified; 790/160/160 subset audited; remote staging pending |
| Worker image digest | Not yet built/published |
| GPU, region and quote | Unverified |
| Budget | Not authorized for this experiment |
| Paid execution | Not run |
| Host CUDA selection | >= 12.8; higher versions accepted |
| Cleanup | No resources created by this stage; account inventory unverified |
