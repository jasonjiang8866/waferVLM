# Global Volume storage / 跨区域数据存储

## Verified platform behavior

Official documentation checked 2026-11-17:

- [Global Volume overview](https://docs.runpod.io/storage/globalvolume/overview):
  region-independent object-backed storage; no S3 API, atomic rename, file locks,
  hard links or POSIX permission bits. Network-volume S3 instructions do **not**
  apply. Updates may not be visible to an already-running reader.
- [Pod attachment](https://docs.runpod.io/storage/globalvolume/globalvolume-pods):
  attach when creating the Pod; read back its actual mount path. CPU Pods support
  Global Volumes through REST v2, although the console attaches GPU Pods only.
- API: `https://api.runpod.io/v2/global-volumes`; create with a name, not a
  preallocated GB size. Pod attachment uses `mounts.global`. Do not guess an ID.

Global Volume requests cost $0.005/1,000 Class A operations and
$0.0005/1,000 Class B operations, **plus ongoing storage**. The published overview
does not quote stored-GB pricing; do not substitute network-volume prices.

The repository `RUNPOD_API_KEY` Secret was verified through the successful
[GET-only Actions preflight](https://github.com/jasonjiang8866/waferVLM/actions/runs/44113725134)
on 2026-11-17 at 23:23 SGT. The account had **1 Global Volume and 1 Pod**, neither
named for waferVLM. No resources were changed. Six 2-vCPU flavors were listed;
`cpu3c` (4 GB RAM) was **$0.06/hour**, with HIGH availability in 15 data centers.
This is catalog evidence, not a capacity reservation. See the
[sanitized receipt](receipts/runpod-storage-preflight-44113725134.json).

## Prepared Actions import (not yet executed)

Use the existing **RunPod observation and offline contract** workflow
(`.github/workflows/preflight.yml`) on `feat/wafer-dataset-collection` at its
reviewed commit. No merge is necessary to dispatch this reviewed branch.

| Input | Value for the initial batch |
|---|---|
| `storage_action` | `import` |
| `confirm_paid_storage_import` | `true`, only after CPU import budget approval |
| `approved_storage_budget_usd` | Proposed **0.50**, including temporary compute/disk/request/cleanup reserve |
| `global_volume_id` | Empty selects the unique existing Global Volume; an explicit owned ID is also accepted |

The controller rechecks inventory and quotes, then runs **two sequential CPU
Pods**: one importer and one fresh-mount verifier. Current compute estimate is
$0.06 × 25/60 = **$0.025** for the full 25-minute aggregate work window.
The controller refuses actual compute rates above $0.10/hour; cleanup has a
separate 5-minute Actions job. The $0.50 proposal includes headroom for temporary
disk/request/cleanup costs; it is not a provider-enforced billing hard cap.
Long-term storage for ~491 MB is covered by the user's data-collection scope,
but its stored-GB rate remains unverified and is not included in this one-off
compute estimate. No prior model-training budget is reused.

Implementation: `runpod/storage_import.py` + `runpod/storage_worker.py`.
The image is official Docker Hub Python 3.12 slim, pinned to the Linux amd64
manifest digest `sha256:05a994d2344eb364cc186910f4fc8786a38215f39f00d0acc25bff33a08f4177`.
The source bundle comes from the checked-out commit; no account key enters the
Pod, and no HTTP/SSH ports are opened. REST v2 exec starts the worker and retrieves
bounded receipts. Pod API mount identity/path and actual price are checked before
any worker writes data. Global Volume attachment is an array of
`{volumeId, path}` entries in `mounts.global`.

Both in-process `finally` cleanup and an independent Actions cleanup job delete
only Pods matching this run's identity, source and pinned image. An ambiguous
create is reconciled through inventory; it is never blindly retried. Cleanup
retains the Global Volume and its contents. An existing version directory blocks
re-import and requires reconciliation. Manual execution/cancellation can still
leave resources if the provider or GitHub is unavailable; final acceptance
requires confirmed inventory removal, not only the worker's result.

## Initial import plan

Purpose: keep licensed public wafer imagery portable across the globally allowed
GPU regions. Prefer an existing `wafervlm-public-datasets` volume, otherwise the sole existing
Global Volume; ambiguous inventory requires an explicit ID. No new volume is
created automatically. Import Carinthia-S, silicon-wafer PL and silicon-wafer
EBSD; omit original Carinthia to avoid a second copy of the same image family.

Run `plan` for exact reviewed limits. Store original archives/files only; unpack
onto a worker's local scratch disk for training. A startup reader should read the
receipt and verify SHA256 before copying data locally. Do not put virtualenvs,
mutable checkpoints, shared locks or training outputs on this Global Volume.

Before provisioning, inspect existing owned Pods/volumes, reuse a suitable Global
Volume when possible, verify region/cloud compatibility and an exact compute
price/runtime/cleanup bound. Data storage below 10 GB is within the user's
collection request; the older model-training budgets are not a new upload-Pod
budget. Resolve the uploader compute cost if a new Pod is necessary. Keep the
account key in GitHub Actions Secrets; do not send it to a Pod.

## Commands on a verified, already attached Pod

Check the Pod API's `mounts.global` ID/path and `findmnt` first. The script checks
that the destination is a mount point, but **cannot identify RunPod ownership or
volume type from a local path**. Supply the verified ID/path; do not merely create
a directory named `/workspace-global`.

```bash
# Run from the reviewed waferVLM commit; local cache must be OUTSIDE the volume.
python scripts/collect_wafer_data.py fetch \
  --datasets carinthia-s silicon-wafer-pl silicon-wafer-ebsd \
  --cache /tmp/wafer-source --receipt /tmp/wafer-download.json

# Set VERIFIED_GLOBAL_MOUNT and VERIFIED_GLOBAL_VOLUME_ID from the actual Pod.
python scripts/collect_wafer_data.py ingest \
  --datasets carinthia-s silicon-wafer-pl silicon-wafer-ebsd \
  --cache /tmp/wafer-source --mount "$VERIFIED_GLOBAL_MOUNT" \
  --volume-id "$VERIFIED_GLOBAL_VOLUME_ID" --receipt /tmp/wafer-import.json

# Repeat on a FRESH mount / independently started reader before final acceptance.
python scripts/collect_wafer_data.py verify-volume \
  --datasets carinthia-s silicon-wafer-pl silicon-wafer-ebsd \
  --mount "$VERIFIED_GLOBAL_MOUNT" --volume-id "$VERIFIED_GLOBAL_VOLUME_ID" \
  --receipt /tmp/wafer-independent-readback.json
```

Only one writer may import into `waferVLM-datasets/` at a time. Enforce this in
the orchestrator (e.g. a fixed Actions concurrency group with cancellation off),
not with a file lock on the volume. The cumulative check relies on a fresh mount
and serialized writes; it is not a provider-enforced quota.

Destination layout:

`<mount>/waferVLM-datasets/<dataset-id>/<provider-version>/<original-file>`

Each version gets `receipt.json` **last**, after payload readback. No remote
rename or chmod is required. A preexisting version directory blocks ingestion;
verify it if completed, or inspect an interrupted import before recovery. Do not
overwrite/delete it automatically. Failed uploads may consume storage and need
explicit reconciliation. A missing/mismatched receipt is never a successful import.

## Acceptance record

Record source commit/catalog hash, dataset bytes/SHA256, actual volume ID/path,
uploader Pod ownership/run ID, measured or estimated costs, first readback,
independent fresh-mount readback, and independent removal of the owned temporary
Pod. Retain the Global Volume. Store sanitized receipts here after execution;
do not claim persistence based only on a local copy or an exit code.
