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
$0.0005/1,000 Class B operations, **plus storage**. These are published request
rates, not a total import quote. Stored-GB pricing, account inventory and an
actual CPU/GPU quote were not verified in this session. No Pod or volume was
created. The authorized API route is the repository `RUNPOD_API_KEY` Secret
through GitHub Actions; browser sign-in is unnecessary. The existing
`preflight.yml` workflow now has a manual, GET-only `storage-observe` job.
It records sanitized volume/Pod inventory and 2-vCPU prices/availability, without
provisioning or exposing Pod environments. Dispatch on the reviewed branch.

## Initial import plan

Purpose: keep licensed public wafer imagery portable across the globally allowed
GPU regions. Candidate volume name: `wafervlm-public-datasets` (**planned name,
not an existing resource**). Import Carinthia-S, silicon-wafer PL and silicon-wafer
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
