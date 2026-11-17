# Collection evidence

## Verified local acquisition

[initial-download.json](initial-download.json) records **24 files / 491,223,979
bytes** downloaded from the primary Zenodo records, with exact size, publisher
MD5, locally computed SHA256 and ZIP expansion checks. All measured SHA256 values
are now pinned in `catalog.json` and a second local verification passed.

| Source | Download bytes | Expanded payload bytes |
|---|---:|---:|
| Carinthia-S (ZIP + description) | 141,330,049 | 155,168,125 |
| Silicon-wafer PL/SEM/CSV | 38,583,583 | 38,583,583 |
| Silicon-wafer EBSD | 311,310,347 | 581,340,387 |

The conservative batch plan reserves **1,334,975,687 bytes** for downloaded plus
expanded content; measured corresponding total is **1,266,316,074 bytes**. Both
are below the user's 10,000,000,000-byte approval threshold. Original archives
and files only would occupy ~491 MB on the volume, plus small receipts/metadata.

[carinthia-s-image-audit.json](carinthia-s-image-audit.json) records the full
image/mask decode and dimension audit. The 12 published pairs are selected by
source filename within each numeric class, with no pixel changes. A separate
original 516,254-byte silicon SEM example is also committed.

Reproduce the full decode audit with the project's Pillow dependency:
`python scripts/audit_collection_images.py`. The archive contains mixed mask
modes and intermediate grayscale values; this audit reports them without silently
normalizing the source annotations.

## Cloud status

**Not imported.** There is no cloud import receipt or fresh-mount readback
evidence yet. No billable resources were created. The repository Secret
successfully accessed RunPod in
[preflight run 44113725134](https://github.com/jasonjiang8866/waferVLM/actions/runs/44113725134).
The sanitized [inventory receipt](runpod-storage-preflight-44113725134.json)
records one existing Global Volume, no wafer-named Pod, and a lowest CPU quote
of **$0.06/hour**. The read-only preflight did not touch unrelated workloads.

The complete CPU import/readback/cleanup workflow is prepared. It requires a
one-off CPU import budget (proposed **$0.50**, 25-minute aggregate work window)
before paid dispatch. The older training budgets are not used. The local source
cache remains reproducible staging, not durable cloud storage. See the
[exact import inputs and cost assumptions](../RUNPOD.md).

The offline tests mock a mount to test the copy/receipt protocol. That is **not**
a live Global Volume integration result. A successful future import must add its
sanitized source/volume identity, readback and temporary-Pod cleanup evidence.
