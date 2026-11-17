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

**Not imported.** There is no Global Volume ID, cloud receipt or fresh-mount
readback evidence in this change. No billable resources were created. The RunPod
storage browser session requires sign-in; actual inventory and storage/compute
quotes remain unverified. The local source cache is reproducible staging, not
durable cloud storage. See [import runbook](../RUNPOD.md).

The offline tests mock a mount to test the copy/receipt protocol. That is **not**
a live Global Volume integration result. A successful future import must add its
sanitized source/volume identity, readback and temporary-Pod cleanup evidence.
