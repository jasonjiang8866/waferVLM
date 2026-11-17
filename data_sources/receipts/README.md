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

**Imported and independently verified.**
[Run 44119575118](https://github.com/jasonjiang8866/waferVLM/actions/runs/44119575118)
used the repository Secret and completed import in EU-RO-1, followed by a second
fresh-mounted Pod in US-CA-2. The [import/readback receipt](runpod-import-44119575118.json)
contains both complete sets of file hashes and actual volume/Pod identities.
[Independent cleanup](runpod-cleanup-44119575118.json) confirmed no owned Pods
remained. The persistent dataset payload is **491,223,979 bytes**, plus receipts.

The earlier [attempt](runpod-import-44118683682.json) failed at the first exec
readiness probe, before the data worker started; its [cleanup](runpod-cleanup-44118683682.json)
also passed. A bounded readiness wait fixed the observed 404 → 409 → ready sequence.
See the [execution ledger](collection-execution.json) for authorization and costs.

The [preflight inventory](runpod-storage-preflight-44113725134.json) and image audit
are historical snapshots taken before cloud import; their original status fields
are preserved. Current status is in `catalog.json` and the accepted run receipt.

The offline tests mock a mount to exercise failures and boundaries. Live storage
evidence comes from the two real Pods and independent cleanup above. Receipts
are committed so they survive the Actions artifact expiry. See
[exact locations and use commands](../RUNPOD.md#verified-import).
