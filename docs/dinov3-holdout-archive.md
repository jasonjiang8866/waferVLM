# Frozen DINOv3 holdout: evidence retention and archival verification

**Durable-copy status: pending.** PR #71 published all verified predictions and the six frozen selected-model hashes, but GitHub Actions artifacts are **expiring evidence, not permanent backups**. Do not claim archival is complete until the original ZIPs have been copied to an approved independent destination, read back, and verified.

## Original artifacts and expiry

The [machine-readable retention manifest](evidence/dinov3-holdout-retention.json) records the exact original byte digests, freeze SHA-256 and selected-export hashes.

| Artifact | ID | Original run | Expiration (UTC) |
| --- | --- | --- | --- |
| Complete holdout, six selected exports and logs | `14682595469` | [44843362544](https://github.com/jasonjiang8866/waferVLM/actions/runs/44843362544) | 2027-02-19 00:22:18 |
| Independent cleanup receipt | `14682995203` | [44844163133](https://github.com/jasonjiang8866/waferVLM/actions/runs/44844163133) | 2027-02-19 00:29:32 |
| Original dispatch intent | `14681916299` | [44843362544](https://github.com/jasonjiang8866/waferVLM/actions/runs/44843362544) | 2027-02-19 00:22:18 |

The primary ZIP is 2,152,945 bytes and contains `experiment.tar.gz` (2,140,488 bytes). The inner archive retains six `selected.safetensors` exports, their recorded training files and logs, and six holdout prediction files (148 predictions each). Git tracks the published prediction-level evidence but **not** all selected-weight bytes.

## Local preservation (no GPU, no RunPod credentials)

1. Download and keep the **original ZIP bytes** for all three GitHub Actions artifacts. Do not unpack and re-ZIP when verifying the outer SHA-256: recompression produces different bytes. GitHub CLI's `gh run download` normally extracts an archive and therefore does not preserve its original ZIP digest.
2. From the repository root, run the committed-evidence check, then verify the original downloaded ZIPs:

```bash
python scripts/verify_dinov3_holdout_evidence.py
python scripts/verify_dinov3_holdout_evidence.py \
  --primary /secure/dinov3-study-holdout-44843362544.zip \
  --cleanup /secure/dinov3-study-cleanup-44843362544.zip \
  --intent /secure/wafervlm-dinov3-study-v1-44843362544.zip
```

The second command verifies the three original outer hashes, inner TAR SHA-256, six frozen model weight hashes, six prediction-file hashes/counts, acceptance, launch spec and both cleanup receipts. The intent ZIP is optional, but primary and independent-cleanup ZIPs must be supplied together. The code uses only the Python standard library and does not access the network.

3. Copy the original verified ZIPs and immutable digest manifest to an access-controlled persistent location (for example, a private versioned object store or independently backed-up encrypted drive). Retrieve the **stored** bytes and repeat the verification against them. Record the approved storage URI, responsible owner and UTC read-back timestamp. Update `durable_copy.status` from `pending` to `verified` in a separate reviewable change only after successful read-back. Neither a deleted Pod nor an ephemeral RunPod disk is a durable backup.
4. Check dataset and model redistribution rights before publishing a Release. The [WM-811K dataset card](../data_sources/cards/wm811k.md) notes that the mirror's CC0 declaration has not independently established the original publisher's terms; do not publish wafer images or pretrained backbone weights solely on the basis of this repository's MIT code license. An access-controlled location avoids assuming publication rights.

## What CI does—and does not—prove

The existing **CPU protocol and model compatibility** workflow runs `python scripts/verify_dinov3_holdout_evidence.py` and the associated tampering tests on PRs. They verify that frozen manifests, committed reports, selected-export hashes, RunPod ledger, artifact identity, conservative cost accounting and independent cleanup records are internally consistent. No GitHub artifact download or paid provider call is made by those new checks.

The CPU-only CI does **not** prove that an independent archive has been created, that future storage is still available, or that actual billed charges equal lifecycle estimates. The manual downloaded-ZIP read-back check above closes the raw-artifact integrity gap; long-term retention remains open until independently verified.
