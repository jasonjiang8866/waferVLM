# Real WM-811K data audit — 2026-11-07

The pinned public recipe completed locally using the repository lock (Python 3.12.14, pandas 2.3.3, NumPy 2.5.4 and Pillow 12.3.0). `wafervlm audit` passed on the generated PNGs and manifests. This is a data result, not a model accuracy result.

## Source and identity

- Mirror: https://www.kaggle.com/datasets/qingyi/wm811k-wafer-map (listing version 1).
- Download: https://www.kaggle.com/api/v1/datasets/download/qingyi/wm811k-wafer-map
- Archive: 156,686,283 bytes; SHA256 `0e4bd4316a110a04e475a88071f5e1c7fe4303b12ee53c851d2cae4d995a93e2`.
- Member: `LSWMD.pkl`, 2,095,505,977 bytes; SHA256 `1d04fccb3dd3176b276878b926b20fead7e077c5751e4d353ea9741a5e7b5c65`.
- Prepared `dataset.json` SHA256: `c588b3ef59d9dba57cfc745ec20db903e6ff2dcc75504112348f8b9f1ad3ab68`.
- Machine-readable counts, manifest digests and protocol: [wm811k-lot17.json](evidence/wm811k-lot17.json).

The mirror lists CC0. This records the mirror's statement, not an independent audit of the original data license. MIR Lab was unavailable during acquisition. The repository does not claim these bytes came directly from NVIDIA/MIR Lab, and does not commit the pickle or rendered images. Downloads fail closed if the mutable mirror URL starts serving different bytes. Checksums establish identity, not pickle safety; explicit trust opt-in remains required.

## Actual counts

The source contains 811,457 rows, including 25,519 rows in the eight target defect classes. The remaining 785,938 normal/unlabeled rows are excluded from this closed-set task.

| Class | Source rows | Quarantined | Train | Validation | Test |
|---|---:|---:|---:|---:|---:|
| center | 4,294 | 1 | 100 | 20 | 20 |
| donut | 555 | 0 | 100 | 20 | 20 |
| edge-loc | 5,189 | 2 | 100 | 20 | 20 |
| edge-ring | 9,680 | 2 | 100 | 20 | 20 |
| loc | 3,593 | 1 | 100 | 20 | 20 |
| near-full | 149 | 0 | 90 | 20 | 20 |
| random | 866 | 0 | 100 | 20 | 20 |
| scratch | 1,193 | 0 | 100 | 20 | 20 |

Three identical-map groups carry conflicting target labels: center/loc (rows 5954/643483), edge-ring/edge-loc (6673/645298), and edge-ring/edge-loc (6917/642913). All six rows are quarantined; no majority-label repair is applied. After quarantine and exact deduplication, 25,422 unique maps remain before sampling.

The original seed 17, 70/15/15 lot-group hash allocation and 100/20/20 caps are unchanged. The resulting subset contains **790 train, 160 validation, 160 test** images from **660, 133 and 144** distinct lots, respectively. All eight classes are present. IDs, lots, raw-map hashes and rendered-image hashes have no cross-split overlap. This does not rule out near-duplicates or unrecorded manufacturing relationships.

## Reproduce

```bash
uv sync --frozen
uv run --frozen python scripts/prepare_public_data.py --trust-pickle --output data/wm811k-lot17
uv run --frozen wafervlm audit --dataset data/wm811k-lot17
diff -u docs/evidence/wm811k-lot17.json data/wm811k-lot17/acquisition.json
```

The public-data CI performs the same independent download, preparation, audit and evidence comparison, preserving reports/manifests as an Actions artifact. It does not upload the raw pickle or images. Rebuild the PNGs from the pinned source before staging an authorized GPU task. No trained checkpoint or RunPod resource is involved.
