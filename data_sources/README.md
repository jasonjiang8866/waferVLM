# Wafer image data collection / 晶圆图像数据收集

Keep this collection in **waferVLM**, alongside the existing experiment. Start here
when choosing data for a new project. Sources were checked on **2026-11-17**.

## Available sources

| Dataset / data card | Modality and scale | Size | Collection status / useful next project |
|---|---|---|---|
| [Carinthia-S](cards/carinthia-s.md) | Real SEM, 4,591 images + masks, 6 numeric classes | ZIP 139,225,942 B; unpacked 153,064,018 B | **12 image/mask pairs in Git**; full download verified; segmentation, rare-defect retrieval |
| [Carinthia](cards/carinthia.md) | Same SEM image family; classification labels | ZIP 133,840,870 B | Metadata only; prefer Carinthia-S to avoid duplicating images |
| [Silicon wafer PL](cards/silicon-wafer-pl.md) | Photoluminescence RAW images + one SEM PNG + CSV measurements | 21 files, 38,583,583 B | CC-BY-4.0; process-condition image comparison; not a large labeled benchmark |
| [Silicon wafer EBSD](cards/silicon-wafer-ebsd.md) | 50 × 50 scan positions, 480 × 480 diffraction patterns | ZIP 311,310,347 B; measured unpacked 581,340,387 B | CC-BY-4.0; orientation / representation learning; not defect-label classification |
| [WM-811K](cards/wm811k.md) | 811,457 electrical bin maps; existing project data | ZIP 156,686,283 B; pickle 2,095,505,977 B | Existing pinned recipe; expand training within a new protocol |
| [MixedWM38](cards/mixedwm38.md) | >38,000 maps; 38 normal/single/mixed patterns | Exact bytes not verified | Links only; confirm original terms and corrected C7/C9 labels before ingestion |
| [Long-tailed SEM](cards/long-tailed-sem.md) | Fab SEM, normal + 11 defect categories | Exact bytes/count not verified | Links only; explicit reuse license not found |

[catalog.json](catalog.json) contains the three first-party CC-BY-4.0 sources
enabled for the downloader. The other four are discovery entries, **not silently
downloaded datasets**. Exact provider file sizes/checksums and license statements
are retained in [metadata/](metadata/). Runtime acquisition and import receipts
belong in [receipts/](receipts/).

## Small data in Git, full data on Global Volume

- Default Git budget: **≤5 MB per preview set, ≤20 MB total** in this collection.
  Include original source IDs, license, attribution and hashes. A small complete
  dataset may also fit; do not truncate one and describe it as complete.
- Larger payloads stay outside Git under ignored `data/collection/`, then in a
  RunPod **Global Volume**, retaining immutable provider versions and archives.
  The collection is excluded from GPU image builds by the current Docker allowlist.
- **10 GB = 10,000,000,000 bytes.** A dataset or batch above this requires the
  user's explicit approval **before ingestion**. Count archives plus planned
  unpacked copies; unknown sizes block acquisition until bounded. The importer
  also adds existing collection bytes, including incomplete uploads, to prevent
  silently crossing the limit through repeated smaller batches.
- This first implementation has **no `--force` or approval boolean bypass**. A
  larger import needs a reviewed, dataset-specific plan with size/cost/approval
  evidence and a separately reviewed bound change. Do not split it to evade review.

## Reproduce without a GPU

Only Python 3.11+ standard-library modules are needed:

```bash
python scripts/collect_wafer_data.py plan \
  --datasets carinthia-s silicon-wafer-pl silicon-wafer-ebsd
python scripts/collect_wafer_data.py fetch \
  --datasets carinthia-s silicon-wafer-pl silicon-wafer-ebsd \
  --receipt data/collection/download.json
python -m unittest discover -s tests -p test_data_collection.py -v
```

The downloader pins provider byte sizes and MD5 checksums, adds SHA256 receipts,
checks ZIP member paths/CRC/expanded size, and never executes data, unpickles it or
extracts archives. MD5 is the publisher's integrity signal, not a safety proof.
Verified SHA256 values are pinned in the catalog after first acquisition. HTML
documentation is retained as data, never executed by the collector.

For Global Volume setup, verified mount requirements, import and fresh-mount
readback, see [RUNPOD.md](RUNPOD.md). **Cloud import is not complete until there is
an independently checked volume receipt; local download is a separate state.**

## Experiment boundaries

收集数据不等于修改现有 benchmark。当前 WM-811K train/validation/test 和 final
holdout 保持原协议；新数据不会自动进入训练。新项目先固定按 wafer/lot/sample
分组的 split，再做裁剪、增强或生成 caption。缺少这些分组时，明确报告限制。

Carinthia 与 Carinthia-S 属于同一个图像家族；MixedWM38 含 GAN 扩增，不能全部
称为真实独立样本。去重至少包含 source ID、原始字节 hash、像素 hash 和近重复
检查。这里公开展示的样本只能用于开发/演示，不可再充当未见过的最终测试集。
不要把图像外观分类或 VLM 生成的解释当作已验证的制造根因。

## Most useful next experiments

1. **SEM classification + segmentation:** Carinthia-S，比较 DINOv3 特征分类与
   mask 定位；按类报告 Dice/IoU、macro-F1 和 rare-class recall。类 5 仅 4 张，
   不适合照搬均匀 80/10/10 划分或声称可靠的泛化结果。
2. **Mixed-pattern recognition:** MixedWM38，使用八维多标签输出；先解决数据许可、
   标签修正版和真实/生成样本分组，再制定实验。
3. **Process imaging:** PL 数据可做配对条件变化展示与信号定量；EBSD 可做表征和
  取向分析。这两类数据不能直接沿用八类 wafer-map 标签。

Dataset licenses are separate from this repository's MIT code license. See each
card and [sample attribution](samples/carinthia-s/ATTRIBUTION.md).
