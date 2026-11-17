# Carinthia-S preview / 六类 SEM 图像与 mask

12 unmodified images and 12 paired masks. Numeric classes are from the source; these are development previews, not a new evaluation split.

[Attribution and CC-BY-4.0 license](ATTRIBUTION.md) · [Exact source IDs and hashes](manifest.json)

| Class | SEM image | Segmentation mask |
|---|---|---|
| 1 | ![Class 1 SEM](0149a2fcc66647119759fddb9ec44061-image.jpg) | ![Class 1 mask](0149a2fcc66647119759fddb9ec44061-mask.png) |
| 2 | ![Class 2 SEM](20cc9526c58743019f12544ca3f6ffee-image.jpg) | ![Class 2 mask](20cc9526c58743019f12544ca3f6ffee-mask.png) |
| 3 | ![Class 3 SEM](0017618f77bc4dcfb36823422bd3d04c-image.jpg) | ![Class 3 mask](0017618f77bc4dcfb36823422bd3d04c-mask.png) |
| 4 | ![Class 4 SEM](007184cb72454f63b3fccd5024bd9b0f-image.jpg) | ![Class 4 mask](007184cb72454f63b3fccd5024bd9b0f-mask.png) |
| 5 | ![Class 5 SEM](0ef66cd9512e4feaa62e4ec3cb75cdc1-image.jpg) | ![Class 5 mask](0ef66cd9512e4feaa62e4ec3cb75cdc1-mask.png) |
| 6 | ![Class 6 SEM](0037f2474b034ca384a3d3ef1911fe1d-image.jpg) | ![Class 6 mask](0037f2474b034ca384a3d3ef1911fe1d-mask.png) |

The second example for each class is also committed and listed in the manifest. Class 6 means no visible defect in the captured view; it is not a guarantee of a healthy wafer.

Recreate this subset from the verified archive:

```bash
python scripts/prepare_collection_samples.py
```
