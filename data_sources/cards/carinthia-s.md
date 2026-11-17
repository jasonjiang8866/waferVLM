# Carinthia-S — SEM classification and segmentation

- Primary source: https://zenodo.org/records/16895427 (v1, 2025-08-20).
- Creators: Corinna Kofler (KAI), Vahidin Hasić (University of Sarajevo).
- License: **CC-BY-4.0**, confirmed through [provider metadata](../metadata/zenodo-16895427.json).
- 4,591 SEM images with paired masks from one production layer of unstructured
  semiconductor wafers. Numeric labels 1–6 are preserved; no invented defect names.
- ZIP: **139,225,942 bytes**, publisher MD5 `3a41242574635cf1302fab15accb7730`;
  SHA256 `e7cc8507657611f89db33345d50fc7951e8784045684888dc997ac5298b7a3a2`.
  Measured expanded payload **153,064,018 bytes**, 9,183 non-directory members.
  The separate source description is 2,104,107 bytes.
- Archive CSV is semicolon-delimited: `image_path;mask_path;filename;label`.
  Measured class counts: **1:55, 2:8, 3:4,008, 4:289, 5:4, 6:227**.
  Source documentation describes class 6 as no visible defect / possible SEM
  misalignment; it should not be relabeled as guaranteed defect-free wafer material.
- Full decode audit: all 4,591 pairs are 480 × 480. Masks include **4,587 L,
  2 RGB and 2 RGBA** files; some grayscale masks contain intermediate values,
  not only 0/255. A future segmentation loader must explicitly define conversion
  and thresholding. The collection preserves original bytes. See the
  [audit receipt](../receipts/carinthia-s-image-audit.json).
- **Overlap:** original Carinthia image family plus masks; never split the two
  releases across training and evaluation as independent datasets.
- Local full download and preview extraction verified. [12 original image/mask
  pairs](../samples/carinthia-s/README.md) are in Git. Global Volume: **not imported**.
- Good projects: defect retrieval, rare-class analysis, classification + mask
  localization. Limits: severe imbalance, single layer, center/border/imaging
  artifacts; no reliable factory-wide or root-cause claims from these labels.

Citation: Kofler, C., & Hasić, V. (2025). *Carinthia-S dataset*. Zenodo.
https://doi.org/10.5281/zenodo.16895427
