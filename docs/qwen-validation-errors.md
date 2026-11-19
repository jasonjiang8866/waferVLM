# Where the six Qwen checkpoints agree

Of 160 development-validation wafer maps, all six checkpoints classified 84 correctly. On 60 maps at least one checkpoint was correct and at least one was wrong. All six missed the remaining 16. Five of those 16 received the same wrong label from every checkpoint.

The analysis uses the prespecified first validation pass at steps 200 and 250 for seeds 17, 42 and 123. Each row represents one map. The six outputs per row are paired observations. All six first passes returned valid labels for every map. The 148-map final study is reported [separately](qwen-holdout-results.md).

| True class | All six correct | Some correct | All wrong, same prediction | All wrong, different predictions |
|---|---:|---:|---:|---:|
| Center | 13 | 5 | 1 | 1 |
| Donut | 10 | 8 | 2 | 0 |
| Edge-loc | 4 | 14 | 0 | 2 |
| Edge-ring | 16 | 3 | 0 | 1 |
| Loc | 3 | 10 | 1 | 6 |
| Near-full | 15 | 5 | 0 | 0 |
| Random | 14 | 5 | 1 | 0 |
| Scratch | 9 | 10 | 0 | 1 |
| **Total** | **84** | **60** | **5** | **11** |

Seven of the 16 shared misses are labeled `loc`. Six of these seven drew different wrong predictions across checkpoints. In contrast, 14 of the 20 `edge-loc` maps were recognized by some checkpoints and missed by others. The class counts describe these 20 examples per class. They do not estimate fab-wide defect frequencies.

## Changes from step 200 to step 250

| Seed | Wrong → correct | Correct → wrong | Wrong → different wrong | Same prediction | Net change in correct maps |
|---|---:|---:|---:|---:|---:|
| 17 | 12 | 7 | 6 | 135 | +5 |
| 42 | 7 | 19 | 12 | 122 | −12 |
| 123 | 9 | 17 | 6 | 128 | −8 |

Each row totals 160 before the net-change column. Both confirmation seeds gained some correct classifications at step 250 and lost more than they gained. A net score alone hides these changes. Fresh-process repeats also showed some seed-123 label variability, so these are observed output changes between checkpoints under the first-pass rule. See the [repeatability analysis](vlm-repeat-results.md).

## Cases with one unanimous wrong prediction

| Validation sample ID | Dataset label | Prediction from all six |
|---|---|---|
| `0ae15163863765ba0c40815d` | donut | center |
| `0dba1dc3f2c5cf63eda306ba` | loc | center |
| `22c61f53f1f5786e9fb394b8` | random | near-full |
| `3d0f3c97b77fd687704cb68a` | center | donut |
| `ddc10f023d1480284b53af51` | donut | loc |

A focused next diagnostic is to inspect the seven shared `loc` misses alongside the three unanimously correct `loc` maps using the development images and source annotations. Record spatial extent, position and any competing visible pattern before inspecting predictions. This can separate candidate failure modes for a future experiment. Agreement alone does not establish a labeling error.

## Reproduce

From the repository root, run the following with Python 3.11 or later. The optional CSV aligns all six predictions with the sample ID and dataset label.

```sh
python analysis/qwen_validation_errors.py --csv validation-by-map.csv
```

The six [first-pass prediction files](evidence/qwen-validation-first-pass) are byte-for-byte copies from the [validation GPU run](https://github.com/jasonjiang8866/waferVLM/actions/runs/44124342191). The script verifies their hashes against the [accepted repeat receipt](evidence/vlm-repeat-complete.json) and verifies the shared [validation manifest](showcase/dinov3-seed17/validation.jsonl) against the frozen Qwen configuration before joining by ID, image hash and label.
