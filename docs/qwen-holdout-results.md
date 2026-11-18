# Qwen classification on unseen wafer lots

Seven frozen Qwen configurations each classified the same 148 WM-811K maps once on 18 November 2026. The maps span 143 lot groups disjoint from development data. Seven classes have 20 maps each. Near-full has eight. All models used the frozen prompt, greedy decoding, identical image order and a fresh CUDA process. The released adapter had been selected on validation before this study.

## Classification and inference

| Configuration | Correct / 148 | Accuracy | Macro-F1 | Invalid | Mean generation time (ms/map) | Peak allocated GPU memory (GiB) |
|---|---:|---:|---:|---:|---:|---:|
| [original](evidence/qwen-holdout/original/predictions.jsonl) | 20 | 13.51% | 0.030488 | 4 | 483.8 | 1.672 |
| [seed17-step200](evidence/qwen-holdout/seed17-step200/predictions.jsonl) | 103 | 69.59% | 0.705910 | 0 | 547.8 | 1.692 |
| [seed17-step250](evidence/qwen-holdout/seed17-step250/predictions.jsonl) | 104 | 70.27% | 0.711854 | 0 | 539.4 | 1.692 |
| [seed42-step200](evidence/qwen-holdout/seed42-step200/predictions.jsonl) | 96 | 64.86% | 0.650024 | 0 | 549.5 | 1.692 |
| [seed42-step250](evidence/qwen-holdout/seed42-step250/predictions.jsonl) | 90 | 60.81% | 0.602608 | 0 | 558.6 | 1.692 |
| [seed123-step200](evidence/qwen-holdout/seed123-step200/predictions.jsonl) | 103 | 69.59% | 0.703457 | 0 | 567.7 | 1.692 |
| [seed123-step250](evidence/qwen-holdout/seed123-step250/predictions.jsonl) | 101 | 68.24% | 0.688999 | 0 | 537.6 | 1.692 |

Timing measures synchronized generation on a single RTX PRO 6000 Blackwell Server Edition. It excludes preprocessing, model loading and decoding the output tokens to text. See the [per-class precision, recall, F1 and confusion counts for every model](evidence/qwen-holdout/metrics.json).

## Does step 250 improve on step 200?

| Training seed | Accuracy change (pp) | Paired 95% interval (pp) | Macro-F1 change | Paired 95% interval |
|---|---:|---:|---:|---:|
| 17 | +0.68 | [-6.04, +7.38] | +0.005944 | [-0.059960, +0.070015] |
| 42 | -4.05 | [-10.81, +2.72] | -0.047416 | [-0.122879, +0.024935] |
| 123 | -1.35 | [-9.33, +6.76] | -0.014458 | [-0.089681, +0.061886] |

Seed 17 was exploratory. For the two confirmation seeds, mean step-250 minus step-200 macro-F1 was −0.030937 (sample SD 0.023305). Mean accuracy change was −2.70 percentage points (sample SD 1.91 points). Step 250 scored lower for both confirmation seeds, as on validation. Each paired interval spans zero. The sample does not resolve a consistent advantage for either checkpoint across lots.

Intervals use 10,000 paired resamples of the 143 lot groups with bootstrap seed 20261110. All eight labels contribute to macro-F1 in each draw, with undefined class F1 scored as zero. Each model saw each reserved map once. The three repeated passes on the 160 validation maps are [reported separately](vlm-repeat-results.md).

## Interpretation

Every adapter improved on the original model on these held-out WM-811K lots. The original model correctly classified only scratch examples. The released seed-17 checkpoint 250 correctly classified 104/148 maps, with macro-F1 0.711854 compared with 0.773992 on validation. It identified 17/20 donut, 9/20 edge-loc and 8/8 near-full maps. With eight near-full examples, one miss would change recall by 12.5 percentage points. Generalization to other fabs, other datasets, and normal or unknown patterns remains open.

## Evidence and cost

The [GPU run](https://github.com/jasonjiang8866/waferVLM/actions/runs/44307271887) used EUR-IS-2 at US$2.69/GPU-hour. The holdout phase took 647.95 seconds. The conservative interval from allocation intent through verified absence was 17.25 minutes, adding US$0.7760 including the disk estimate. The campaign reached US$11.4231 and 791.22 minutes, within its US$30 and 1,100-minute limits. Both the controller and [independent cleanup](https://github.com/jasonjiang8866/waferVLM/actions/runs/44310707459) recorded three consecutive absence checks.

Raw predictions and endpoint identities are in the [evidence directory](evidence/qwen-holdout). Reanalysis of the downloaded archive reproduced all 1,036 parsed predictions, seven metric sets and all paired intervals under the [frozen procedure](vlm-final-holdout.md). The result ZIP SHA-256 is `3e678e2cb8c14b05dec72ac67a48cdfcfb888233f0bc45cdb5be53429e11d1d7`. The independent cleanup ZIP SHA-256 is `a73c14fdba15d560fdf83d7ff1ef99f3b64f16dfed7c0cf4c519bc3587badd7a`.
