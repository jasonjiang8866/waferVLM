# Qwen misses more loc wafers as the defective area grows

All six checkpoints have lower `loc` recall in the second training-area quartile than in the first. All six miss the four validation `loc` wafers above the training median. In that same upper half they also make 16 false `loc` calls on eight wafers from other classes. A larger yellow region alone therefore cannot explain which images receive a `loc` prediction.

## Counts by area

We fixed three nearest-rank cut points using all 790 training maps, before scoring validation. Area is yellow pixels divided by yellow plus blue pixels in the original 256 × 256 rendering. Black padding is excluded. Comparisons use exact pixel ratios, with ties assigned to the lower bin. Percentages in the table headings are rounded for display.

| Train-defined yellow area | Training loc / all maps | Validation loc / other maps | True loc recalled | Other maps called loc | Other maps correctly classified |
| --- | ---: | ---: | ---: | ---: | ---: |
| Q1 ≤12.35% | 41 / 198 | 7 / 31 | 31/42 (73.8%) | 18/186 (9.7%) | 151/186 (81.2%) |
| Q2 >12.35–20.51% | 37 / 197 | 9 / 25 | 14/54 (25.9%) | 9/150 (6.0%) | 122/150 (81.3%) |
| Q3 >20.51–38.44% | 20 / 198 | 2 / 41 | 0/12 | 15/246 (6.1%) | 185/246 (75.2%) |
| Q4 >38.44% | 2 / 197 | 2 / 43 | 0/12 | 1/258 (0.4%) | 206/258 (79.8%) |

The first two count columns describe distinct wafers. The last three count first-pass predictions from the same six seed/step combinations on each wafer. Overall there are 160 validation maps, 960 predictions, 45 true `loc` calls, 75 missed `loc` calls and 43 false `loc` calls. The last two bins have just two true `loc` maps each. Counts from different checkpoints on a single map are correlated. Class mix also changes across bins. For example, all 20 validation `near-full` maps fall in Q4 and 15 of the 20 `scratch` maps fall in Q1.

### True loc hits by checkpoint

Each row evaluates the same maps. Column headings give the number of true `loc` maps available to that checkpoint.

| Training seed / step | Q1 (7) | Q2 (9) | Q3 (2) | Q4 (2) |
| --- | ---: | ---: | ---: | ---: |
| 17 / 200 | 6 | 2 | 0 | 0 |
| 17 / 250 | 6 | 3 | 0 | 0 |
| 42 / 200 | 5 | 1 | 0 | 0 |
| 42 / 250 | 3 | 0 | 0 | 0 |
| 123 / 200 | 6 | 5 | 0 | 0 |
| 123 / 250 | 5 | 3 | 0 | 0 |

## Where the errors go

Each entry below counts checkpoint predictions. The two directions use different reference classes. For example, in Q3 six real `loc` predictions went to `edge-loc`, while two real `edge-loc` predictions went to `loc`.

| Bin | True loc predicted as another class | Classes incorrectly predicted as loc |
| --- | --- | --- |
| Q1 | scratch 7, center 3, edge-loc 1 | scratch 9, edge-loc 5, center 4 |
| Q2 | center 17, scratch 12, edge-loc 7, donut 3, random 1 | edge-loc 7, scratch 2 |
| Q3 | edge-loc 6, random 5, donut 1 | donut 10, center 3, edge-loc 2 |
| Q4 | near-full 5, random 4, donut 2, center 1 | near-full 1 |

## Implication for the next training experiment

Prioritize Q2 for a targeted comparison. It has nine validation `loc` maps, 37 training `loc` maps and substantial confusion with `center`, `scratch` and `edge-loc`. A candidate ablation pairs training `loc` maps with examples of those classes at similar yellow area and compares that sampling policy with the current sampler at the same number of optimizer steps. Track `loc` recall, false `loc` calls by source class and eight-class macro-F1 on all 160 validation maps. The sparse upper bins can be reported individually. Only two Q4 training `loc` maps are available. These exploratory results do not justify a decision threshold or a claim about unseen lots.

## Reproduce

The [image review](qwen-loc-morphology.md) motivated this analysis. The script checks the training and validation manifest hashes, all 950 input image hashes and the six archived prediction files. Scoring uses the frozen first validation pass of each checkpoint. The 148-map unseen-lot evaluation is outside the analysis. From the repository root, run the following with the [prepared public dataset](../README.md).

```sh
uv run --frozen python analysis/qwen_area_bins.py data/wm811k-lot17 /tmp/qwen-area
diff -ru docs/evidence/qwen-validation-area /tmp/qwen-area
```

The [training fit](evidence/qwen-validation-area/fit.json) contains exact cut points and class support, the [map table](evidence/qwen-validation-area/validation.csv) identifies each validation input, and the [checkpoint table](evidence/qwen-validation-area/checkpoints.csv) contains denominators and both directions of confusion for all 24 bin/checkpoint combinations.
