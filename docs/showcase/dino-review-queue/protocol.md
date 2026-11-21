# Review budget protocol

Use the six frozen linear and vision LoRA exports (seeds 17, 42 and 123) from PR #71 and their primary, repeat-1 predictions. Rank each model's maps within each split by ascending maximum of the eight probabilities, with ascending sample ID for ties. Labels are used only after ranking. Batch membership must be known before ranking.

The primary budget is 10%. Report 5% and 20% as sensitivity checks. Review the first floor(N × budget / 100) maps so the workload never exceeds its cap. Report realized counts and compare against uniform random sampling of the same number of maps, whose expected error capture fraction is k/N. For each budget report errors sent / all errors, errors left / all unreviewed maps, and sent / total counts for both true-loc misses and false loc calls. Export the full per-map order.

The development pilot already used 160 validation maps. Apply the same rule separately to the published 148-map holdout as a retrospective analysis. All six models are reported separately. Set the illustration model to seed-17 LoRA on development. Illustrate the first two review-eligible true-loc misses by ascending rank, and all remaining true-loc misses outside the 10% queue by descending confidence. A reviewable error means the image reached a reviewer. Human detection or correction has not been measured.
