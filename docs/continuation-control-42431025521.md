# Constant learning-rate continuation control

Run [42431025521](https://github.com/jasonjiang8866/waferVLM/actions/runs/42431025521)
completed on 2026-11-08 at 10:42 SGT. It restored checkpoint-100 from run
42385615853, then executed exactly 50 + 50 optimizer steps at constant LR 2e-4.
All 100 new gradient checks were finite and nonzero. This is a single-seed study.

| Validation (160 images) | Step 100 | Step 150 | Step 200 |
|---|---:|---:|---:|
| Accuracy | 0.64375 | 0.66250 | 0.55000 |
| Macro-F1 | 0.6406758251 | 0.6363060419 | 0.5099391680 |
| Answer-token loss | 0.2705825592 | 0.2466523605 | 0.3405332985 |
| Invalid rate | 0 | 0 | 0 |

Selection was fixed by validation macro-F1 before the selected model's test pass.
Checkpoint-100 won; its repeated test accuracy 0.53125 and macro-F1 0.5216590953
exactly match the earlier evaluation. No test result for checkpoint-200 is claimed.
Segment training losses were 0.2984383 (101–150) and 0.3151583 (151–200).
The degradation supports investigating optimization; it alone does not prove
overfitting or establish the cause of the class-dependent shifts.

The ZIP digest was checked, all nested file hashes and resume chains passed the
66-file archive verifier, and all four evaluation metrics were recomputed from
predictions. Dataset, label order, prompt, model revision and split stayed fixed.

- Artifact: [13600967558](https://github.com/jasonjiang8866/waferVLM/actions/runs/42431025521/artifacts/13600967558)
- ZIP SHA256: `19a73a01065bb12a546ee11474bae761889b76644c57796c49b017411cfe088b`
- Inner archive SHA256: `1475467fefa53decd6d665b9a75f794fefb539fb7afd33ad625f6df4c65dd330`
- Controller: `7ccc9894c3e2b09305371a2b649c60118c295945`
- Image source: `d1111efdbe31f9b866610ad7f1ab4e167a7709ef`
- GPU: PRO 6000 Blackwell MIG 1g.24gb, US-PA-1, observed GPU rate $0.69/hour.
- Temporary Pod `b6a0k19d5jbzas`: three consecutive absence checks; no persistent volumes.
- Independent cleanup: [42439659492](https://github.com/jasonjiang8866/waferVLM/actions/runs/42439659492).
- Cumulative conservative estimate: $3.485102 and 298.905 minutes, including earlier reserves.

## Next arm: isolate learning rate

Start again from the same checkpoint-100 and run to 200 with constant LR 5e-5.
Keep optimizer moments, RNG, data position, seed 17, LoRA r8, accumulation 8,
answer mask, frozen vision tower and data unchanged. Do not add LR decay in this
arm, so a single intervention can be compared against the control above.

The LR override is explicit and applies only after Trainer restores optimizer and
scheduler state. Adam step counts must match the checkpoint, and every actual
optimizer-step LR is recorded and independently checked. The second segment
restores the new checkpoint without another intervention. Evaluate 100/150/200
on validation; select by the same macro-F1 rule before the selected test pass.
Repeated development on this validation/test split is not an unbiased final
production benchmark; a later final assessment needs untouched holdout data.

This arm is exploratory and has no guarantee of improvement. Keep checkpoint-100
if validation macro-F1 does not improve. The current experiment authorization
remains US$20 / 500 cumulative minutes; reserve US$3.50 / 300 minutes for history.
