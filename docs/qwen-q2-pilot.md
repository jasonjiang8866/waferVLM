# Matched-area Qwen training pilot

Does seeing a localized defect and a similarly dense competing pattern in the same optimizer update help the model distinguish them? The [validation-area study](qwen-validation-area.md) identified nine Q2 validation loc maps. The pilot uses only training images for matching. It is a single-seed development experiment.

| Setting | Control and matched arms |
| --- | --- |
| Start | Fresh pinned Qwen3.5-0.8B, seed 17, identical language-LoRA initialization |
| Training exposure | All 790 lot17 training maps once per arm. Microbatch 1, accumulation 8, 99 AdamW updates including the last six maps |
| Optimization | Rank 8, constant LR 0.0002, BF16, frozen vision, unchanged labels and prompt |
| Evaluation | One greedy pass on the same 160 validation images after each arm, 16-token cap |
| Primary comparison | Paired changes on the nine Q2 loc maps, with all 20 loc maps and the other 140 validation maps as context |
| Trade-offs | Overall loc recall, false loc counts by true class, eight-class macro-F1, accuracy and invalid responses |

The control uses the default Trainer sampler. Its observed order determines the treatment order. Both arms verify every training ID occurs exactly once, record its position and compare the initial trainable-weight hashes. The matched arm places each pair in adjacent microbatches in its own optimizer update. All 716 unpaired maps retain their relative order. Moving pairs also changes which other maps share each update, so the comparison estimates the effect of this ordering policy as a whole.

All 37 Q2 training loc maps receive one distinct peer. In ascending loc-area order, assign the peer classes cyclically and pick the remaining same-Q2 candidate with the smallest absolute area gap, breaking ties by ID. The [frozen pairs](evidence/qwen-validation-area/training-pairs.json) include 13 center, 12 scratch and 12 edge-loc peers. Median area difference is 0.19 percentage points and maximum is 3.18. Q2 boundaries are inherited unchanged from the 790-map fit.

Report all nine paired predictions and both directions of change even if results worsen. Treat an increase in Q2 hits with no increase in overall false loc calls and no decrease in macro-F1 as a reason to repeat with additional seeds. A single Q2 map moves recall by 11.1 percentage points. No reserved-lot score is used for this decision.

## Run and verify

With the locked training dependencies installed, run from the repository root:

```sh
uv run --frozen python analysis/qwen_q2_pilot.py pairs data/wm811k-lot17 /tmp/q2-pairs.json
diff docs/evidence/qwen-validation-area/training-pairs.json /tmp/q2-pairs.json
uv run --frozen --extra train python analysis/qwen_q2_pilot.py run data/wm811k-lot17 outputs/q2
uv run --frozen python analysis/qwen_q2_pilot.py score data/wm811k-lot17 outputs/q2
```

The [manual GPU workflow](../.github/workflows/qwen-q2.yml) checks global inventory and live pricing for a single Pod shared by both arms. It uses the pinned Qwen runtime and a checksum-verified bundle of source files from the reviewed checkout. The Pod stages this bundle on its ephemeral disk before starting the HTTP worker. Training reads the same frozen data and model revisions in both arms.

The [first allocation](https://github.com/jasonjiang8866/waferVLM/actions/runs/44990448161) received an RTX 4090 in US-NC-1 at US$0.89/hour. The worker endpoint returned HTTP 404 until startup timed out. The allocation consumed an estimated US$0.31313 and 20.90 minutes through verified absence, including ephemeral disk. This leaves 69.10 minutes of work and ten minutes of cleanup in the pilot budget. Its original startup command raises `FileNotFoundError` on an empty filesystem without `/workspace`. The remote diagnostic records error categories only. A replacement requires the original startup-timeout artifact and both deletion receipts, verified by SHA-256, plus exactly one new intent and the original intent. The controller subtracts the first allocation from the **same US$5 and 100 minutes**, including startup and validation, and reserves ten minutes for cleanup. It rechecks inventory and account headroom immediately before allocating. The default dispatch performs read-only preflight.

After both arms finish, the controller retrieves the two adapters, observed epoch orders, gradient diagnostics and all 320 raw validation predictions. It recalculates the paired comparison from those predictions. A separate [cleanup workflow](../.github/workflows/qwen-q2-cleanup.yml) checks the originating Pod independently. Cost uses an allocation-intent-to-absence estimate including ephemeral disk.

### Dispatch receipts

From a checkout containing the receipt workflow, use an authenticated GitHub CLI, jq and Python 3. To test the GitHub dispatch path on a development branch, pass its pushed branch name and reviewed commit. Only the receipt job runs there. On `main` the default also checks live inventory and the reconciled budget.

```bash
bash scripts/qwen_q2_dispatch.sh submit /tmp/qwen-q2-probe "$(git branch --show-current)" "$(git rev-parse HEAD)"
bash scripts/qwen_q2_dispatch.sh find /tmp/qwen-q2-probe
```

The first command records the request locally before posting it once. GitHub API version `2026-03-10` returns the run ID and URL in the response. If the response is lost, reuse the same directory with `find`. It performs GET requests and matches the workflow, branch, revision and request ID. The ID identifies a request for lookup. Submitting from a new directory creates a separate request.

For an authorized paid launch, use a new receipt directory, the `main` ref, its reviewed SHA and the explicit `--paid` switch. A run URL confirms dispatch. GPU allocation and both arms' predictions require their respective controller artifacts. The Actions web form also accepts an optional request ID.

[GitHub workflow dispatch API](https://docs.github.com/en/rest/actions/workflows?apiVersion=2026-03-10#create-a-workflow-dispatch-event)
