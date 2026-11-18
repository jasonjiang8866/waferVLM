# Qwen evaluation on unseen lots

The final evaluation runs the original Qwen model and all six frozen adapters
once each, in fresh CUDA processes. Each receives the same 148 maps in the
order declared in the reservation. The 143 duplicate-connected lot groups form
the sampling units for inference. See the [frozen protocol](seed-confirmation-plan.md)
and [validation results](vlm-repeat-results.md).

The controller verifies the completed repeat-study archive and both cleanup
records. The worker verifies each adapter and the rendered PNG hashes before
using the fixed prompt and greedy decoding. The controller retrieves raw model
responses and independently parses all 1,036 rows. It reports accuracy, macro-F1,
per-class recall, invalid-response rate, mean generation time and peak allocated
CUDA memory for every endpoint. For each seed it reports step 250 minus step 200
with a paired 95% percentile interval. Each of 10,000 draws samples 143 groups
with replacement using NumPy's default generator seeded with 20261110. Both
checkpoints share the same group multiplicities. All eight classes contribute to
macro-F1, including a zero for an undefined class F1.

Use the **WaferVLM final unseen-lot evaluation** manual workflow on main with
the reviewed main SHA. The campaign ledger starts this stage at an estimated
US$10.610880 and 771.55 minutes. The job allows 60 minutes of work and 10 minutes
of cleanup within the existing US$30 and 1,100 minutes. A subsequent attempt
requires reconciling any previous paid dispatch and any partial holdout exposure.
The workflow saves partial worker evidence when available, and a second workflow
independently checks removal of the originating Pod.

The [first dispatch](https://github.com/jasonjiang8866/waferVLM/actions/runs/44303065030)
received a capacity rejection when creating an RTX 4090 in EU-RO-1. RunPod
returned no Pod ID. The controller recorded eight absence checks, and the
[independent cleanup](https://github.com/jasonjiang8866/waferVLM/actions/runs/44304259531)
recorded three more. No worker started or accessed the holdout. The conservative
cumulative estimate through cleanup is US$10.647178 and 773.97 minutes.

For the next dispatch the controller verifies both archived receipts, carries
the full estimate forward, and excludes the rejected placement for one hour.
It prefers GPU types with more eligible regions among fresh, price-approved
placements, then applies the existing GPU priority to ties.
