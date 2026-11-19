# DINOv3 confirmation: allocation recovery

The first confirmation dispatch, [44526240351](https://github.com/jasonjiang8866/waferVLM/actions/runs/44526240351), used controller `9b63d4e5d5122930ce8ce790868b936c5fb2e3df` and passed the DINO-only ledger, pinned-image, input-artifact and live-price preflight. The selected RTX 4090 quote was US$0.89/hour; CUDA was filtered with a minimum of 12.8.

The remote creation request failed with `RemoteHTTPError`. No Pod ID, allocation receipt, CUDA evidence or model result was returned. The original handler discarded the HTTP status and categories, so the cause cannot be attributed to GPU capacity. This attempt completed **0/18 validation inferences and 0/888 holdout predictions**.

Both the originating workflow and [independent cleanup 44526952153](https://github.com/jasonjiang8866/waferVLM/actions/runs/44526952153) verified absence in 12 consecutive inventory checks. Their immutable artifact IDs, SHA-256 hashes and exact cleanup records are committed in `runpod/dinov3-study-ledger.json`.

Accounting conservatively includes the entire 224.654337-second interval from creation intent through verified absence, even though allocation was not confirmed. The added estimate is US$0.0560967, taking the DINO cumulative estimate to **US$0.7042184 and 33.1775 minutes** against the existing US$10 / 500-minute authorization. These estimates are not provider invoices.

The controller now retains a fixed operation stage, HTTP status and fixed diagnostic categories, without remote response text, signed URLs or credentials. The next dispatch can explicitly use the already-approved RTX 5090 fallback at no more than US$1.39/hour. It is a new, ledger-reconciled attempt, with one create request and independent cleanup; it does not rerun the failed dispatch or change any training, selection or holdout protocol.

## RTX 5090 capacity disappeared before creation

[Dispatch 44528940006](https://github.com/jasonjiang8866/waferVLM/actions/runs/44528940006), from `e61de66d4d5ea4c6e8ac462f6dccb95d8914c59d`, initially observed RTX 5090 capacity in EUR-IS-1 at US$1.39/hour. Its second catalog query, immediately before creating any resource, returned the same price but **no eligible regions**. The saved error identifies the `catalog` stage; there is no `provision.json`, create request, Pod ID or GPU inference output. Added GPU cost and runtime are zero. This does not explain the earlier RTX 4090 HTTP error.

The controller now makes at most four read-only catalog observations with three 15-second pauses when no approved quote/capacity is available. Automatic selection retains 4090-first order and the same CUDA minimum, global eligible regions and price limits. All observations are saved. A create request is still sent at most once; remote creation errors are never automatically retried. The frozen model, data and inference protocols are unchanged.

Both this dispatch and [independent cleanup 44529384082](https://github.com/jasonjiang8866/waferVLM/actions/runs/44529384082) verified 12 consecutive inventory absences. Their exact receipts and artifact hashes are in the DINO ledger. Cumulative estimated usage remains US$0.7042184 / 33.1775 minutes; completed new inference remains 0/18 validation and 0/888 holdout.

## Allocated 4090: fingerprint contract failure

[Dispatch 44530840974](https://github.com/jasonjiang8866/waferVLM/actions/runs/44530840974) allocated Pod `p482h2x23vd2u8`, an RTX 4090 in EU-RO-1 at US$0.89/hour. The worker became reachable and returned `rejected`, `ValueError`, `failed_phase=verifying_embedded_weights` before CUDA or training evidence was produced.

The new controller incorrectly passed the old receipt's `weights` object, which is the **output** of `verify_files`, as its **input**. `verify_files(expected)` requires the original CPU `report()` fingerprint, including the parameter count and preflight status. These two schemas differ even when their model and file hashes agree. A CPU regression reproduces the exact early rejection with the real published receipt, then verifies that the canonical fingerprint reaches the file-verification stage. The fix checks the published model/revision/file hashes and sends the canonical fingerprint. It does not weaken or change the image's weight verifier.

The controller also received an `HTTPError` while retrieving the 588-byte rejected-worker archive. Its request omitted the explicit User-Agent used by the existing working archive client; restore that header and test real HTTP authentication/header, size and hash behavior. The old handler did not retain the HTTP status, so the exact remote rejection is unknown. Future diagnostics preserve HTTP status and safe worker frame locations without exception text or URLs.

The originating workflow verified three consecutive absences after deletion; [independent cleanup 44532010441](https://github.com/jasonjiang8866/waferVLM/actions/runs/44532010441) verified twelve. The billed-time bound is 364.550146 seconds / estimated US$0.0910290. Cumulative DINO usage is now **US$0.7952475 / 39.2533 minutes**, leaving US$9.2047525 / 460.7467 minutes under the original authorization. No training or inference completed: validation **0/18**, holdout **0/888**.

## Fourth attempt: provider resource rejection

[Dispatch 44533432837](https://github.com/jasonjiang8866/waferVLM/actions/runs/44533432837), from `5c75effc853ce3a0afde67a8cca394dfcf44466e`, passed the corrected fingerprint preflight. Its catalog initially showed no 4090 or 5090 regions, then offered 4090 in EUR-IS-2 and US-IL-1 at US$0.89/hour. The single create request returned HTTP 400 with the fixed `resource` category. No Pod ID or worker/CUDA result was returned. This does not demonstrate a new worker failure or establish the provider's exact resource shortage.

The originating cleanup verified 12 consecutive absences. The conservative intent-to-absence bound is 227.468095 seconds / US$0.0567993, including the interval even though allocation was not confirmed. Cumulative DINO usage is **US$0.8520468 / 43.0444 minutes**, leaving **US$9.1479532 / 456.9556 minutes**. Independent cleanup and immutable receipt hashes are recorded in the ledger. New completion remains **0/18 validation passes and 0/888 holdout predictions**. The next attempt uses the already-approved 5090 fallback with the same protocol, caps and one-create rule.

## Fifth attempt: sustained 5090 catalog shortage

[Dispatch 44535674111](https://github.com/jasonjiang8866/waferVLM/actions/runs/44535674111), from `59f9ca1a5d6a7d5cc6b183820bbeec4173b9eff1`, passed its initial 5090 quote check. All four observations during the second catalog check returned US$1.39/hour but zero eligible regions. The run stopped before creating a Pod; its result artifact contains no provision record. Added GPU cost/runtime are zero, and cumulative usage remains US$0.8520468 / 43.0444 minutes. Both absence receipts are recorded in the ledger.

The read-only catalog window is now bounded to 20 observations with 19 pauses of 15 seconds (285 seconds total sleep), instead of giving up after 45 seconds of pauses. Automatic mode continues checking 4090 then 5090 in each observation, globally at CUDA >=12.8 and the original price caps. No creation retries are introduced. The 45-minute work limit, 10-minute cleanup reserve, 80-minute Actions timeout and US$10 / 500-minute cumulative allowance remain unchanged. All training, model selection and holdout rules remain frozen. The regression checks the bound, GET-only behavior and unchanged price rejection.

## Seventh dispatch: holdout creation HTTP 500

After [successful validation run 44537796019](https://github.com/jasonjiang8866/waferVLM/actions/runs/44537796019) accepted all **18/18 validation passes**, the six weight hashes and inference/statistical plan were committed and merged at `aa33e8b949e396892764a1a7503aaadc09a95fc4` (2026-11-19 22:20:36 SGT). [Holdout dispatch 44543065761](https://github.com/jasonjiang8866/waferVLM/actions/runs/44543065761) started afterwards and used exactly that commit.

Its live quote offered RTX 4090 at US$0.89/hour in EU-RO-1, EUR-IS-2 and US-IL-1. The one creation request returned **HTTP 500**, fixed category `unclassified`, at `create_pod`. No Pod ID, allocation receipt, worker archive, CUDA execution or prediction file was returned. The record does not establish the provider's underlying cause. Holdout completion remains **0/888**, separately from the accepted validation **18/18**.

The originating workflow and [independent cleanup 44544073894](https://github.com/jasonjiang8866/waferVLM/actions/runs/44544073894) each verified twelve consecutive absences. Immutable artifact IDs, ZIP SHA-256 and exact records are in the DINO ledger. The conservative intent-to-absence bound adds **227.379308 seconds / US$0.0567772**, even though no allocation was confirmed. Cumulative DINO usage is **US$1.1195446 / 60.8989 minutes**, leaving **US$8.8804554 / 439.1011 minutes** under the same authorization.

The next dispatch may use the approved RTX 5090 fallback, globally at CUDA >=12.8 and <=US$1.39/hour. This is a new ledger-reconciled attempt after independently confirmed absence, not a retry of an ambiguous create operation. All six frozen exports, manifests, model-selection rules and statistical settings remain unchanged; no holdout prediction has been read or used for selection.
