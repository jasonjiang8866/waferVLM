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
