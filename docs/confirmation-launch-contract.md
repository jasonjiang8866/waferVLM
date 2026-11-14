# Confirmation launch contract and archive acceptance

Prepared 2026-11-12 against main `b5e17163bd1b9a22f9cab4531dfd4d369c52f711`.
PR #19 is merged; main CPU CI passed. The observed image build is PR-only run
42977173731, not a publication. A published confirmation digest remains unavailable.
No GPU was launched by this change.

## Implemented

- `python -m runpod.confirmation_contract proposal.json` validates one **fresh**
  training seed and emits a bounded phase plan. Supply exact source/controller
  SHAs, a published immutable confirmation image, GPU/rate, and a cumulative
  ledger snapshot (`usd`, `seconds`, and `attempts`). Each attempt records
  `run_id`, `marker_artifact_id`, `evidence_sha256`, and `cleanup`.
- Ledger floors include the last continuation: US$5.903039 / 491.911 minutes.
  Each training allocation reserves at most 185 minutes work plus 10 minutes
  cleanup, including 60 GB ephemeral disk in its estimate. All attempts and
  failures consume the existing US$30 / 1,100-minute total.
- Seed 123 requires an accepted, cleanup-verified seed-42 attempt in the supplied
  ledger. An existing attempt for the requested seed blocks fresh training;
  recovery needs its own reviewed allocation, not a free fresh retry.
- `runpod.confirmation_verify.verify_training_archive(archive, dataset, spec)`
  independently checks fresh training output. `dataset` needs only the frozen
  `dataset.json`; no images or holdout are needed. It validates canonical archive
  members, size/count bounds, recipe, completion, four-stage hash/state/LR chains,
  source/image provenance and the 200/250 endpoint hashes. It does not unpickle
  model, optimizer or RNG state. Tensor-level finite-state checks remain separate.
- The archive layout is `results/{worker.json,hardware.json,acquisition.json,training/}`.
  Worker metadata identifies `profiles=["seed-confirmation"]`, seed, source SHA,
  controller SHA and config SHA256. `training/` is the unchanged fresh-seed runner
  output. Absolute checkpoint paths are never trusted during retrieval.

## Limits and remaining integration

The launch contract is **offline proposal validation**, not live admission or
proof of authorization. It cannot prove that a supplied ledger is exhaustive or
that the quoted rate is current. Before any resource mutation, the controller
must independently enumerate Actions authorization markers, download and hash
all corresponding evidence, reconcile every attempt and reject unknown entries.
It must verify image provenance, live price/capacity/account limits and absence
of owned Pods. Source/image correspondence is not proven by matching string formats.

This change does not modify the old paid controller, consume/reconcile an
authorization marker, create a launch trigger, or extend the original authorization
expiry. The next integration must route fresh seed 42 through the existing bounded
transport/cleanup, call this verifier before acceptance, then reconcile the actual
cost and artifacts before seed 123. Recovery and the 18-pass repeat workload need
separate archive acceptance wiring. Do not dispatch the old `gpu.yml`: it still
runs the extension experiment.

Both Dockerfiles now copy `runpod/` because the new archive test imports the
verifier. CI runs the archive tests with lightweight project dependencies; the
model compatibility suite also discovers them. A new image must be built from
this revision before these modules can be imported inside a worker container.

## Verification

Synthetic complete checkpoint packages exercise successful relocation and rejection
of damaged state, resealed wrong seed/source metadata and holdout flags. No GPU or
model/data download is needed. Archive path tests cover traversal, duplicate names,
symlinks, hardlinks, devices, noncanonical paths and size/member limits. Contract
tests cover missing/old digests, stale/duplicate/unreconciled ledgers, phase ordering,
per-GPU rates and cumulative money/time limits. Synthetic tests do not prove
real CUDA execution, provider cleanup, or tensor correctness.

Local validation on 2026-11-12: 14 existing confirmation tests, 6 archive tests
and 48 RunPod tests passed (68 total, including 13 new tests). A broader
confirmation discovery also attempted the existing real-Trainer test, which could
not import Torch in this environment; that test was not rerun successfully here.
No container build is claimed for this change; PR CI must verify both images.
