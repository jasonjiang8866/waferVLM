# Seed 42 completed: verifier-only recovery

Verified 2026-11-14 SGT. **Training/execution accepted; model quality not yet evaluated.**

[Run 43562568201](https://github.com/jasonjiang8866/waferVLM/actions/runs/43562568201)
completed fresh training through step 250 on RTX 4090 in EUR-IS-2. The original
Actions conclusion remains **failure**: after retrieving the completed archive,
the controller rejected a tokenizer JSON exceeding its generic per-JSON limit.
No GPU rerun was needed to recover this result.

## Root cause and corrected verification

The compressed worker archive is 396,728,629 bytes; its 113 members total
493,785,165 unpacked bytes, well below the existing 4 GiB aggregate limit.
Each of the four canonical `training/step*/adapter/tokenizer.json` files is
19,989,343 bytes. The old generic JSON limit was 8,000,000 bytes.

Only these four tokenizer paths now have a 32,000,000-byte ceiling. Other JSON
retains its 8 MB limit. Total size/member limits, canonical-path enforcement,
duplicate/link rejection, full checksums, source/config/model identities,
gradient/LR checks and optimizer/scheduler/RNG parent chains remain enforced.
The real archive passes all of those gates with the corrected verifier.

The receipt explicitly retains the failed original outcome and pins the original
ZIP, worker archive, authorization marker and independent cleanup. CI re-downloads
the original ZIP and independently re-verifies it with CPU only. A changed run,
rerun, damaged archive, failed worker result or missing cleanup still blocks the
next allocation; the recovery does not simply relabel a workflow as green.

## Observed result

| Item | Verified result |
|---|---|
| Seed / stages | 42; 0 → 100 → 150 → 200 → 250 |
| Runtime | Torch 2.10.0+cu128; CUDA 12.8; RTX 4090 |
| Training subprocess | 5,035.811 seconds, exit 0 |
| Lifecycle billing bound | 5,524.365 seconds / 92.073 minutes |
| Additional estimated cost | US$1.379447, including temporary disk |
| Cumulative VLM usage | US$7.282486 / 583.984 minutes |
| Remaining authorization | Approximately US$22.717514 / 516.016 minutes |
| Checkpoints | Full step-200 and step-250 states independently accepted |
| Quality / holdout | No new validation scores; no test or holdout inference |

The cost is a conservative intent-to-absence estimate, not a provider invoice.
The source is `11d2e5f088b350dd9391864dbe2869417fff5af7`; launch controller is
`537e59bd34e1415bf5091b90bb270fcc3ac08f22`. The image is
`ghcr.io/jasonjiang8866/wafervlm@sha256:a6d31b4f12c2c3ee64c59d08b1e67c89e9bbf9b46cbdb4179417f1d1efa3fa79`.

## Artifact and cleanup evidence

- [Original result artifact](https://github.com/jasonjiang8866/waferVLM/actions/runs/43562568201/artifacts/14104346663),
  ZIP SHA256 `fabbcef0155c8c5f379da2d353654717f8a39fb4f699751a6b733b112a6d7c1e`.
- Inner archive SHA256 `3296b25380b1e2f553cc1d29908eb347d1f4bfdf6a7c6d51d1607af4b3ddaf1e`.
- Original cleanup: `verified_absent`, three consecutive inventory checks,
  creation confirmed, no persistent volume created.
- [Independent cleanup 43570004971](https://github.com/jasonjiang8866/waferVLM/actions/runs/43570004971):
  another three absence checks for `wafer-20261107-usd20-vlm42-43562568201`.
  Cleanup ZIP SHA256 `fc27bae978bf4f1256c1f99992007a29b5b5925034bec65fea3f61e28be67edd`.
- [Machine-readable training/evidence summary](evidence/seed42-43562568201.json).
- [Pinned cumulative receipt](../runpod/vlm-completed.json).

Seed 123 follows the unchanged recipe after fresh admission checks. Its worker
also records visible CPU quota/affinity and throttling counters without changing
the training process. Kernel/thread optimizations remain a separate experiment;
the current result makes no efficiency or accuracy improvement claim.
