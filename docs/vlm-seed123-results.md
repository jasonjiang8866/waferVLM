# Seed 123 completed and independently verified

Verified 2026-11-15 SGT. [Run 43609894428](https://github.com/jasonjiang8866/waferVLM/actions/runs/43609894428)
completed all 250 updates and passed the controller verifier. The original
396,837,007-byte ZIP was downloaded, SHA256-checked and fully reverified locally
against the launch revision `a583f29ee997b64acd27f3cef3024bb3ba130a0a`.
Step-200/250 full-state checkpoints are accepted; this is not a model-quality result.

| Measurement | Observed value |
|---|---:|
| GPU | RTX 4090; Torch 2.10.0+cu128 |
| Training subprocess | 2,969.572 seconds / 49.493 minutes |
| Intent-to-absence billing bound | 3,403.234 seconds / 56.721 minutes |
| Additional estimated GPU and disk cost | US$0.849796 |
| Cumulative VLM estimate | US$8.132282 / 640.704 minutes |
| Remaining US$30 / 1100-minute authorization | US$21.867718 / 459.296 minutes |

The controller and [independent cleanup](https://github.com/jasonjiang8866/waferVLM/actions/runs/43615153978)
each observed the owned temporary Pod absent three times. No persistent volume
was created. Costs are conservative lifecycle estimates, not provider invoices.

The container reported 64 logical/affinity CPUs but a visible cgroup-v1 quota
of 17.85 CPUs. During the worker interval, 5,622 of 30,343 scheduling periods
(18.53%) were throttled. This confirms quota throttling, not its contribution to
GPU idle time; accumulated throttled time must not be interpreted as a wall-time
percentage. Hidden ancestor limits remain unverified. No thread or kernel setting
was changed, and the faster runtime than seed 42 is not a controlled speedup.

The [pinned receipt](../runpod/vlm-completed.json) retains exact run/artifact hashes,
endpoint identities, cumulative budget and both cleanup records. [Machine-readable evidence](evidence/seed123-43609894428.json)
contains the observations. Test and final holdout inference were not performed.
The next phase freezes all six endpoints before 18 fresh-process validation passes.
