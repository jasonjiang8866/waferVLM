# VLM startup diagnostics

The globally scheduled final-two run [44108007848](https://github.com/jasonjiang8866/waferVLM/actions/runs/44108007848)
reported a RUNNING Pod without a worker response. RUNNING alone does not establish
that the image started, the Python service became ready, or GPU inference ran.

The controller records a bounded snapshot after five minutes without a worker
response and another at the existing twenty-minute startup timeout. Each snapshot
contains the latest observed runtime counters, the last proxy HTTP status or
transport-error classification, and category counts from container/system logs.

Log transport follows the official [RunPod REST v2 SSE implementation](https://github.com/runpod/runpodctl/blob/72afce1305ad3a25217aafdb9022a16e9b569661/internal/api/logs.go).
It makes one GET with a tail of 80 lines per source. Reads have byte/frame/socket
limits and run in a child process with a twelve-second outer deadline. The child
inherits the existing Actions secret; no credential is added to the Pod or command
arguments. Provider text, command lines, environment values, addresses and raw
errors are never saved or printed. Unknown frames are counted without retaining
their contents.

Categories are clues, not a root-cause verdict. A silent stream can indicate no
matching logs or a transport problem. Missing runtime telemetry is recorded as
unreported, not proof that the container is stopped. Diagnostics do not alter
creation retries, GPU/region selection, the twenty-minute startup limit, workload
budget, cleanup ownership checks or the frozen evaluation inputs.

The allocated startup failure is now separately reconciled using its pinned result
and authorization artifacts, terminal run identities and independent cleanup. The
full $0.943195406 intent-to-absence estimate is retained, bringing cumulative cost
to $10.471823366 and runtime to 45736.061768 seconds. This is not a capacity
rejection and contributes no new inference passes.

The approved final-two continuation uses one create and a 60-minute work window
plus ten minutes for cleanup. At the highest allowed $2.69/h rate, the cumulative
compute/disk estimate is below $13.63 and cumulative time below 833 minutes, within
the existing $30 / 1100-minute campaign. The failed PRO 6000 / US-PA-1 pair joins
the placement history so fresh eligible regions are preferred. Unknown later
attempts still block another allocation; the flag does not erase consumed markers.

The worker log category summary is also printed to the live Actions log after
five minutes, permitting diagnosis before final artifact upload. No raw log text
or credentials are printed. The pinned image, 16 existing passes, two missing
outputs and frozen evaluator are unchanged.

Historical artifact GETs retry 502/503/504 and transport failures at most three
times, with two- and five-second waits. A fresh signed URL is obtained for each
attempt, without forwarding the GitHub token to storage. Authorization failures,
unexpected destinations and SHA/size mismatches remain fatal. This addresses the
observed transient artifact-storage 503 before any Pod is purchased.

The independent data collection scripts added by PR #47 are excluded by exact
path from the VLM image-workload diff, as is the existing DINO proxy diagnostic.
All existing evaluator scripts, model source, configs and dependency pins remain
checked; the collection is not used as new model input by this continuation.
