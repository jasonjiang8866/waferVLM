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

This change does not reconcile or authorize replacement of the current paid run.
Another paid attempt still requires its completed receipts, costs and independent
cleanup to be incorporated into the existing ledger.
