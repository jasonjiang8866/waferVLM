# DINOv3 third dispatch: pre-allocation diagnosis

The explicitly authorized third dispatch [43609129272](https://github.com/jasonjiang8866/waferVLM/actions/runs/43609129272)
used source `fc7975cef669acf12f1470f0e2288dd249f1db46` and image
`ghcr.io/jasonjiang8866/wafervlm@sha256:f4fdd981a052a41637233390fc2e161d707fe6061cc163fd9a8abb59c073077c`.
Official CPU weight loading, 103 lifecycle tests and 15 model/protocol tests passed.
The current-image/catalog/inventory preflight failed with `ValueError` at
15:15:50 UTC. The marker and paid step were skipped. The old controller suppressed
the guard reason, so neither capacity, credentials nor price can be identified
as the cause from that error alone.

The downloaded [artifact 14119199431](https://github.com/jasonjiang8866/waferVLM/actions/runs/43609129272/artifacts/14119199431)
has SHA256 `0d8c2c03b9d511ede0deaf5015fe6f3c4ac8c89cb862db02c73ad7fed976401a`.
Its exact entries are `cleanup.json`, `error.json`, `weights/checksums.json`,
and `weights/weights.json`. Cleanup confirms 12 absent-inventory observations,
no persistent volume, and no create intent or allocation exists. This dispatch
therefore added no GPU allocation; the two previous paid attempts remain
conservatively accounted at US$0.10 and 360 seconds in total.

Independent anonymous registry verification also confirmed this image's index,
linux/amd64 manifest, config digest and embedded source SHA. Main advanced only
at 15:19:12 UTC (PR #25), after the failure; that later merge does not explain it.

The read-only diagnosis workflow checks the same weights/cache, immutable image,
inventory, companion, catalog, quote, minimum CUDA capacity and account reserve.
It accepts the image's historical source separately from the current diagnostic
controller source. It never calls the launch/cleanup functions, emits no paid
authorization marker, and uses a provider wrapper that rejects all mutations.
The existing paid workflow retains its exact-current-main and paid-history gates.

Stage records expose fixed guard reasons and sanitized HTTP categories only;
unknown exception strings and provider responses are excluded. A passing
observation is not permission to allocate or a successful training result.
