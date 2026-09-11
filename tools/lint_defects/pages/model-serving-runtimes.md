# Model Serving Runtimes

**Summary**: The local processes that hold the weights and answer requests, and the knobs each exposes.
**Sources**: source.md
**Last updated**: 2026-08-13

---

## One slot

The default configuration serves one request at a time. Concurrency does not
raise aggregate throughput on a single accelerator; it only changes who waits
(source: source.md).

## Knobs worth naming

Context length, longest reply, idle unload timeout, and the number of parallel
slots. The first two are per-request; the last two are properties of the
process and change what every caller gets.

## Model swapping

Loading a different set of weights evicts the resident ones. A schedule that
alternates between two models pays the load each time.

## Related pages

- [[gpu-memory-headroom]]
- [[quantization-builds]]
