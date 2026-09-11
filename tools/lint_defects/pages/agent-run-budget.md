# Agent Run Budget

**Summary**: The total time one run is allowed before it stops itself and leaves the remaining work for the next run.
**Sources**: source.md
**Last updated**: 2026-07-19

---

## Why a run stops itself

A run that overruns holds the only serving slot on the box. Everything else
queues behind it, so a late run is not merely late — it blocks unrelated work
(source: source.md).

## Mechanism

The budget is started once, at the top of the run. Each unit of work checks the
remaining time before it begins. A unit that cannot fit is not started, and the
run exits non-zero with its unfinished work still queued.

## Choosing the number

Take the slowest observed run, add half again, and round up. A budget tighter
than the real distribution turns an ordinary slow day into a failed job.

## Related pages

- [[wall-clock-ceiling]]
- [[log-rotation]]
