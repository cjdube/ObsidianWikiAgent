# GPU Memory Headroom

**Summary**: Leaving enough unified memory free that the weights and the key-value cache both stay resident.
**Sources**: source.md
**Last updated**: 2026-08-27

---

## Two consumers, not one

Weights are a fixed cost. The key-value cache is not: it grows with the
configured context length, and it is the one that pushes a model off the
accelerator (source: source.md).

## The symptom

Nothing errors. Part of the model is served from ordinary memory instead, and
throughput drops several-fold. The only reliable tell is checking which
processor the loaded model reports.

## Practice

Measure after any context-length change, not after any weight change. Raising
the window is the edit that silently costs the throughput.

## Related pages

- [[context-window-limits]]
- [[model-serving-runtimes]]
