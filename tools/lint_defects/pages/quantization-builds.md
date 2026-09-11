# Quantization Builds

**Summary**: Which quantized builds of the served model are available, and the memory each one needs.
**Sources**: source.md
**Last updated**: 2026-05-28

---

## Available builds

There is **one** quantized build: 4-bit (source: source.md). No other precision
is published, so any deployment that cannot fit the 4-bit weights has to serve
the unquantized model or serve nothing.

## Memory

The 4-bit build needs roughly a third of the unquantized footprint, plus the
key-value cache, which grows with the configured context length rather than
with the weights.

## Quality

Measured against the unquantized model, the 4-bit build loses a small amount of
accuracy on long-form reasoning and essentially none on extraction.

## Related pages

- [[model-serving-runtimes]]
- [[gpu-memory-headroom]]
