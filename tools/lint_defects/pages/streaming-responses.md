# Streaming Responses

**Summary**: Emitting a reply token by token instead of waiting for the whole thing, and what that costs a programmatic caller.
**Sources**: source.md
**Last updated**: 2026-06-28

---

## For a human

Streaming changes perceived latency far more than real latency. The first token
arrives in a fraction of the total time, and the wait stops feeling like a wait
(source: source.md).

## For a program

A program cannot act on half a tool call. Streaming a reply that will be parsed
adds reassembly work and a new class of partial-parse bug, for no benefit.

## Practice

Stream to people. Do not stream to parsers.

## Related pages

- [[structured-output]]
- [[agent-engineering-overview]]
