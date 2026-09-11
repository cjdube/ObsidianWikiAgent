# Context Window Limits

**Summary**: The hard ceiling on one call's transcript, and what happens when a run crosses it.
**Sources**: source.md
**Last updated**: 2026-08-02

---

## The ceiling

One call may carry at most **32,768 tokens** of transcript (source: source.md).
This is a property of how the server is configured, not of the request, so a
caller cannot raise it per run.

## Overflow is silent

Crossing the ceiling does not raise. The oldest messages are dropped and the
call proceeds against a truncated view, so the only symptom is that the answers
get worse. That is why the ceiling is worth stating in a page at all.

## Practice

Measure the opening prompt. If it alone fills half the ceiling, the run is one
long tool result away from dropping its own instructions.

## Related pages

- [[token-budgeting]]
- [[gpu-memory-headroom]]
