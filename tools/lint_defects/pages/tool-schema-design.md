# Tool Schema Design

**Summary**: Writing tool definitions the model can call correctly on the first attempt.
**Sources**: source.md
**Last updated**: 2026-07-16

---

## Name the argument the way the model thinks

Most first-attempt failures are argument-name mismatches, not logic errors. A
tool that swallows near-miss keyword names and answers with a clear message
costs less than one that raises (source: source.md).

## Bound every result

A result sized by the corpus rather than by the question breaks as the corpus
grows. Prefer a search that returns a fixed number of rows over a listing that
returns everything.

## Say what to send instead

An error that says a call failed teaches nothing. An error that names the
correct call is the difference between one retry and four identical ones.

## Related pages

- [[structured-output]]
- [[function-call-retry-policy]]
