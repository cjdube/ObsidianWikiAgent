# Embedding Storage

**Summary**: Where document vectors live on disk and how they are keyed back to their source documents.
**Sources**: source.md
**Last updated**: 2026-08-09

---

## Layout

Vectors sit in a column beside the document row, keyed by document id. There is
no separate vector store to keep in step with the document store, which removes
a whole class of drift.

## Population

The column starts empty. A document's vector is computed **lazily, on the first
read that needs it**, and written back for later reads (source: source.md). A
document that is never queried is never embedded, which is what keeps the bill
proportional to traffic rather than to corpus size.

## Consequence

The first query touching a cold document pays the embedding latency. Warming is
a matter of issuing that first query ahead of the user.

## Related pages

- [[batch-embedding]]
- [[vector-index-refresh]]
