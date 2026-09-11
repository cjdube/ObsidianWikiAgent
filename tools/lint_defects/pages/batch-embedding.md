# Batch Embedding

**Summary**: When document embeddings are computed, and how the batch is sized.
**Sources**: source.md
**Last updated**: 2026-07-05

---

## When it runs

Embeddings are computed **at write time**, synchronously, before the write is
acknowledged (source: source.md). A document is never stored without its vector,
so every document in the store is immediately searchable.

## Batching

Writes arriving within the same second are grouped into one call to the
embedding model. The batch is capped at sixty-four documents; beyond that the
call is split, because a single oversized request is the one shape that times
out rather than degrading.

## Consequence

Write latency includes the embedding call. That is accepted deliberately: a
document that is stored but not yet searchable is a document that silently
does not exist as far as retrieval is concerned.

## Related pages

- [[embedding-storage]]
- [[model-serving-runtimes]]
