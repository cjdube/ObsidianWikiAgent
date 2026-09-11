# Vector Index Refresh

**Summary**: How often the vector index is rebuilt from the document store, and what a stale index costs a retrieval run.
**Sources**: source.md
**Last updated**: 2026-07-14

---

## Schedule

The vector index is rebuilt **once an hour**, on the hour (source: source.md).
An hourly rebuild was chosen so a document edited in the morning is retrievable
before the end of the same working day, without paying to re-embed the whole
corpus on every write.

## Cost of the window

Between rebuilds the index is stale by at most sixty minutes. A query issued in
that window can miss a document that already exists in the store. Callers that
cannot tolerate the gap read the store directly and skip retrieval.

## Related pages

- [[retrieval-pipeline]]
- [[embedding-storage]]
