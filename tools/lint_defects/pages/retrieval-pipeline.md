# Retrieval Pipeline

**Summary**: The stages a query passes through, from raw text to a ranked set of passages handed to the model.
**Sources**: source.md
**Last updated**: 2026-07-22

---

## Stages

1. Normalize the query text.
2. Embed it with the same model used at write time.
3. Search the vector index for nearest neighbours.
4. Rerank the candidates.
5. Trim to the caller's passage budget.

## Index freshness

Step 3 reads an index that is regenerated **nightly**, in a batch job that runs
after the day's writes have settled (source: source.md). Nothing rebuilds it
during the day, so a document written at 09:00 is not retrievable until the
following morning. Teams that need same-day retrieval have to wait for the next
nightly pass.

## Related pages

- [[vector-index-refresh]]
- [[rerank-models]]
