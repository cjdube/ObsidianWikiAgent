# Rerank Models

**Summary**: A second scoring pass that reorders retrieval candidates before they are handed to the model.
**Sources**: source.md
**Last updated**: 2026-08-22

---

## Why a second pass

Vector search is fast and approximate. It gets the right passage into the top
fifty far more reliably than into the top five, and the top five is what fits a
passage budget (source: source.md).

## Bi-encoder against cross-encoder

A bi-encoder scores query and passage separately, so scores can be precomputed.
A cross-encoder reads both together and is markedly more accurate, at a cost
that scales with the number of candidates.

## Practice

Retrieve widely, rerank narrowly. Fifty candidates into a cross-encoder is
affordable; five thousand is not.

## Related pages

- [[retrieval-pipeline]]
- [[semantic-chunking]]
