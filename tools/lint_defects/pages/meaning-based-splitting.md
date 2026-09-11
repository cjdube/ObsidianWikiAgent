# Meaning-Based Splitting

**Summary**: A document-division strategy that places cut points where the topic shifts instead of every N characters.
**Sources**: source.md
**Last updated**: 2026-08-25

---

## Motivation

Character-count division is cheap and wrong in a specific way: it has no idea
what it is cutting. The result is passages that begin mid-clause and headings
orphaned from their bodies (source: source.md).

## Procedure

1. Score adjacent sentence pairs for topical continuity.
2. Mark the low-scoring joints as candidate cut points.
3. Cut at candidates, subject to a minimum and maximum passage length.

## What it costs

Every sentence must be scored before any cut is made, so the whole document is
processed before the first passage is emitted. Streaming a very large document
through this is not possible without buffering it.

## Related pages

- [[batch-embedding]]
- [[eval-harnesses]]
