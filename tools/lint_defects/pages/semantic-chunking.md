# Semantic Chunking

**Summary**: Splitting a document at meaning boundaries rather than at a fixed character count.
**Sources**: source.md
**Last updated**: 2026-06-21

---

## The idea

A fixed-width split cuts sentences in half and strands a heading away from the
text it introduces. Semantic chunking looks for the places where the subject
actually changes and cuts there instead (source: source.md).

## How the boundary is found

Embed each sentence. Walk the document and compare each sentence to the running
mean of the current chunk. When the similarity drops below a threshold, close
the chunk and start a new one.

## Trade-off

Chunk sizes become uneven, which makes downstream budgeting harder. A long,
uniform section can produce one very large chunk that no passage budget will
accept whole.

## Related pages

- [[retrieval-pipeline]]
- [[rerank-models]]
