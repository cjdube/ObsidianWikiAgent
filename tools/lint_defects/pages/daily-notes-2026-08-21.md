# Daily Notes 2026-08-21

**Summary**: Working notes for 21 August 2026, covering the new 8-bit build and a reranker swap.
**Sources**: source.md
**Last updated**: 2026-08-21

---

## An 8-bit build shipped

An **8-bit quantized build was published today**, alongside the existing 4-bit
one (source: source.md). It needs about twice the weight memory of the 4-bit
build and recovers most of the long-form reasoning gap.

The practical effect is that a box with headroom no longer has to choose
between 4-bit and unquantized. See [[quantization-builds]], which still
describes 4-bit as the only option.

## Reranker swap

The cross-encoder reranker replaced the bi-encoder in the ranking step. Latency
rose, precision rose more. See [[rerank-models]].

## Also noted

The annual volunteer day sign-up sheet went round. It came through the same collected source and was filed as [[volunteer-day-logistics]].

## Related pages

- [[quantization-builds]]
- [[rerank-models]]
