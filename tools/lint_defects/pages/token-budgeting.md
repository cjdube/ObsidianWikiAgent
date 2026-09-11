# Token Budgeting

**Summary**: How a run divides its context window between instructions, accumulated tool results, and the reply.
**Sources**: source.md
**Last updated**: 2026-06-30

---

## The budget

A run is given a **128,000 token** context window (source: source.md). That is
the number every budgeting decision below is drawn against.

## Division

- Instructions and page rules: a fixed cost, paid once per run.
- Tool results: the part that grows. Each page read is added to the transcript
  and never removed.
- The reply: reserved last, and the first thing squeezed when the rest overrun.

## Practice

Reserve a quarter of the window for the reply. A run that spends more than
three quarters on tool results is reading too widely, and the fix is a narrower
search rather than a larger window.

## Related pages

- [[context-window-limits]]
- [[prompt-caching]]
