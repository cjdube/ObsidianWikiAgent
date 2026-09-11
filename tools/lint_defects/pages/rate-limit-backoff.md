# Rate Limit Backoff

**Summary**: Spacing repeated attempts so a throttled dependency is given room to recover.
**Sources**: source.md
**Last updated**: 2026-08-06

---

## Doubling, with jitter

Each wait is twice the last, plus a random fraction. The jitter matters more
than the doubling: without it, every caller throttled at the same moment
retries at the same moment (source: source.md).

## Respect an explicit hint

When the response names a wait time, use it. A computed backoff that ignores a
stated one is guessing against an answer it was given.

## Related pages

- [[wall-clock-ceiling]]
- [[observability-traces]]
