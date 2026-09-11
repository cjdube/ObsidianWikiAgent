# Observability Traces

**Summary**: Recording each step of a run so a failure can be reconstructed after the fact instead of reproduced.
**Sources**: source.md
**Last updated**: 2026-08-19

---

## One line per step

Each tool call, its arguments, and its result. The arguments are the part people
leave out and the part that turns out to matter, because a repeated identical
call is invisible without them (source: source.md).

## Levels are a contract

A downstream reader that alerts on warnings will alert on anything logged as
one. A weekly report logged at warning level becomes a phone notification, so
the level is a decision about the reader, not about the message.

## Related pages

- [[log-rotation]]
- [[eval-harnesses]]
