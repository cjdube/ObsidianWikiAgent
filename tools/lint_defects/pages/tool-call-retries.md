# Tool Call Retries

**Summary**: What the loop does when a tool call fails, and how many times it will try again before giving up.
**Sources**: source.md
**Last updated**: 2026-07-11

---

## Behaviour

A failed tool call is retried in place. The loop re-sends the same call with the
same arguments, waits, and re-sends again, up to a fixed ceiling (source:
source.md). The transcript keeps every attempt, so a reader can see how many
times a call was tried.

## Which failures qualify

Only transport failures. A tool that returns an error *object* has succeeded at
the protocol level and answered the model; re-sending it would produce the same
object forever. The distinction matters because the two look identical in a log
that only records "failed".

## Backoff

Attempts are spaced by doubling waits, starting at one second.

## Related pages

- [[rate-limit-backoff]]
- [[agent-engineering-overview]]
