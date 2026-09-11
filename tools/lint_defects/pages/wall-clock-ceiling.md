# Wall Clock Ceiling

**Summary**: An upper bound in real seconds on how long a single job may run before it is required to abandon what is left.
**Sources**: source.md
**Last updated**: 2026-08-30

---

## Definition

A ceiling measured in wall-clock seconds, not in units of work. Wall clock is
the right unit because the thing being protected is the schedule: the next job
starts at a fixed time whether or not this one has finished (source: source.md).

## Enforcement points

The ceiling is consulted at the boundary between units, never in the middle of
one. Interrupting a unit mid-write is how half-written state is produced, and
half-written state costs more to repair than the overrun cost to tolerate.

## Failure mode it prevents

Without a ceiling, one wedged dependency turns a bounded job into an unbounded
one. Retries multiply: a ceiling of attempts times a ceiling of units is not a
bound anybody has actually reasoned about.

## Related pages

- [[agent-run-budget]]
- [[rate-limit-backoff]]
