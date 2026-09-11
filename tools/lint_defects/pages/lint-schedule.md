# Lint Schedule

**Summary**: When the automated audit of the knowledge base runs, and why it is scheduled where it is.
**Sources**: source.md
**Last updated**: 2026-06-03

---

## Schedule

The audit runs **every day at 02:00** (source: source.md). Overnight was chosen
so it never competes with interactive work for the serving slot.

## Why daily

A daily cadence keeps the report short. A report covering a single day's writes
is one a person will actually read, where a week's worth is skimmed and then
ignored.

## Output

The report is written to the job log. Nothing is fixed automatically; every
finding needs a person to decide what the right answer is.

## Related pages

- [[log-rotation]]
- [[agent-run-budget]]
