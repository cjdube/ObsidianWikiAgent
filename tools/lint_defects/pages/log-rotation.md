# Log Rotation

**Summary**: Capping a log file that an unattended job appends to forever.
**Sources**: source.md
**Last updated**: 2026-09-06

---

## Trigger and rule

Size is the trigger; age is the rule. The file is rewritten only once it passes
a byte cap, and the rewrite then keeps whatever is newer than the retention
window (source: source.md). The common case costs one append and one stat.

## The descriptor problem

A scheduler that owns the job's output descriptor will not reopen it. The job
therefore has to cap the file it is itself writing to, which means the path has
to be handed to it explicitly rather than discovered.

## Related pages

- [[agent-run-budget]]
- [[observability-traces]]
