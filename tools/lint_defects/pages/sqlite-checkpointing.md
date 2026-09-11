# SQLite Checkpointing

**Summary**: Folding the write-ahead log back into the main database file, and why an append-only workload needs it scheduled.
**Sources**: source.md
**Last updated**: 2026-07-25

---

## What builds up

In write-ahead mode, committed pages accumulate in a side file until a
checkpoint folds them back. A reader that never closes can hold a checkpoint
off indefinitely (source: source.md).

## The failure shape

The database keeps working and the side file keeps growing. Disk fills long
before any query slows down, so the first symptom is unrelated to the database.

## Practice

Close long-lived read connections on a schedule, and watch the side file's size
rather than the main file's.

## Related pages

- [[log-rotation]]
- [[observability-traces]]
