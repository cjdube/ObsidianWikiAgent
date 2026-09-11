# Daily Notes 2026-09-06

**Summary**: Working notes for 6 September 2026, covering the audit schedule change and log rotation sizing.
**Sources**: source.md
**Last updated**: 2026-09-06

---

## Audit moved to weekly

The automated audit **no longer runs daily at 02:00**. It now runs **once a
week, on Sunday at 10:00**, after that morning's ingest, so it audits a full
week of new pages (source: source.md).

Daily was producing a near-empty report six days out of seven, and the empty
reports trained everyone to skip the seventh. See [[lint-schedule]], which
still states the old daily schedule.

## Log rotation sizing

The job log was capped at five megabytes with ninety days of retention. Size is
the trigger, age is the rule. See [[log-rotation]].

## Also noted

Benefits enrollment dates were circulated and were filed as [[dental-plan-enrollment]].

## Related pages

- [[lint-schedule]]
- [[log-rotation]]
