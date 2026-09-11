# Daily Notes 2026-08-14

**Summary**: Working notes for 14 August 2026, covering the retry ceiling change and a serving-slot contention incident.
**Sources**: source.md
**Last updated**: 2026-08-14

---

## Retry ceiling raised

The default retry ceiling was **raised from three attempts to eight** today
(source: source.md). Three was absorbing single blips but not the repeated
short outages seen over the previous fortnight, where a dependency bounced four
or five times in a minute and every affected run failed outright.

Eight attempts with the existing doubling backoff spans roughly four minutes,
which covers every outage measured in that period.

## Serving slot contention

A long run held the only slot for eleven minutes. Nothing was lost, but three
queued jobs started late. See [[agent-run-budget]].

## Also noted

Room and catering for this month's team social were confirmed and filed as [[retrospective-pizza-night]].

## Related pages

- [[retry-ceiling-default]]
- [[agent-run-budget]]
