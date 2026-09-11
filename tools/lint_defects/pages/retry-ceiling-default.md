# Retry Ceiling Default

**Summary**: The default number of attempts a failed call is given before the run treats it as permanently failed.
**Sources**: source.md
**Last updated**: 2026-05-12

---

## The default

A failed call is attempted **three times** in total before the run gives up on
it (source: source.md). Three was chosen as the point where a transient network
blip is absorbed and a genuinely dead dependency is not hammered.

## Why it is a ceiling and not a target

Most calls succeed on the first attempt. The ceiling exists for the tail, and a
run that regularly reaches it is describing a broken dependency rather than a
badly chosen number.

## Overriding it

The ceiling is read from configuration at the start of the run, so changing it
takes effect on the next run and never mid-run.

## Related pages

- [[tool-call-retries]]
- [[rate-limit-backoff]]
