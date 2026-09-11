# Config Precedence

**Summary**: Which source of a setting wins when a file and the environment both name it.
**Sources**: source.md
**Last updated**: 2026-07-09

---

## The order

A real environment variable beats a value loaded from a settings file (source:
source.md). The loader fills in what is missing; it does not overwrite what is
already set.

## Why that direction

It makes a per-job override possible without editing a file that every other
job also reads. One scheduled job can name a different value in its own
definition and leave every other job alone.

## The trap

A setting that must never differ per job should not be settable this way at
all. Precedence is a feature for the settings you meant to vary.

## Related pages

- [[agent-engineering-overview]]
- [[lint-schedule]]
