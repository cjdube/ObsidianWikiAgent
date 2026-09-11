# Eval Harnesses

**Summary**: The scaffolding that turns a subjective impression of quality into a number two runs can be compared on.
**Sources**: source.md
**Last updated**: 2026-08-11

---

## What a harness has to hold

A fixed input set, a fixed scoring rule, and enough trials that a difference is
not noise (source: source.md). Drop any one of the three and the number stops
being comparable across runs.

## Planted defects

The most useful input set for an audit task is one with known defects planted in
it. Recall against a known answer key is measurable; "did the report seem good"
is not.

## What not to automate

Precision. A keyword rule can check whether a report names the right page. It
cannot tell a true claim from a confident false one, and a harness that pretends
otherwise reports a worse number than no harness at all.

## Related pages

- [[structured-output]]
- [[observability-traces]]
