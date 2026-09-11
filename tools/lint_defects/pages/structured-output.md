# Structured Output

**Summary**: Constraining a reply to a schema so a caller can parse it without guessing.
**Sources**: source.md
**Last updated**: 2026-06-25

---

## Why constrain

An unconstrained reply is prose, and parsing prose is a guess that fails
silently. A schema turns a parse failure into an error at the boundary, where
somebody can see it (source: source.md).

## Where it goes wrong

An over-tight schema pushes the model into filling required fields with
plausible nonsense rather than admitting it does not know. Optional fields and
an explicit "unknown" value are usually better than a required string.

## Related pages

- [[tool-schema-design]]
- [[eval-harnesses]]
