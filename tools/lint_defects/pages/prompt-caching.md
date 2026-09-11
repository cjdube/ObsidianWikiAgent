# Prompt Caching

**Summary**: Reusing the server's computed state for a repeated prompt prefix, and the conditions under which the reuse actually happens.
**Sources**: source.md
**Last updated**: 2026-07-02

---

## What is cached

The computed state for a prefix of the prompt. A later call whose prompt begins
with the identical bytes can skip recomputing that prefix (source: source.md).

## The prefix rule

Reuse only happens on a strict extension. Changing anything near the start —
even a timestamp in the system text — invalidates everything after it, so the
whole prompt is recomputed and the cache buys nothing.

## Practice

Put the stable material first and the varying material last. A prompt that
opens with the current time is a prompt that can never hit.

## Related pages

- [[token-budgeting]]
- [[model-serving-runtimes]]
