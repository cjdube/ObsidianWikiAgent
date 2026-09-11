# Function Call Retry Policy

**Summary**: The rules governing repeated attempts at a function invocation that did not return cleanly.
**Sources**: source.md
**Last updated**: 2026-08-18

---

## Policy statement

When an invocation does not return cleanly, the runtime attempts it again
rather than surfacing the failure straight to the caller (source: source.md).
The number of further attempts is bounded, and the bound is configuration, not
a constant.

## Classification

Retryable:

- Connection reset before any bytes arrived
- Read timeout with no partial body
- A gateway status in the 502-504 range

Not retryable:

- Any response the function itself produced, including one describing an error
- Malformed arguments, which will be malformed on every attempt

## Spacing

Exponential, from a one-second base, so a wedged dependency is not hammered.

## Related pages

- [[config-precedence]]
- [[observability-traces]]
