# Agent Engineering Overview

**Summary**: A short map of what this knowledge base covers and how its main subjects relate to each other.
**Sources**: source.md
**Last updated**: 2026-09-01

---

## Three layers

Serving, retrieval, and the agent loop. Most problems that look like model
quality turn out to live in one of the lower two (source: source.md).

## Serving

What the weights cost to hold and how fast they answer. See
[[model-serving-runtimes]] and [[gpu-memory-headroom]].

## Retrieval

Turning a question into a small set of passages worth reading. Splitting is the
first decision — see [[semantic-chunking]] and [[meaning-based-splitting]] —
and ranking is the last.

## The loop

Calling tools, handling their failures, and stopping in time. See
[[function-call-retry-policy]], [[tool-schema-design]] and
[[structured-output]].

## Operating it

[[streaming-responses]], [[sqlite-checkpointing]] and [[observability-traces]]
cover what it takes to run this in front of people.
