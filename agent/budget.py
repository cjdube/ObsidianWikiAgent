"""Wall-clock and retry ceilings for one unattended run.

A scheduled run competes for a shared resource. Ollama runs with
OLLAMA_NUM_PARALLEL=1 — one request at a time, everything else queues
silently — so a wiki_ingest run that keeps retrying holds the only slot and
starves every other consumer on the box. On 2026-08-03 the MLX runner wedged
at 09:02:30 and the 09:00 ingest retried against it until 11:54: nearly three
hours in which the retries themselves *were* the outage for everything else
talking to that Ollama.

Nothing in the retry path noticed, because every individual wait was
reasonable. The damage is in the product: OLLAMA_TIMEOUT (300s by default, set
to 600 in config/.env) x _MAX_HTTP_ATTEMPTS (5) x MAX_INGEST_ITERATIONS (30) x
MAX_INGEST_ATTEMPTS (3) x every pending source. So a run is bounded twice over — by wall clock for
the whole process, and by transport retries per source.

Both limits are process-global rather than parameters threaded through the
call chain, because both are properties of "this run" and not of any single
model call. agent/loop.py consults them from inside its own retry loop, which
is where the hours were actually spent; a check that only ran between sources
would not have stopped that morning's run any sooner.

With no budget started every check is a no-op, so interactive runs and the test
suite are unaffected. wiki_lint starts one for its --deep pass only: that pass
is a scheduled unattended model conversation and has the same unbounded
retry product, while its structural pass is pure Python over local files and
consults nothing here.
"""

import math
import time


class BudgetExceeded(RuntimeError):
    """The run's wall-clock deadline or a source's retry ceiling is spent.

    Deliberately not an Exception subclass the ingest loops treat as a normal
    per-source failure: callers that catch broad exceptions to retry must let
    this one through, or the budget does nothing.
    """


_deadline: float | None = None
_source: str | None = None
_retries_left: int | None = None


def start_run(seconds: float) -> None:
    """Begin the wall-clock budget for this process."""
    global _deadline
    _deadline = time.monotonic() + seconds


def start_source(name: str, max_retries: int) -> None:
    """Begin a fresh retry ceiling for one unit of work (one raw source, or
    one file being sorted). Retries are counted across the whole unit, not
    per model call: a wedged server otherwise stays under every per-call cap
    while the totals run to hundreds."""
    global _source, _retries_left
    _source = name
    _retries_left = max_retries


def reset() -> None:
    """Clear all budget state (used by the test suite between tests)."""
    global _deadline, _source, _retries_left
    _deadline = None
    _source = None
    _retries_left = None


def remaining() -> float | None:
    """Seconds left in the run budget, or None when no budget is running."""
    if _deadline is None:
        return None
    return _deadline - time.monotonic()


def check(what: str) -> None:
    """Raise BudgetExceeded if the run's wall clock is already spent."""
    left = remaining()
    if left is not None and left <= 0:
        raise BudgetExceeded(f"run budget exhausted before {what}")


def before_retry(delay: float, reason: str) -> None:
    """Gate one transport retry: raise BudgetExceeded rather than waiting when
    the source's retry ceiling is spent, or when sleeping `delay` and trying
    again would run past the run deadline. Otherwise consume one retry."""
    global _retries_left

    if _retries_left is not None:
        if _retries_left <= 0:
            raise BudgetExceeded(
                f"retry ceiling for '{_source}' exhausted ({reason})"
            )
        _retries_left -= 1

    left = remaining()
    if left is not None and delay >= left:
        raise BudgetExceeded(
            f"run budget has {left:.0f}s left, less than the {delay:.0f}s "
            f"backoff before the next attempt ({reason})"
        )


def clamp_timeout(timeout: int) -> int:
    """Shorten a request timeout so one slow call can't overshoot the deadline
    by its full length — OLLAMA_TIMEOUT is 600s, ten minutes of overshoot on
    a 45-minute budget."""
    left = remaining()
    if left is None:
        return timeout
    return max(1, min(timeout, math.ceil(left)))
