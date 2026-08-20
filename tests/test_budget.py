"""Tests for the run budget in agent/budget.py and its use in agent/loop.py.

The failure these guard against is the 2026-08-03 run: a wedged Ollama, a
retry loop that stayed inside every per-call cap, and 2h54m of wall clock. So
the assertions are about *stopping* — that a spent budget raises instead of
sleeping, and that nothing on the way out swallows it.
"""

import time

import requests

from agent import budget, loop
from tests.test_loop import FakeResp, _sequenced


# --- deadline --------------------------------------------------------------


def test_no_budget_is_a_noop():
    budget.check("anything")
    budget.before_retry(999, "reason")
    assert budget.remaining() is None
    assert budget.clamp_timeout(600) == 600


def test_check_raises_once_deadline_passes():
    budget.start_run(-1)
    try:
        budget.check("the next source")
        assert False, "expected BudgetExceeded"
    except budget.BudgetExceeded as e:
        assert "the next source" in str(e)


def test_check_passes_while_time_remains():
    budget.start_run(60)
    budget.check("the next source")


# --- hard deadline (SIGALRM) -----------------------------------------------


def test_watchdog_raises_from_a_blocking_call_no_check_can_reach():
    """The 2026-08-13 case: the deadline passes while the process sits in a
    syscall, with no checkpoint between it and the start of the run. sleep()
    stands in for the read() that macOS held behind a consent prompt."""
    budget.start_run(0.05)
    try:
        time.sleep(5)
        assert False, "expected BudgetExceeded"
    except budget.BudgetExceeded as e:
        assert "blocked outside any checkpoint" in str(e)


def test_watchdog_is_disarmed_by_reset():
    """Otherwise one test's timer fires partway through the next one."""
    budget.start_run(0.05)
    budget.reset()
    time.sleep(0.2)  # must not raise


def test_clamp_timeout_shortens_to_remaining():
    budget.start_run(5)
    assert budget.clamp_timeout(600) == 5
    assert budget.clamp_timeout(2) == 2


def test_before_retry_refuses_a_backoff_that_outlasts_the_budget():
    budget.start_run(3)
    try:
        budget.before_retry(17.0, "ReadTimeout from model API")
        assert False, "expected BudgetExceeded"
    except budget.BudgetExceeded as e:
        assert "backoff" in str(e)


# --- per-source retry ceiling ----------------------------------------------


def test_retry_ceiling_is_spent_across_calls_not_per_call():
    budget.start_source("a-source.md", max_retries=2)
    budget.before_retry(0.1, "first")
    budget.before_retry(0.1, "second")
    try:
        budget.before_retry(0.1, "third")
        assert False, "expected BudgetExceeded"
    except budget.BudgetExceeded as e:
        assert "a-source.md" in str(e)


def test_retry_ceiling_resets_per_source():
    budget.start_source("first.md", max_retries=1)
    budget.before_retry(0.1, "reason")
    budget.start_source("second.md", max_retries=1)
    budget.before_retry(0.1, "reason")  # fresh ceiling, must not raise


# --- integration with _post_with_retry -------------------------------------


def test_post_with_retry_stops_when_the_ceiling_is_spent(monkeypatch):
    """The wedged-Ollama case: every attempt times out, and the ceiling — not
    _MAX_HTTP_ATTEMPTS — is what ends it."""
    monkeypatch.setattr(loop.time, "sleep", lambda s: None)
    attempts = []

    def fake_post(*a, **k):
        attempts.append(1)
        raise requests.exceptions.ReadTimeout("wedged")

    monkeypatch.setattr(loop._session, "post", fake_post)
    budget.start_source("a-source.md", max_retries=2)

    try:
        loop._post_with_retry("http://x", {})
        assert False, "expected BudgetExceeded"
    except budget.BudgetExceeded:
        pass
    assert len(attempts) == 3  # two retries consumed, third refused


def test_post_with_retry_stops_when_the_deadline_passes(monkeypatch):
    monkeypatch.setattr(loop.time, "sleep", lambda s: None)
    monkeypatch.setattr(
        loop._session, "post",
        _sequenced([requests.exceptions.ReadTimeout("wedged")] * loop._MAX_HTTP_ATTEMPTS),
    )
    budget.start_run(-1)

    try:
        loop._post_with_retry("http://x", {})
        assert False, "expected BudgetExceeded"
    except budget.BudgetExceeded:
        pass


def test_post_with_retry_clamps_the_request_timeout(monkeypatch):
    seen = {}

    def fake_post(url, json=None, headers=None, timeout=None):
        seen["timeout"] = timeout
        return FakeResp(200, json_data={"ok": 1})

    monkeypatch.setattr(loop._session, "post", fake_post)
    budget.start_run(10)
    loop._post_with_retry("http://x", {}, timeout=600)
    assert seen["timeout"] <= 10


# --- integration with _dispatch_tool ---------------------------------------


def test_dispatch_does_not_demote_the_budget_to_a_tool_error():
    """_dispatch_tool reports a failing tool back to the model instead of
    killing the run, which is right for every exception except this one. The
    watchdog fires wherever the process happens to be, and a file read — the
    2026-08-13 blocking read() — happens inside exactly this call."""
    def wedged(**kwargs):
        raise budget.BudgetExceeded("run budget of 45 min expired")

    try:
        loop._dispatch_tool("wedged", {}, {"wedged": wedged}, None)
        assert False, "expected BudgetExceeded"
    except budget.BudgetExceeded:
        pass


def test_budget_from_a_tool_call_ends_the_whole_loop(monkeypatch):
    """And it must reach the caller, not just leave _dispatch_tool: the loop
    would otherwise run on with the one-shot watchdog already spent."""
    monkeypatch.setattr(
        loop, "_post_with_retry",
        lambda *a, **k: FakeResp(200, json_data={
            "message": {"tool_calls": [{"function": {"name": "t", "arguments": {}}}]}
        }),
    )

    def wedged(**kwargs):
        raise budget.BudgetExceeded("run budget of 45 min expired")

    try:
        loop.run_agent("sys", "user", tools=[], dispatch={"t": wedged}, provider="ollama")
        assert False, "expected BudgetExceeded"
    except budget.BudgetExceeded:
        pass


def test_an_ordinary_tool_failure_is_still_reported_to_the_model():
    """The guard above must not turn every tool error into a dead run."""
    def boom(**kwargs):
        raise RuntimeError("kaboom")

    result = loop._dispatch_tool("boom", {}, {"boom": boom}, None)
    assert "kaboom" in result["error"]


def test_healthy_run_is_untouched_by_a_generous_budget(monkeypatch):
    monkeypatch.setattr(loop.time, "sleep", lambda s: None)
    monkeypatch.setattr(
        loop._session, "post",
        _sequenced([FakeResp(503), FakeResp(200, json_data={"ok": 1})]),
    )
    budget.start_run(600)
    budget.start_source("a-source.md", max_retries=8)
    assert loop._post_with_retry("http://x", {}).json() == {"ok": 1}
