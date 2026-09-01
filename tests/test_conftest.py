"""The suite's own guards, tested.

A guard that has quietly stopped working looks exactly like a guard that has
nothing to stop: the suite is green either way. These tests are the difference.
Each one does the thing the guard exists to refuse and asserts that it is
refused — so if a future edit to conftest.py loosens one, this file fails
instead of the phone ringing.
"""

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

import pytest

from conftest import (
    _REAL_LOGS_DIR,
    _ModelEgress,
    _NtfyEgress,
    _ProductionLogWrite,
)


def test_the_real_logs_dir_is_the_repo_s_own():
    """If this resolves somewhere else, the handler guard below is guarding
    nothing and the other tests here would still pass."""
    assert _REAL_LOGS_DIR == Path(__file__).resolve().parent.parent / "logs"


def test_a_handler_on_the_real_logs_dir_is_refused():
    with pytest.raises(_ProductionLogWrite):
        RotatingFileHandler(_REAL_LOGS_DIR / "should-never-exist.log")
    assert not (_REAL_LOGS_DIR / "should-never-exist.log").exists()


def test_a_handler_on_a_tmp_dir_still_works(tmp_path):
    """The guard must refuse one directory, not file logging in general —
    setup_logger has to keep working under the redirect."""
    handler = RotatingFileHandler(tmp_path / "fine.log")
    handler.close()
    assert (tmp_path / "fine.log").exists()


def test_setup_logger_is_redirected_away_from_production(tmp_path):
    """The isolate-and-refuse pair, together: _isolate_logs points LOGS_DIR at
    tmp_path, so the guard never fires for a logger built the normal way."""
    from agent import common

    assert common.LOGS_DIR != _REAL_LOGS_DIR
    logger = common.setup_logger("guard-check")
    logger.info("hello")
    for handler in logger.handlers:
        handler.close()
    assert (common.LOGS_DIR / "guard-check.log").exists()


def test_the_ntfy_push_is_refused():
    """agent.notify.notify catches bare Exception, so the guard has to raise
    something that is not one — otherwise this call returns {"error": ...}
    and reads as a clean, switched-off push."""
    from agent import notify

    with pytest.raises(_NtfyEgress):
        notify.notify("this must never leave the machine")


def test_notify_failure_is_refused_too():
    """The wrapper the scheduled jobs actually call. It is best-effort by
    design and swallows everything below it, which is exactly why the guard
    must not be swallowable."""
    from agent import notify

    with pytest.raises(_NtfyEgress):
        notify.notify_failure("wiki_ingest[v]", "boom")


def test_the_model_session_is_refused():
    """Both providers post through this one Session, so this covers Ollama and
    Gemini together."""
    from agent import loop

    with pytest.raises(_ModelEgress):
        loop._session.post("http://localhost:11434/api/chat", json={})
    with pytest.raises(_ModelEgress):
        loop._session.get("http://localhost:11434/api/tags")


def test_post_with_retry_does_not_swallow_the_model_guard(monkeypatch):
    """_post_with_retry catches RequestException and retries five times. A guard
    raised as one would be retried, slept over, and finally re-raised as a
    network error — the wrong failure, several seconds later."""
    from agent import loop

    monkeypatch.setattr(loop.time, "sleep", lambda s: pytest.fail("guard was retried"))
    with pytest.raises(_ModelEgress):
        loop._post_with_retry("http://localhost:11434/api/chat", {})


def test_the_model_env_is_hermetic():
    """_hermetic_model_env: the suite must not read the developer's config/.env,
    which agent/__init__.py loads at import."""
    import os

    assert os.getenv("OLLAMA_MODEL") == "test-model"
    assert os.getenv("LLM_PROVIDER") is None
    assert os.getenv("GEMINI_API_KEY") is None


def test_the_budget_does_not_leak_between_tests():
    """_clear_budget: budget state is module-global, so a run started in one
    test would otherwise set the deadline for the next."""
    from agent import budget

    budget.check("nothing should be running")
