"""Tests for the logging plumbing in agent/common.py."""

import requests

from agent import common, notify


# --- trim_launchd_log ------------------------------------------------------


def test_trim_is_a_noop_without_the_env_var(monkeypatch, tmp_path):
    monkeypatch.delenv(common.LAUNCHD_LOG_ENV, raising=False)
    common.trim_launchd_log()  # must not raise


def test_trim_leaves_a_small_log_alone(monkeypatch, tmp_path):
    log = tmp_path / "job.launchd.log"
    log.write_text("line\n" * 10)
    monkeypatch.setenv(common.LAUNCHD_LOG_ENV, str(log))

    common.trim_launchd_log()
    assert log.read_text() == "line\n" * 10


def test_trim_keeps_the_tail_and_the_inode(monkeypatch, tmp_path):
    """The inode must survive: launchd holds this file open as the job's
    stdout, so a rename or re-create would send the rest of the run's output
    to a file nobody reads."""
    log = tmp_path / "job.launchd.log"
    log.write_bytes(b"".join(b"line %06d\n" % i for i in range(700_000)))
    original_size = log.stat().st_size
    original_inode = log.stat().st_ino
    monkeypatch.setenv(common.LAUNCHD_LOG_ENV, str(log))

    common.trim_launchd_log()

    assert log.stat().st_size < original_size
    assert log.stat().st_ino == original_inode
    text = log.read_text()
    assert text.endswith("line 699999\n")
    assert text.startswith("line ")  # no partial first line


def test_trim_survives_a_missing_file(monkeypatch, tmp_path):
    monkeypatch.setenv(common.LAUNCHD_LOG_ENV, str(tmp_path / "gone.log"))
    common.trim_launchd_log()  # must not raise


# --- notify ----------------------------------------------------------------


def test_notify_reports_when_push_is_switched_off(monkeypatch):
    monkeypatch.delenv("NTFY_URL", raising=False)
    assert "error" in notify.notify("anything")


def test_notify_never_raises_on_a_dead_server(monkeypatch):
    monkeypatch.setenv("NTFY_URL", "http://example.invalid/topic")

    def boom(*a, **k):
        raise requests.exceptions.ConnectionError("no route")

    monkeypatch.setattr(notify.requests, "post", boom)
    result = notify.notify("ingest failed")
    assert "ConnectionError" in result["error"]


def test_notify_failure_swallows_a_failed_push(monkeypatch):
    """The push reports a failure; it must never become one."""
    monkeypatch.setattr(notify, "notify", lambda **kw: {"error": "down"})
    warnings = []

    class L:
        def warning(self, msg):
            warnings.append(msg)

    notify.notify_failure("wiki_ingest[v]", "boom", L())
    assert warnings and "down" in warnings[0]
