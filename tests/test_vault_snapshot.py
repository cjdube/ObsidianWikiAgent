"""Tests for vault_snapshot.py.

Real git, no network: each test builds a throwaway repo under tmp_path and
pushes to a local bare repo standing in for the remote.
"""

import logging
import subprocess

import pytest

import vault_snapshot as vs


def git(repo, *args):
    subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    )


@pytest.fixture
def logger():
    log = logging.getLogger("test_vault_snapshot")
    log.handlers.clear()
    log.addHandler(logging.NullHandler())
    return log


@pytest.fixture
def repo(tmp_path):
    """A repo with one commit, tracking a local bare 'remote'."""
    remote = tmp_path / "remote.git"
    subprocess.run(["git", "init", "-q", "--bare", str(remote)], check=True)

    work = tmp_path / "vault"
    work.mkdir()
    git(work, "init", "-q", "-b", "main")
    git(work, "config", "user.email", "test@example.com")
    git(work, "config", "user.name", "Test")
    (work / "seed.md").write_text("seed", encoding="utf-8")
    git(work, "add", "-A")
    git(work, "commit", "-q", "-m", "seed")
    git(work, "remote", "add", "origin", str(remote))
    git(work, "push", "-q", "-u", "origin", "main")
    return work


def head_message(repo):
    return subprocess.run(
        ["git", "-C", str(repo), "log", "-1", "--pretty=%s"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()


def test_missing_directory_fails(tmp_path, logger):
    assert vs.snapshot_vault(str(tmp_path / "nope"), logger) is not None


def test_non_git_directory_fails(tmp_path, logger):
    plain = tmp_path / "plain"
    plain.mkdir()
    assert vs.snapshot_vault(str(plain), logger) is not None


def test_clean_repo_makes_no_commit(repo, logger):
    before = head_message(repo)
    assert vs.snapshot_vault(str(repo), logger) is None
    assert head_message(repo) == before


def test_commits_and_pushes_changes(repo, logger):
    (repo / "wiki").mkdir()
    (repo / "wiki" / "new page.md").write_text("body", encoding="utf-8")
    (repo / "seed.md").write_text("edited", encoding="utf-8")

    assert vs.snapshot_vault(str(repo), logger) is None

    # Filenames with spaces count as one file, not two.
    assert head_message(repo).endswith(": 2 files")
    assert subprocess.run(
        ["git", "-C", str(repo), "status", "--porcelain"],
        capture_output=True, text=True, check=True,
    ).stdout == ""
    # The remote actually received it.
    local = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        capture_output=True, text=True, check=True,
    ).stdout
    remote = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "origin/main"],
        capture_output=True, text=True, check=True,
    ).stdout
    assert local == remote


def test_failed_push_keeps_the_commit(repo, logger):
    """A non-fast-forward must not lose work or trigger a force push."""
    # Another machine pushed first.
    other = repo.parent / "other"
    subprocess.run(
        ["git", "clone", "-q", str(repo.parent / "remote.git"), str(other)],
        check=True,
    )
    git(other, "config", "user.email", "other@example.com")
    git(other, "config", "user.name", "Other")
    (other / "elsewhere.md").write_text("theirs", encoding="utf-8")
    git(other, "add", "-A")
    git(other, "commit", "-q", "-m", "from another machine")
    git(other, "push", "-q")

    (repo / "mine.md").write_text("mine", encoding="utf-8")
    assert "git push failed" in vs.snapshot_vault(str(repo), logger)

    # Local commit survived, and the other machine's commit is still the tip
    # of the remote — nothing was overwritten.
    assert head_message(repo).endswith(": 1 files")
    tip = subprocess.run(
        ["git", "-C", str(other), "rev-parse", "HEAD"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    remote_tip = subprocess.run(
        ["git", "-C", str(repo.parent / "remote.git"), "rev-parse", "main"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    assert tip == remote_tip


# --- run boundaries (what LocalLLMAgent's dashboard parses) ----------------


def _log_lines(repo, tmp_path):
    return (tmp_path / f"vault_snapshot.{repo.name}.log").read_text().splitlines()


def test_main_logs_run_boundaries(repo, monkeypatch, tmp_path):
    # Without these two markers the log is a flat stream of one-off lines and
    # nothing can tell where one nightly run ends and the next begins.
    (repo / "new.md").write_text("new", encoding="utf-8")
    monkeypatch.setattr("sys.argv", ["vault_snapshot.py", "--vault", str(repo)])
    assert vs.main() == 0

    lines = _log_lines(repo, tmp_path)
    assert any(f"Starting vault snapshot run for vault: {repo}" in ln for ln in lines)
    assert any("Vault snapshot run complete" in ln for ln in lines)


def test_failed_run_does_not_log_as_complete(repo, monkeypatch, tmp_path):
    # A "run complete" line after a failed push would close the run as a
    # success and paint over the ERROR lines above it.
    monkeypatch.setattr(vs, "notify_failure", lambda *a, **k: None)
    git(repo, "remote", "set-url", "origin", str(tmp_path / "nonexistent.git"))
    (repo / "new.md").write_text("new", encoding="utf-8")
    monkeypatch.setattr("sys.argv", ["vault_snapshot.py", "--vault", str(repo)])
    assert vs.main() == 1

    lines = _log_lines(repo, tmp_path)
    assert not any("run complete" in ln for ln in lines)
    assert any("Vault snapshot run failed" in ln for ln in lines)


# --- failure notification --------------------------------------------------


def test_main_pushes_the_reason_a_push_failed(repo, monkeypatch, tmp_path):
    # The bug this alerting exists for: an unreachable remote failed silently
    # every night for ten nights. The alert has to name the cause, or it just
    # sends you to the same log nobody was reading.
    pushed = []
    monkeypatch.setattr(
        vs, "notify_failure",
        lambda job, detail, logger=None: pushed.append((job, str(detail))),
    )
    git(repo, "remote", "set-url", "origin", str(tmp_path / "nonexistent.git"))
    (repo / "new.md").write_text("new", encoding="utf-8")
    monkeypatch.setattr("sys.argv", ["vault_snapshot.py", "--vault", str(repo)])

    assert vs.main() == 1
    assert len(pushed) == 1
    assert pushed[0][0] == f"vault_snapshot[{repo.name}]"
    assert "git push failed" in pushed[0][1]


def test_main_is_quiet_on_a_clean_run(repo, monkeypatch, tmp_path):
    # A vault with nothing to commit is the common case; it must not alert.
    pushed = []
    monkeypatch.setattr(
        vs, "notify_failure", lambda *a, **k: pushed.append(1),
    )
    monkeypatch.setattr("sys.argv", ["vault_snapshot.py", "--vault", str(repo)])

    assert vs.main() == 0
    assert pushed == []
