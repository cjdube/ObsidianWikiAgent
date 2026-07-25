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
    assert vs.snapshot_vault(str(tmp_path / "nope"), logger) == 1


def test_non_git_directory_fails(tmp_path, logger):
    plain = tmp_path / "plain"
    plain.mkdir()
    assert vs.snapshot_vault(str(plain), logger) == 1


def test_clean_repo_makes_no_commit(repo, logger):
    before = head_message(repo)
    assert vs.snapshot_vault(str(repo), logger) == 0
    assert head_message(repo) == before


def test_commits_and_pushes_changes(repo, logger):
    (repo / "wiki").mkdir()
    (repo / "wiki" / "new page.md").write_text("body", encoding="utf-8")
    (repo / "seed.md").write_text("edited", encoding="utf-8")

    assert vs.snapshot_vault(str(repo), logger) == 0

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
    assert vs.snapshot_vault(str(repo), logger) == 1

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
