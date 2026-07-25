"""Commit and push a snapshot of a vault to its git remote.

This is its own job rather than a step at the end of wiki_ingest.py because a
vault has two writers: the scheduled ingest, and the human editing in Obsidian.
Tying commits to ingest would miss every hand edit made on a day with no new
raw/ sources, and would make an ingest failure cost you the backup of those
edits too.

Never touches vault *content* — it only commits what is already on disk.

Python rather than a shell script for one specific reason: a vault under
~/Documents (or ~/Desktop) is protected by macOS TCC, and a launchd-spawned
/bin/bash has no grant for it, so git there fails with "Operation not
permitted". The .venv interpreter that already runs the ingest does have the
grant, and git inherits it as a subprocess.

Vault-agnostic, like the other entrypoints: a vault with no git remote simply
does not schedule this job.

Usage:
    python vault_snapshot.py --vault ~/Documents/llm-wiki-learnings
"""

import argparse
import os
import subprocess
import sys
from datetime import date
from pathlib import Path

from agent.common import setup_logger


def _git(vault: Path, *args: str) -> subprocess.CompletedProcess:
    # GIT_TERMINAL_PROMPT=0: there is no tty under launchd, so a missing
    # credential must fail and be logged rather than block on a username
    # prompt forever.
    return subprocess.run(
        ["git", "-C", str(vault), *args],
        capture_output=True,
        text=True,
        env={**os.environ, "GIT_TERMINAL_PROMPT": "0"},
    )


def snapshot_vault(vault: str, logger) -> int:
    path = Path(vault).expanduser()

    if not path.is_dir():
        logger.error(f"No such directory: {path}")
        return 1
    if _git(path, "rev-parse", "--git-dir").returncode != 0:
        logger.error(f"Not a git repository: {path}")
        return 1

    add = _git(path, "add", "-A")
    if add.returncode != 0:
        logger.error(f"git add failed: {add.stderr.strip()}")
        return 1

    staged = _git(path, "diff", "--cached", "--name-only").stdout.splitlines()
    if not staged:
        logger.info(f"No changes in {path}")
        return 0

    message = f"Vault snapshot {date.today():%Y-%m-%d}: {len(staged)} files"
    commit = _git(path, "commit", "-m", message)
    if commit.returncode != 0:
        logger.error(f"git commit failed: {commit.stderr.strip()}")
        return 1
    logger.info(f"Committed {len(staged)} files")

    # Plain push, never --force. A failure here is almost always a
    # non-fast-forward because the vault was also edited on another machine —
    # that wants a human deciding which side wins, not a scheduled job
    # overwriting one of them. The commit is already safe locally, so nothing
    # is lost by stopping.
    push = _git(path, "push")
    if push.returncode != 0:
        logger.error(f"git push failed: {push.stderr.strip()}")
        logger.error(f"Commit is safe locally; resolve by hand in {path}")
        return 1

    upstream = _git(path, "rev-parse", "--abbrev-ref", "@{upstream}").stdout.strip()
    logger.info(f"Pushed to {upstream}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vault", required=True, help="Path to the Obsidian vault.")
    args = parser.parse_args()

    vault_name = Path(args.vault).name
    logger = setup_logger(f"vault_snapshot.{vault_name}")

    try:
        return snapshot_vault(args.vault, logger)
    except Exception as e:
        logger.exception(f"Vault snapshot failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
