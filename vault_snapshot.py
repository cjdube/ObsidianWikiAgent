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
    python vault_snapshot.py --vault ~/Vaults/llm-wiki-learnings
"""

import argparse
import os
import subprocess
import sys
from datetime import date
from pathlib import Path
from typing import Optional

from agent.common import setup_logger, trim_launchd_log
from agent.notify import notify_failure


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


def snapshot_vault(vault: str, logger) -> Optional[str]:
    """Commit and push the vault. Returns None on success, or a short reason on
    failure — main() turns that reason into the exit code and the ntfy alert, so
    the phone push says what broke instead of only that something did."""
    path = Path(vault).expanduser()

    if not path.is_dir():
        reason = f"No such directory: {path}"
        logger.error(reason)
        return reason
    if _git(path, "rev-parse", "--git-dir").returncode != 0:
        reason = f"Not a git repository: {path}"
        logger.error(reason)
        return reason

    add = _git(path, "add", "-A")
    if add.returncode != 0:
        reason = f"git add failed: {add.stderr.strip()}"
        logger.error(reason)
        return reason

    staged = _git(path, "diff", "--cached", "--name-only").stdout.splitlines()
    if not staged:
        logger.info(f"No changes in {path}")
        return None

    message = f"Vault snapshot {date.today():%Y-%m-%d}: {len(staged)} files"
    commit = _git(path, "commit", "-m", message)
    if commit.returncode != 0:
        reason = f"git commit failed: {commit.stderr.strip()}"
        logger.error(reason)
        return reason
    logger.info(f"Committed {len(staged)} files")

    # Plain push, never --force. A failure here is almost always a
    # non-fast-forward because the vault was also edited on another machine —
    # that wants a human deciding which side wins, not a scheduled job
    # overwriting one of them. The commit is already safe locally, so nothing
    # is lost by stopping. Leaving that decision to a human only works if the
    # human is told, which is why this reason becomes an alert.
    push = _git(path, "push")
    if push.returncode != 0:
        reason = f"git push failed: {push.stderr.strip()}"
        logger.error(reason)
        logger.error(f"Commit is safe locally; resolve by hand in {path}")
        return reason

    upstream = _git(path, "rev-parse", "--abbrev-ref", "@{upstream}").stdout.strip()
    logger.info(f"Pushed to {upstream}")
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vault", required=True, help="Path to the Obsidian vault.")
    args = parser.parse_args()

    vault_name = Path(args.vault).name
    logger = setup_logger(f"vault_snapshot.{vault_name}")
    trim_launchd_log(logger)

    # The two markers a run is bounded by. Without them this job's log is a flat
    # stream of one-off lines: readable, but nothing can tell where one nightly
    # run ends and the next begins — so LocalLLMAgent's dashboard, which reports
    # on this repo's launchd jobs, showed it as zero runs. (See its
    # docs/external-tasks.md; the ingest has had these markers all along.)
    logger.info(f"Starting vault snapshot run for vault: {args.vault}")
    # Alerting, like the ingest has: this job failed every night from 2026-07-25
    # to 2026-08-03 and nobody noticed, because its only symptom was an ERROR
    # line in a log with no reason to be opened. A backup that stops backing up
    # silently is the failure mode worth paying a push notification for.
    job = f"vault_snapshot[{vault_name}]"
    try:
        failure = snapshot_vault(args.vault, logger)
    except Exception as e:
        logger.exception(f"Vault snapshot failed: {e}")
        notify_failure(job, e, logger)
        return 1
    # Conditional, not unconditional: a "run complete" line after a failed push
    # would close the run as a success and paint over the ERROR lines above it.
    if failure is None:
        logger.info("Vault snapshot run complete")
        return 0
    logger.error("Vault snapshot run failed")
    notify_failure(job, failure, logger)
    return 1


if __name__ == "__main__":
    sys.exit(main())
