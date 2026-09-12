#!/usr/bin/env python3
"""Rebuild <vault>/runs/<date>.md for runs that finished before run notes existed.

Run it:

    .venv/bin/python tools/backfill_run_notes.py --vault ~/Vaults/llm-wiki-learnings \\
        --since 2026-09-05 --dry-run

Then drop --dry-run to write.

WHY THIS IS A PARSER WHEN agent/run_summary.py DELIBERATELY IS NOT

agent/run_summary.py accumulates facts while a run is still running, because a
log parser goes stale the moment a log string is reworded. That argument holds
for every future run and not for a single past one: these runs are over, they
recorded nothing, and the log is the only evidence left. So this is a one-off
backfill, not a second way of producing notes. It is not scheduled, and a run
from now on writes its own note.

WHAT THE LOG CANNOT GIVE BACK

- The out-of-scope text (`plan.skipped`). It reaches the log only inside the
  repr of the submit_plan tool call, mixed in with page content that carries
  its own quotes and brackets. Parsing that is guesswork, so it is left out and
  every reconstructed note says so.
- Runs whose log was rotated away, or that have no end-of-run line. Those are
  skipped and named on stderr rather than guessed at.

WHAT IT READS, AND WHY EACH LINE IS TRUSTWORTHY

- `Starting wiki ingest run for vault: X (budget N min)` — start and budget.
- `Skipping N binary source(s) ... : a.pdf, b.pdf` — the binaries.
- `Ingesting 'X'` — one per pending source, in queue order.
- `[INFO] tool_call write_wiki_page(` / `edit_wiki_page(` — create versus
  update. The *plan's* action is a guess that _execute_unit overwrites from
  disk, and the "Plan for" line is logged before that correction, so the plan
  line is not usable for this. The dispatched tool is: a page that exists is
  given only the edit path and a new one only the create path. Error results
  are logged at WARNING, so only INFO lines count.
- `Wrote 'p' for 'X'` — the write landed; the tool seen just above it says
  which kind it was. `No write from 'p' for 'X'` is a spent attempt and is
  ignored, because a later attempt for the same page may still have landed.
- `'X': n/m planned page(s) written; failed on a, b` — the failed pages.
- `No usable plan for 'X'` — stage 1 never produced one.
- `'X' was not fully ingested` — with no failed pages above it, this is stage 3
  failing after the pages landed.
- `Run stopped after n/m source(s)` — how many sources were pending in total,
  which nothing else in the log reports for an abandoned run.
- `Wiki ingest run complete` / `abandoned after N min:` / `failed:` — the end.
"""

import argparse
import re
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.run_summary import (  # noqa: E402
    ABANDONED,
    LOG_FAILED,
    NO_PLAN,
    OK,
    PARTIAL,
    RunSummary,
    SourceResult,
    write_run_note,
)

_TS = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d{3} \[(\w+)\] (.*)$")

_START = re.compile(r"^Starting wiki ingest run for vault: (\S+) \(budget ([\d.]+) min\)")
_BINARIES = re.compile(r"^Skipping \d+ binary source\(s\) in raw/ — [^:]+: (.+)$")
_INGESTING = re.compile(r"^Ingesting '(.+?)'$")
_TOOL = re.compile(r"^tool_call (write_wiki_page|edit_wiki_page)\(")
_WROTE = re.compile(r"^Wrote '(.+?)' for '(.+?)': ")
_NO_PLAN = re.compile(r"^No usable plan for '(.+?)'")
_PARTIAL = re.compile(r"^'(.+?)': \d+/\d+ planned page\(s\) written; failed on (.+?)\.")
_UNMARKED = re.compile(r"^'(.+?)' was not fully ingested")
_STOPPED = re.compile(r"^Run stopped after (\d+)/(\d+) source\(s\)")
_COMPLETE = re.compile(r"^Wiki ingest run complete$")
_ABANDONED = re.compile(r"^Wiki ingest run abandoned after [\d.]+ min: (.+)$")
_FAILED = re.compile(r"^Wiki ingest run failed: (.+)$")


class _Run:
    """One run being assembled, before it becomes a RunSummary."""

    def __init__(self, started_at: datetime, budget_minutes: float):
        self.started_at = started_at
        self.budget_minutes = budget_minutes
        self.ended_at: datetime | None = None
        self.outcome = ""
        self.detail = ""
        self.binaries: list[str] = []
        self.pending: int | None = None
        self.order: list[str] = []
        self.sources: dict[str, SourceResult] = {}

    def source(self, filename: str) -> SourceResult:
        if filename not in self.sources:
            self.order.append(filename)
            self.sources[filename] = SourceResult(filename=filename)
        return self.sources[filename]


def parse(lines) -> list[_Run]:
    """Every complete run in the given lines, oldest first.

    A run with no end-of-run line is dropped by the caller, not here — this
    keeps what it saw so the caller can say which run it is throwing away.
    """
    runs: list[_Run] = []
    run: _Run | None = None
    # The write tool dispatched most recently at INFO. Stage 2 gives a page one
    # tool or the other, so this is what decides created versus updated.
    tool = ""

    for line in lines:
        m = _TS.match(line.rstrip("\n"))
        if not m:
            continue  # a continuation line of a multi-line record
        stamp, level, body = m.group(1), m.group(2), m.group(3)
        when = datetime.strptime(stamp, "%Y-%m-%d %H:%M:%S")

        start = _START.match(body)
        if start:
            run = _Run(when, float(start.group(2)))
            runs.append(run)
            tool = ""
            continue

        if run is None:
            continue  # a rotated log that begins mid-run

        if level == "INFO":
            found = _TOOL.match(body)
            if found:
                # Only INFO. A refused call is logged at WARNING and wrote
                # nothing, and counting it would name the wrong kind of write.
                tool = found.group(1)
                continue

        wrote = _WROTE.match(body)
        if wrote:
            page, filename = wrote.group(1), wrote.group(2)
            source = run.source(filename)
            target = source.created if tool == "write_wiki_page" else source.updated
            if page not in target:
                target.append(page)
            tool = ""
            continue

        ingesting = _INGESTING.match(body)
        if ingesting:
            run.source(ingesting.group(1))
            tool = ""
            continue

        binaries = _BINARIES.match(body)
        if binaries:
            run.binaries = [b.strip() for b in binaries.group(1).split(",")]
            continue

        no_plan = _NO_PLAN.match(body)
        if no_plan:
            run.source(no_plan.group(1)).status = NO_PLAN
            continue

        partial = _PARTIAL.match(body)
        if partial:
            source = run.source(partial.group(1))
            source.status = PARTIAL
            source.failed = [n.strip() for n in partial.group(2).split(",")]
            continue

        unmarked = _UNMARKED.match(body)
        if unmarked:
            source = run.source(unmarked.group(1))
            # PARTIAL is logged first and is the more specific reason. Reaching
            # here with an untouched status means every planned page landed and
            # the log entry is what did not.
            if source.status == OK:
                source.status = LOG_FAILED
            continue

        stopped = _STOPPED.match(body)
        if stopped:
            run.pending = int(stopped.group(2))
            # The source in flight when the budget blew. Its pages are already
            # collected above; only its standing changes.
            if run.order:
                run.sources[run.order[-1]].status = ABANDONED
            continue

        if _COMPLETE.match(body):
            run.outcome, run.ended_at = "complete", when
            continue

        abandoned = _ABANDONED.match(body)
        if abandoned:
            run.outcome, run.detail, run.ended_at = "abandoned", abandoned.group(1), when
            continue

        failed = _FAILED.match(body)
        if failed:
            run.outcome, run.detail, run.ended_at = "failed", failed.group(1), when

    return runs


def to_summary(run: _Run, vault_path: str, log_name: str) -> RunSummary:
    summary = RunSummary(
        vault=vault_path,
        started_at=run.started_at,
        budget_minutes=run.budget_minutes,
        # A completed run reached every pending source, so the count of
        # "Ingesting" lines is the whole queue. An abandoned one did not, and
        # its "Run stopped after n/m" line is the only place m appears.
        pending=run.pending if run.pending is not None else len(run.order),
        binaries=run.binaries,
        outcome=run.outcome,
        detail=run.detail,
        elapsed_minutes=(run.ended_at - run.started_at).total_seconds() / 60,
        reconstructed_from=log_name,
    )
    for filename in run.order:
        summary.add_source(run.sources[filename])
    return summary


def _logs(vault_path: str, given: list[str]) -> list[str]:
    """The log files for this vault, newest content last.

    Rotations are .log.1, .log.2 and so on, oldest at the highest number, so
    they are read in reverse before the live file.
    """
    if given:
        return given
    name = Path(vault_path).name
    live = Path("logs") / f"wiki_ingest.{name}.log"
    rotated = sorted(Path("logs").glob(f"wiki_ingest.{name}.log.*"), reverse=True)
    return [str(p) for p in [*rotated, live] if p.exists()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--vault", required=True, help="Vault to write runs/ into.")
    parser.add_argument("--log", action="append", default=[], help="Log file to read.")
    parser.add_argument("--since", help="Earliest run date to rebuild (YYYY-MM-DD).")
    parser.add_argument("--until", help="Latest run date to rebuild (YYYY-MM-DD).")
    parser.add_argument(
        "--dry-run", action="store_true", help="Print the notes; write nothing."
    )
    args = parser.parse_args()

    logs = _logs(args.vault, args.log)
    if not logs:
        print(f"No log files found for vault '{args.vault}'.", file=sys.stderr)
        return 1

    runs: list[tuple[_Run, str]] = []
    for path in logs:
        with open(path, encoding="utf-8", errors="replace") as handle:
            for run in parse(handle):
                runs.append((run, Path(path).name))

    kept: list[tuple[_Run, str]] = []
    for run, log_name in runs:
        day = run.started_at.date().isoformat()
        if args.since and day < args.since:
            continue
        if args.until and day > args.until:
            continue
        if run.ended_at is None:
            print(
                f"Skipping the run of {run.started_at}: the log has no "
                "end-of-run line, so how it ended is not recoverable.",
                file=sys.stderr,
            )
            continue
        kept.append((run, log_name))

    if not kept:
        print("No runs matched.", file=sys.stderr)
        return 1

    runs_dir = Path(args.vault) / "runs"
    written, skipped = [], []
    for run, log_name in kept:
        summary = to_summary(run, args.vault, log_name)
        path = runs_dir / f"{summary.started_at:%Y-%m-%d}.md"
        if args.dry_run:
            print(f"--- would append to {path} ---")
            print(summary.render_markdown())
            continue
        # A note that is already there was either written by the run itself or
        # by an earlier backfill. Appending would duplicate it, and this script
        # has no business overwriting a note a real run produced.
        if path.exists() and path not in written:
            skipped.append(path)
            continue
        write_run_note(args.vault, summary)
        if path not in written:
            written.append(path)

    for path in skipped:
        print(f"Left alone (already there): {path}", file=sys.stderr)
    for path in written:
        print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
