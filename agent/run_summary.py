"""What one ingest run did, collected as it happens and rendered at the end.

The run already knows all of this. wiki_ingest splits its planned pages into
created and updated to write the log entry, counts the sources it processed to
pick an exit code, and takes a start time it only reads when the budget blows —
and then throws every one of those away, leaving a single "Wiki ingest run
complete" line in a log nobody opens.

So this is an accumulator, not a log parser. tools/measure_stage2.py parses
logs because it had to reconstruct runs that were already over; anything asked
of a run while it is still running should be recorded rather than re-derived,
because a parser goes stale the moment a log string is reworded.

The rendered note lands at <vault>/runs/YYYY-MM-DD.md — beside the wiki, not
inside it. Every tool the model is given resolves under raw/ or wiki/
(agent/wiki_tools.py:54-58), and wiki_lint walks wiki/ only, so nothing here is
ever visible to the model, to the index, or to a lint pass.
"""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from agent.wiki_tools import atomic_write

# Per-source outcomes. The distinction that matters to a human reading the note
# is not "did pages get written" but "is this source finished" — an unfinished
# source is silently retried on the next scheduled run, and a note that did not
# say so would read like a success.
OK = "ok"                  # every planned page landed and the log entry was written
PARTIAL = "partial"        # some planned pages failed
NO_PLAN = "no-plan"        # stage 1 never produced a usable plan
LOG_FAILED = "log-failed"  # pages landed, stage 3 did not
ABANDONED = "abandoned"    # the run's budget expired while this source was mid-flight

_RETRIED = {PARTIAL, NO_PLAN, LOG_FAILED, ABANDONED}


@dataclass
class SourceResult:
    """One raw source's trip through the three stages."""

    filename: str
    status: str = OK
    created: list[str] = field(default_factory=list)
    updated: list[str] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)
    skipped: str = ""

    @property
    def retried(self) -> bool:
        return self.status in _RETRIED


@dataclass
class RunSummary:
    """The whole run. Built in main() so it survives every exit path."""

    vault: str
    started_at: datetime
    budget_minutes: float
    # None until the run has counted the queue. wiki_ingest sorts raw/, loads
    # rules and lists files before it can set this, and any of those can raise;
    # a run that dies in that window has not found zero pending sources, it has
    # not looked. The two readings send a reader to different repositories, so
    # the note must not collapse them — see render_markdown.
    pending: int | None = None
    binaries: list[str] = field(default_factory=list)
    sources: list[SourceResult] = field(default_factory=list)
    outcome: str = "running"  # "complete" | "abandoned" | "failed"
    detail: str = ""
    elapsed_minutes: float = 0.0
    # Set only by tools/backfill_run_notes.py, which rebuilds the notes for runs
    # that finished before this module existed. A reconstructed note is not the
    # same evidence as a recorded one — it is a reading of a log, and the log
    # does not carry every field — so it says so on its face rather than sitting
    # beside the real ones looking identical.
    reconstructed_from: str = ""

    def add_source(self, result: SourceResult) -> None:
        self.sources.append(result)

    @property
    def created(self) -> list[str]:
        return [name for s in self.sources for name in s.created]

    @property
    def updated(self) -> list[str]:
        return [name for s in self.sources for name in s.updated]

    @property
    def failed(self) -> list[str]:
        return [name for s in self.sources for name in s.failed]

    @property
    def ingested(self) -> int:
        """Sources that are finished and will not come back tomorrow."""
        return sum(1 for s in self.sources if s.status == OK)

    @property
    def unreached(self) -> int:
        """Pending sources the run never started. Only meaningful once the run
        has stopped early — a completed run reaches everything."""
        if self.pending is None:
            return 0
        return max(0, self.pending - len(self.sources))

    def _headline(self) -> str:
        # "0 of ?" rather than "0 of 0": the run never reached the count, and a
        # zero here would read as a total the run had actually established.
        total = "?" if self.pending is None else self.pending
        return (
            f"{self.outcome} in {self.elapsed_minutes:.1f} min — "
            f"{self.ingested} of {total} source(s) ingested, "
            f"{len(self.created)} page(s) created, "
            f"{len(self.updated)} page(s) updated, "
            f"{len(self.failed)} page(s) failed"
        )

    def render_markdown(self) -> str:
        """One `## HH:MM` block, ready to append to the day's note."""
        lines = [f"## {self.started_at:%H:%M} — {self._headline()}", ""]

        if self.detail:
            lines += [f"> {self.detail}", ""]

        if self.reconstructed_from:
            lines += [
                f"*Reconstructed from `{self.reconstructed_from}`, not recorded "
                "by the run itself. Out-of-scope notes are not in the log and "
                "are missing here.*",
                "",
            ]

        if self.outcome != "complete" and self.unreached:
            lines += [
                f"{self.unreached} pending source(s) were never started, and "
                "stay queued for the next run.",
                "",
            ]

        if self.pending is None:
            # The run died before it counted the queue — during the raw/ sort,
            # the rules load, or the file listing. Saying "nothing was pending"
            # here would send the reader to the upstream job that fills raw/,
            # when the fault is in this run and its reason is the line above.
            lines += [
                "The run stopped before it counted the pending sources, so the "
                "queue is unknown. The reason above is where to look.",
                "",
            ]
        elif not self.pending:
            # Worth saying rather than leaving the note empty. On a vault whose
            # raw/ is filled by a scheduled job, nothing pending means that job
            # did not run — a silent note would hide it.
            lines += ["Nothing was pending. No raw sources awaited ingest.", ""]

        for s in self.sources:
            lines.append(f"### {s.filename}")
            if s.created:
                lines.append(f"- created: {', '.join(s.created)}")
            if s.updated:
                lines.append(f"- updated: {', '.join(s.updated)}")
            if s.failed:
                lines.append(f"- failed: {', '.join(s.failed)}")
            if s.skipped:
                lines.append(f"- out of scope: {s.skipped}")
            if s.status == NO_PLAN:
                lines.append("- no usable plan — nothing was written")
            elif s.status == LOG_FAILED:
                lines.append("- pages landed, but the log entry did not")
            elif s.status == ABANDONED:
                lines.append("- the run stopped here, with this source part-done")
            if s.retried:
                lines.append("- left unmarked, so the next run redoes it")
            lines.append("")

        if self.binaries:
            lines += [
                "### Skipped binaries",
                f"- {', '.join(self.binaries)}",
                "- these need OCR or a vision model, not a text read",
                "",
            ]

        return "\n".join(lines).rstrip() + "\n"

    def render_log_block(self) -> str:
        """The same facts for the run log, as one multi-line record.

        Logged as a single INFO so it carries one timestamp. That also keeps it
        away from tools/measure_stage2.py, which skips any line without a
        leading timestamp — only the first line below is visible to it, and
        none of these lines carry the phrases it anchors runs and sources on.
        """
        lines = [f"Run summary: {self._headline()}"]
        if self.detail:
            lines.append(f"  {self.detail}")
        for s in self.sources:
            parts = []
            if s.created:
                parts.append(f"created {', '.join(s.created)}")
            if s.updated:
                parts.append(f"updated {', '.join(s.updated)}")
            if s.failed:
                parts.append(f"failed {', '.join(s.failed)}")
            # Settled before the retry marker is added, so a source that wrote
            # nothing still says so instead of reading as "retried next run".
            wrote = "; ".join(parts) or "nothing written"
            if s.retried:
                wrote += "; retried next run"
            lines.append(f"  {s.filename}: {wrote}")
        return "\n".join(lines)


def write_run_note(vault_path: str, summary: RunSummary) -> Path:
    """Append this run's block to <vault>/runs/YYYY-MM-DD.md and return the path.

    Appends rather than overwrites because a repair run on the same day is
    normal, and the first run of the day is the one you would least want to
    lose. Rewritten whole through atomic_write for the reason that helper
    exists: the budget watchdog raises from a SIGALRM handler at whatever point
    the process happens to be, including inside this write.
    """
    runs_dir = Path(vault_path) / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    path = runs_dir / f"{summary.started_at:%Y-%m-%d}.md"

    if path.exists():
        existing = path.read_text(encoding="utf-8").rstrip() + "\n\n"
    else:
        existing = f"# Ingest runs — {summary.started_at:%Y-%m-%d}\n\n"

    atomic_write(path, existing + summary.render_markdown())
    return path
