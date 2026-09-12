"""Guards tools/backfill_run_notes.py.

The backfill exists because runs that finished before agent/run_summary.py have
no record but the log. Every rule below is a place where the log can be read
two ways and only one of them is true, so each test pins the reading against a
line the real ingest actually emits.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
import backfill_run_notes as b  # noqa: E402

from agent.run_summary import ABANDONED, LOG_FAILED, NO_PLAN, OK, PARTIAL  # noqa: E402

START = (
    "2026-09-11 09:00:00,000 [INFO] Starting wiki ingest run for vault: "
    "/Users/x/Vaults/demo (budget 45 min)\n"
)


def _lines(*body: str) -> list[str]:
    return [START, *body]


def _write(page: str, source: str, at="09:01:00") -> str:
    return f"2026-09-11 {at},000 [INFO] tool_call write_wiki_page({{'name': '{page}'}}) -> ok\n"


def _edit(page: str, at="09:01:00") -> str:
    return f"2026-09-11 {at},000 [INFO] tool_call edit_wiki_page({{'name': '{page}'}}) -> ok\n"


def _wrote(page: str, source: str, at="09:01:01") -> str:
    return f"2026-09-11 {at},000 [INFO] Wrote '{page}' for '{source}': done\n"


def _ingesting(source: str, at="09:00:30") -> str:
    return f"2026-09-11 {at},000 [INFO] Ingesting '{source}'\n"


COMPLETE = "2026-09-11 09:10:00,000 [INFO] Wiki ingest run complete\n"


def _one(lines):
    (run,) = b.parse(lines)
    return run


# --- create versus update --------------------------------------------------


def test_the_dispatched_tool_decides_create_or_update_not_the_plan():
    """_execute_unit overwrites the plan's action from disk, and the 'Plan for'
    line is logged before that correction. So a page the plan called 'create'
    can be an edit, and only the tool that was actually dispatched knows."""
    run = _one(_lines(
        _ingesting("s.md"),
        "2026-09-11 09:00:40,000 [INFO] Plan for 's.md': 2 page(s) — "
        "alpha (create), beta (create)\n",
        _write("alpha", "s.md"),
        _wrote("alpha", "s.md"),
        _edit("beta"),
        _wrote("beta", "s.md"),
        COMPLETE,
    ))

    assert run.sources["s.md"].created == ["alpha"]
    assert run.sources["s.md"].updated == ["beta"]


def test_a_refused_tool_call_does_not_decide_the_kind():
    """A call that comes back an error is logged at WARNING and wrote nothing.
    Letting it set the kind would label the next landed page from the wrong
    tool."""
    run = _one(_lines(
        _ingesting("s.md"),
        _write("alpha", "s.md", at="09:01:00"),
        "2026-09-11 09:01:02,000 [WARNING] tool_call edit_wiki_page({'name': 'alpha'}) "
        '-> {"error": "not this step\'s page"}\n',
        _wrote("alpha", "s.md", at="09:01:04"),
        COMPLETE,
    ))

    assert run.sources["s.md"].created == ["alpha"]
    assert run.sources["s.md"].updated == []


def test_a_retried_page_is_counted_once():
    """'No write from' is a spent attempt, not a result. The page is retried in
    the same step and may still land, so the attempt must not be read as a
    failure and the landing must not be counted twice."""
    run = _one(_lines(
        _ingesting("s.md"),
        "2026-09-11 09:01:00,000 [INFO] No write from 'alpha' for 's.md': prose\n",
        _write("alpha", "s.md", at="09:01:10"),
        _wrote("alpha", "s.md", at="09:01:11"),
        _write("alpha", "s.md", at="09:01:20"),
        _wrote("alpha", "s.md", at="09:01:21"),
        COMPLETE,
    ))

    assert run.sources["s.md"].created == ["alpha"]


# --- how many sources were pending -----------------------------------------


def test_a_completed_run_counts_its_pending_sources_from_the_queue():
    """Nothing logs the pending total for a run that finished, but a completed
    run reached every source, so the 'Ingesting' lines are the whole queue."""
    run = _one(_lines(_ingesting("a.md"), _ingesting("b.md"), COMPLETE))

    summary = b.to_summary(run, "/v", "x.log")
    assert summary.pending == 2


def test_an_abandoned_run_takes_its_pending_total_from_the_stop_line():
    """A run that stopped early never reached the rest of its queue, so the
    'Ingesting' lines undercount it. 'Run stopped after n/m' is the only place
    m appears."""
    run = _one(_lines(
        _ingesting("a.md"),
        _write("alpha", "a.md"),
        _wrote("alpha", "a.md"),
        "2026-09-11 09:45:00,000 [ERROR] Run stopped after 1/4 source(s): run budget "
        "of 45 min expired. The remaining sources stay unmarked.\n",
        "2026-09-11 09:45:00,100 [ERROR] Wiki ingest run abandoned after 45.0 min: "
        "run budget of 45 min expired\n",
    ))

    summary = b.to_summary(run, "/v", "x.log")
    assert summary.pending == 4
    assert summary.unreached == 3
    assert summary.outcome == "abandoned"
    assert "run budget of 45 min expired" in summary.detail


def test_the_source_in_flight_when_the_budget_blew_keeps_its_pages():
    """Same reason the live accumulator has an ABANDONED status: the pages it
    already wrote are on disk and must not vanish from the note."""
    run = _one(_lines(
        _ingesting("a.md"),
        _write("alpha", "a.md"),
        _wrote("alpha", "a.md"),
        "2026-09-11 09:45:00,000 [ERROR] Run stopped after 0/2 source(s): spent\n",
        "2026-09-11 09:45:00,100 [ERROR] Wiki ingest run abandoned after 45.0 min: spent\n",
    ))

    assert run.sources["a.md"].status == ABANDONED
    assert run.sources["a.md"].created == ["alpha"]


# --- the ways a source stays unfinished ------------------------------------


def test_a_source_with_no_usable_plan_is_marked():
    run = _one(_lines(
        _ingesting("a.md"),
        "2026-09-11 09:02:00,000 [WARNING] No usable plan for 'a.md' after 3 "
        "attempts — nothing was written.\n",
        COMPLETE,
    ))

    assert run.sources["a.md"].status == NO_PLAN


def test_failed_pages_come_from_the_line_that_names_them():
    run = _one(_lines(
        _ingesting("a.md"),
        _write("alpha", "a.md"),
        _wrote("alpha", "a.md"),
        "2026-09-11 09:05:00,000 [WARNING] 'a.md': 1/3 planned page(s) written; "
        "failed on beta, gamma. Leaving the source unmarked so the next run redoes it.\n",
        COMPLETE,
    ))

    source = run.sources["a.md"]
    assert source.status == PARTIAL
    assert source.failed == ["beta", "gamma"]


def test_an_unmarked_source_with_no_failed_pages_is_a_log_failure():
    """Stage 3 is the non-idempotent one. Its failure leaves real pages and no
    log.md entry, which reads nothing like a page failing."""
    run = _one(_lines(
        _ingesting("a.md"),
        _write("alpha", "a.md"),
        _wrote("alpha", "a.md"),
        "2026-09-11 09:05:00,000 [WARNING] 'a.md' was not fully ingested — leaving "
        "it unmarked so the next run retries it.\n",
        COMPLETE,
    ))

    assert run.sources["a.md"].status == LOG_FAILED


def test_a_partial_source_is_not_downgraded_by_the_unmarked_line():
    """Both lines are logged for the same source, PARTIAL first. It is the more
    specific reason and has the failed page names attached."""
    run = _one(_lines(
        _ingesting("a.md"),
        "2026-09-11 09:05:00,000 [WARNING] 'a.md': 0/1 planned page(s) written; "
        "failed on beta.\n",
        "2026-09-11 09:05:01,000 [WARNING] 'a.md' was not fully ingested — leaving "
        "it unmarked.\n",
        COMPLETE,
    ))

    assert run.sources["a.md"].status == PARTIAL


# --- whole runs ------------------------------------------------------------


def test_a_run_that_ingested_nothing_still_parses():
    run = _one(_lines(
        "2026-09-11 09:00:06,000 [INFO] Nothing to ingest — all raw sources "
        "already processed.\n",
        COMPLETE,
    ))

    summary = b.to_summary(run, "/v", "x.log")
    assert summary.pending == 0
    assert "Nothing was pending" in summary.render_markdown()


def test_binaries_are_carried_over():
    run = _one(_lines(
        "2026-09-11 09:00:06,000 [INFO] Skipping 2 binary source(s) in raw/ — these "
        "need OCR or a vision model, not a text read: scan.pdf, photo.png\n",
        _ingesting("a.md"),
        COMPLETE,
    ))

    assert run.binaries == ["scan.pdf", "photo.png"]


def test_lines_before_the_first_run_header_are_dropped():
    """A rotated log starts mid-run. Those sources belong to a run whose start
    time and budget are gone, and attributing them to the next run would put
    yesterday's work in today's note."""
    runs = b.parse([
        _ingesting("orphan.md"),
        _wrote("orphan", "orphan.md"),
        START,
        _ingesting("real.md"),
        COMPLETE,
    ])

    (run,) = runs
    assert list(run.sources) == ["real.md"]


def test_a_run_with_no_end_line_is_left_incomplete():
    """The process was killed, or the log was cut. How it ended is not
    recoverable, and main() drops it rather than guessing."""
    run = _one(_lines(_ingesting("a.md")))

    assert run.ended_at is None
    assert run.outcome == ""


def test_elapsed_is_measured_between_the_first_and_last_line_of_the_run():
    run = _one(_lines(_ingesting("a.md"), COMPLETE))

    assert b.to_summary(run, "/v", "x.log").elapsed_minutes == pytest.approx(10.0)


def test_the_note_says_it_was_reconstructed():
    """A backfilled note sits in the same folder as the real ones. Without this
    line a reader would take a reading of a log for a record of a run."""
    run = _one(_lines(_ingesting("a.md"), COMPLETE))

    note = b.to_summary(run, "/v", "wiki_ingest.demo.log").render_markdown()
    assert "Reconstructed from `wiki_ingest.demo.log`" in note
    assert "Out-of-scope notes are not in the log" in note


# --- writing ---------------------------------------------------------------


def _log_file(tmp_path, body):
    path = tmp_path / "wiki_ingest.demo.log"
    path.write_text("".join(body), encoding="utf-8")
    return str(path)


def test_a_note_that_already_exists_is_left_alone(vault, monkeypatch, tmp_path, capsys):
    """A run from now on writes its own note. Appending to one would duplicate
    a block, and overwriting one would replace a record with a reading."""
    runs_dir = Path(vault.path) / "runs"
    runs_dir.mkdir(parents=True)
    (runs_dir / "2026-09-11.md").write_text("# Ingest runs — 2026-09-11\n", encoding="utf-8")

    log = _log_file(tmp_path, _lines(_ingesting("a.md"), COMPLETE))
    monkeypatch.setattr("sys.argv", [
        "backfill_run_notes.py", "--vault", vault.path, "--log", log,
    ])

    assert b.main() == 0
    assert (runs_dir / "2026-09-11.md").read_text(encoding="utf-8") == (
        "# Ingest runs — 2026-09-11\n"
    )
    assert "Left alone" in capsys.readouterr().err


def test_two_runs_on_one_day_both_land_in_that_day_s_note(vault, monkeypatch, tmp_path):
    second = (
        "2026-09-11 14:00:00,000 [INFO] Starting wiki ingest run for vault: "
        "/Users/x/Vaults/demo (budget 45 min)\n"
        "2026-09-11 14:00:30,000 [INFO] Ingesting 'b.md'\n"
        "2026-09-11 14:05:00,000 [INFO] Wiki ingest run complete\n"
    )
    log = _log_file(tmp_path, [*_lines(_ingesting("a.md"), COMPLETE), second])
    monkeypatch.setattr("sys.argv", [
        "backfill_run_notes.py", "--vault", vault.path, "--log", log,
    ])

    assert b.main() == 0

    text = (Path(vault.path) / "runs" / "2026-09-11.md").read_text(encoding="utf-8")
    assert text.count("# Ingest runs — 2026-09-11") == 1
    assert text.index("## 09:00") < text.index("## 14:00")
    assert "a.md" in text and "b.md" in text


def test_the_note_lands_beside_the_wiki_not_inside_it(vault, monkeypatch, tmp_path):
    log = _log_file(tmp_path, _lines(_ingesting("a.md"), COMPLETE))
    monkeypatch.setattr("sys.argv", [
        "backfill_run_notes.py", "--vault", vault.path, "--log", log,
    ])

    assert b.main() == 0
    assert (Path(vault.path) / "runs" / "2026-09-11.md").is_file()
    assert not (Path(vault.path) / "wiki" / "runs").exists()


def test_a_dry_run_writes_nothing(vault, monkeypatch, tmp_path, capsys):
    log = _log_file(tmp_path, _lines(_ingesting("a.md"), COMPLETE))
    monkeypatch.setattr("sys.argv", [
        "backfill_run_notes.py", "--vault", vault.path, "--log", log, "--dry-run",
    ])

    assert b.main() == 0
    assert not (Path(vault.path) / "runs").exists()
    assert "## 09:00" in capsys.readouterr().out


def test_since_and_until_bound_what_is_rebuilt(vault, monkeypatch, tmp_path):
    later = (
        "2026-09-12 09:00:00,000 [INFO] Starting wiki ingest run for vault: "
        "/Users/x/Vaults/demo (budget 45 min)\n"
        "2026-09-12 09:00:30,000 [INFO] Ingesting 'b.md'\n"
        "2026-09-12 09:05:00,000 [INFO] Wiki ingest run complete\n"
    )
    log = _log_file(tmp_path, [*_lines(_ingesting("a.md"), COMPLETE), later])
    monkeypatch.setattr("sys.argv", [
        "backfill_run_notes.py", "--vault", vault.path, "--log", log,
        "--since", "2026-09-12",
    ])

    assert b.main() == 0

    names = sorted(p.name for p in (Path(vault.path) / "runs").glob("*.md"))
    assert names == ["2026-09-12.md"]
