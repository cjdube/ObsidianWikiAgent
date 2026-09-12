"""Tests for the run summary in agent/run_summary.py.

These cover what the note says and where it lands. What a run actually puts
into the summary is covered in tests/test_wiki_ingest.py, against a real
ingest_vault call.
"""

import sys
from datetime import datetime
from pathlib import Path

import pytest

from agent.run_summary import (
    LOG_FAILED,
    NO_PLAN,
    OK,
    PARTIAL,
    RunSummary,
    SourceResult,
    write_run_note,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
import measure_stage2 as m  # noqa: E402


def _summary(**kw):
    base = dict(
        vault="/vaults/demo",
        started_at=datetime(2026, 9, 12, 5, 0),
        budget_minutes=45,
        pending=1,
        outcome="complete",
        elapsed_minutes=8.6,
    )
    base.update(kw)
    return RunSummary(**base)


def _source(**kw):
    base = dict(filename="Daily-2026-09-11.md", status=OK)
    base.update(kw)
    return SourceResult(**base)


# --- what the note says ----------------------------------------------------


def test_the_headline_carries_the_counts_a_run_used_to_throw_away():
    """The whole point of the feature: "Wiki ingest run complete" replaced by
    the numbers the run already had in hand."""
    s = _summary(pending=2)
    s.add_source(_source(created=["alpha", "beta"], updated=["gamma"]))
    s.add_source(_source(filename="Other.md", updated=["delta"]))

    note = s.render_markdown()
    assert "## 05:00 — complete in 8.6 min" in note
    assert "2 of 2 source(s) ingested" in note
    assert "2 page(s) created" in note
    assert "2 page(s) updated" in note
    assert "0 page(s) failed" in note


def test_created_and_updated_are_listed_per_source():
    s = _summary()
    s.add_source(_source(created=["alpha"], updated=["beta", "gamma"]))

    note = s.render_markdown()
    assert "### Daily-2026-09-11.md" in note
    assert "- created: alpha" in note
    assert "- updated: beta, gamma" in note


def test_out_of_scope_text_is_kept():
    """plan.skipped is the model's own account of what it refused, and it is
    the only place a reader learns a source was partly ignored on purpose."""
    s = _summary()
    s.add_source(_source(created=["alpha"], skipped="two recipe links"))

    assert "- out of scope: two recipe links" in s.render_markdown()


def test_binaries_are_reported_with_the_reason_they_were_skipped():
    s = _summary(binaries=["scan.pdf"])
    assert "- scan.pdf" in s.render_markdown()
    assert "OCR" in s.render_markdown()


def test_an_empty_run_says_so_rather_than_rendering_nothing():
    """Sources are written overnight by another repo, so nothing pending means
    that chain did not run — which is exactly the morning a silent note would
    hide."""
    note = _summary(pending=0).render_markdown()
    assert "Nothing was pending" in note


def test_an_abandoned_run_carries_its_reason():
    s = _summary(outcome="abandoned", detail="run budget of 45 min spent")
    assert "## 05:00 — abandoned in 8.6 min" in s.render_markdown()
    assert "> run budget of 45 min spent" in s.render_markdown()


# --- the three ways a source stays unfinished ------------------------------


@pytest.mark.parametrize("status", [PARTIAL, NO_PLAN, LOG_FAILED])
def test_an_unfinished_source_says_it_will_be_retried(status):
    """An unmarked source is redone on the next scheduled run. A note that
    listed its pages and stopped there would read like a clean success."""
    s = _summary()
    s.add_source(_source(status=status, created=["alpha"]))

    assert "left unmarked, so the next run redoes it" in s.render_markdown()


def test_a_failed_page_is_named():
    s = _summary()
    s.add_source(_source(status=PARTIAL, created=["alpha"], failed=["beta"]))

    note = s.render_markdown()
    assert "- failed: beta" in note
    assert "1 page(s) failed" in note


def test_a_source_with_no_usable_plan_says_nothing_was_written():
    s = _summary()
    s.add_source(_source(status=NO_PLAN))

    assert "no usable plan — nothing was written" in s.render_markdown()


def test_pages_that_landed_without_a_log_entry_are_called_out():
    """Stage 3 is the non-idempotent one. Its failure leaves real pages on disk
    and no log.md entry, which is a different thing from a page failing."""
    s = _summary()
    s.add_source(_source(status=LOG_FAILED, created=["alpha"]))

    assert "pages landed, but the log entry did not" in s.render_markdown()


def test_only_finished_sources_count_as_ingested():
    s = _summary(pending=3)
    s.add_source(_source(filename="a.md", status=OK, created=["alpha"]))
    s.add_source(_source(filename="b.md", status=PARTIAL, failed=["beta"]))
    s.add_source(_source(filename="c.md", status=NO_PLAN))

    assert s.ingested == 1
    assert "1 of 3 source(s) ingested" in s.render_markdown()


# --- the log block ---------------------------------------------------------


def test_the_log_block_cannot_be_mistaken_for_a_run_or_a_source_line():
    """tools/measure_stage2.py anchors runs and sources on log phrases. Its
    _when() already skips any line without a leading timestamp, so only the
    first line of a multi-line record reaches those regexes — but a summary
    that reused one of these phrases would silently double every count the
    script reports, so pin it."""
    s = _summary(pending=1)
    s.add_source(_source(created=["alpha"], updated=["beta"]))
    block = s.render_log_block()

    assert not m._RUN.search(block)
    assert not m._SOURCE.search(block)
    assert not m._EDIT.search(block)
    assert not m._INCOMPLETE.search(block)

    # And the first line — the only one that carries a timestamp once the
    # formatter has prefixed it — is not a run header.
    assert not m._RUN.search(block.splitlines()[0])


def test_the_log_block_names_every_source():
    s = _summary(pending=2)
    s.add_source(_source(filename="a.md", created=["alpha"]))
    s.add_source(_source(filename="b.md", status=PARTIAL, failed=["beta"]))

    block = s.render_log_block()
    assert "a.md: created alpha" in block
    assert "b.md: failed beta; retried next run" in block


def test_a_source_that_wrote_nothing_says_so_in_the_log_block():
    s = _summary()
    s.add_source(_source(status=NO_PLAN))

    assert "nothing written" in s.render_log_block()


# --- where the note lands --------------------------------------------------


def test_the_note_lands_beside_the_wiki_not_inside_it(vault):
    """Every tool the model is given resolves under raw/ or wiki/, and
    wiki_lint walks wiki/ only. runs/ is outside both, so nothing written here
    can reach a prompt, the index, or a lint finding."""
    path = write_run_note(vault.path, _summary())

    assert path == Path(vault.path) / "runs" / "2026-09-12.md"
    assert not (Path(vault.path) / "wiki" / "runs").exists()
    assert list((Path(vault.path) / "wiki").glob("*.md")) == []


def test_a_new_note_gets_the_day_as_its_title(vault):
    path = write_run_note(vault.path, _summary())
    assert path.read_text(encoding="utf-8").startswith(
        "# Ingest runs — 2026-09-12\n"
    )


def test_a_second_run_the_same_day_appends(vault):
    """A repair run on the same day is normal, and the morning's run is the one
    you would least want overwritten."""
    first = _summary(started_at=datetime(2026, 9, 12, 5, 0))
    first.add_source(_source(filename="a.md", created=["alpha"]))
    write_run_note(vault.path, first)

    second = _summary(started_at=datetime(2026, 9, 12, 14, 30))
    second.add_source(_source(filename="b.md", created=["beta"]))
    path = write_run_note(vault.path, second)

    text = path.read_text(encoding="utf-8")
    assert text.count("# Ingest runs — 2026-09-12") == 1
    assert "## 05:00" in text and "## 14:30" in text
    assert text.index("## 05:00") < text.index("## 14:30")
    assert "alpha" in text and "beta" in text


def test_a_different_day_gets_its_own_note(vault):
    write_run_note(vault.path, _summary(started_at=datetime(2026, 9, 12, 5, 0)))
    write_run_note(vault.path, _summary(started_at=datetime(2026, 9, 13, 5, 0)))

    names = sorted(p.name for p in (Path(vault.path) / "runs").glob("*.md"))
    assert names == ["2026-09-12.md", "2026-09-13.md"]


# --- a run that stopped early ----------------------------------------------


def test_pages_that_landed_before_the_budget_blew_are_kept():
    """The reason ABANDONED exists. The budget watchdog raises from a signal
    handler at whatever point the process is in, so without a recorded result
    the pages already on disk for the in-flight source appear nowhere at all."""
    from agent.run_summary import ABANDONED

    s = _summary(pending=3, outcome="abandoned", detail="run budget spent")
    s.add_source(_source(filename="a.md", created=["alpha"]))
    s.add_source(_source(filename="b.md", status=ABANDONED, created=["beta"]))

    note = s.render_markdown()
    assert "- created: beta" in note
    assert "the run stopped here, with this source part-done" in note
    assert "left unmarked, so the next run redoes it" in note
    # And the headline counts the page even though its source is unfinished.
    assert "2 page(s) created" in note
    assert "1 of 3 source(s) ingested" in note


def test_sources_the_run_never_started_are_counted():
    from agent.run_summary import ABANDONED

    s = _summary(pending=5, outcome="abandoned", detail="run budget spent")
    s.add_source(_source(filename="a.md", created=["alpha"]))
    s.add_source(_source(filename="b.md", status=ABANDONED))

    assert s.unreached == 3
    assert "3 pending source(s) were never started" in s.render_markdown()


def test_a_completed_run_does_not_mention_unreached_sources():
    s = _summary(pending=1, outcome="complete")
    s.add_source(_source(created=["alpha"]))

    assert s.unreached == 0
    assert "never started" not in s.render_markdown()
