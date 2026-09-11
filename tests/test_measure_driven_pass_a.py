"""Guards the definitions in tools/measure_driven_pass_a.py.

The sweep's whole claim is that every page was judged. That claim rests on two
pieces of pure Python — how a reply becomes a verdict, and which planted
defects the score is allowed to count — and a drift in either would make two
measurements incomparable while still printing a confident number.

The verdict parser is pinned hardest because it already failed once. A
verdict-first prompt made the model answer FINDING and then argue itself back
to CLEAN in the body; reading the first line scored nine in-scope pages as
scope violations on the 35-page fixture. Reading the LAST verdict line scored
the same fixture 6 flagged, 3 of 3 recall, zero false positives.
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

import measure_driven_pass_a as m  # noqa: E402


# --- verdict_of -------------------------------------------------------------

def test_a_bare_verdict_line_is_read():
    assert m.verdict_of("VERDICT: CLEAN") == m.CLEAN
    assert m.verdict_of("VERDICT: FINDING") == m.FINDING


def test_the_last_verdict_wins_not_the_first():
    """The bug this parser exists for.

    With think=False the model's reasoning has nowhere to live but the reply,
    so it reaches its answer on the way down. An early word is working out
    loud; the verdict line is the answer.
    """
    reply = (
        "VERDICT: FINDING\n"
        "Wait, let me re-read the Scope section.\n"
        "GPU memory is part of local model serving, so it is in scope.\n"
        "VERDICT: CLEAN"
    )
    assert m.verdict_of(reply) == m.CLEAN


def test_both_words_in_the_reasoning_do_not_decide_it():
    reply = (
        "This is not CLEAN at first glance, and a FINDING would be tempting.\n"
        "The Scope section excludes employee benefits.\n"
        "1. dental-plan-enrollment covers benefits; delete it.\n"
        "VERDICT: FINDING"
    )
    assert m.verdict_of(reply) == m.FINDING


def test_markdown_decoration_around_the_verdict_is_tolerated():
    assert m.verdict_of("**VERDICT: CLEAN**") == m.CLEAN
    assert m.verdict_of("## VERDICT: FINDING.") == m.FINDING


def test_a_reply_with_no_verdict_line_is_unparsed_not_clean():
    """Never fold an unreadable answer into 'clean'.

    A sweep that treats prose it could not read as a clean page prints exactly
    the false all-clear this design exists to prevent.
    """
    assert m.verdict_of("The page is clean on both questions.") == m.UNPARSED
    assert m.verdict_of("NONE") == m.UNPARSED
    assert m.verdict_of("") == m.UNPARSED


def test_an_unrecognised_verdict_word_is_unparsed():
    assert m.verdict_of("VERDICT: MAYBE") == m.UNPARSED


# --- answer_key -------------------------------------------------------------

def _seed(tmp_path, defects):
    p = tmp_path / "seed.json"
    p.write_text(json.dumps({"defects": defects}), encoding="utf-8")
    return p


def test_only_out_of_scope_defects_are_scorable(tmp_path):
    """A single-page pass cannot see a pairwise defect, so it is not credited
    or blamed for one. Every `outdated` defect in the real pack names a
    superseding page that the pass never holds."""
    path = _seed(tmp_path, [
        {"id": "O1", "kind": "out_of_scope", "pages": ["pizza-night"]},
        {"id": "U1", "kind": "outdated", "pages": ["retry-ceiling"],
         "superseded_by": "daily-notes-2026-08-14"},
        {"id": "C1", "kind": "contradiction", "pages": ["a", "b"]},
        {"id": "D1", "kind": "duplicate", "pages": ["c", "d"]},
    ])
    assert m.answer_key(path) == {"pizza-night": "O1"}


def test_the_real_pack_offers_exactly_three_scorable_defects():
    """Pins the claim the handoff brief got wrong: nine of the twelve planted
    defects need two pages, so Pass A can be scored on three."""
    key = m.answer_key(
        Path(__file__).resolve().parent.parent / "tools" / "lint_defects"
        / "seed.json"
    )
    assert sorted(key.values()) == ["O1", "O2", "O3"]


# --- report -----------------------------------------------------------------

def _row(name, verdict, seconds=1.0):
    return {"name": name, "reply": verdict, "seconds": seconds,
            "verdict": verdict}


def test_recall_counts_only_flagged_pages_named_in_the_key():
    rows = [
        _row("pizza-night.md", m.FINDING),
        _row("volunteer-day.md", m.CLEAN),
        _row("in-scope-page.md", m.FINDING),
    ]
    key = {"pizza-night": "O1", "volunteer-day": "O2"}
    out = m.report(rows, 3.0, key, out=lambda s: None)
    assert out["caught"] == 1
    assert out["flagged"] == 2


def test_unparsed_is_reported_separately_from_clean():
    rows = [_row("a.md", m.CLEAN), _row("b.md", m.UNPARSED)]
    out = m.report(rows, 2.0, None, out=lambda s: None)
    assert out["unparsed"] == 1
    assert out["flagged"] == 0


def test_report_names_the_misses(capsys):
    rows = [_row("pizza-night.md", m.CLEAN)]
    lines = []
    m.report(rows, 1.0, {"pizza-night": "O1"}, out=lines.append)
    assert any("missed ['O1']" in line for line in lines)


def test_flagged_pages_outside_the_key_are_not_called_wrong():
    """On the fixture the three daily-notes pages are flagged and all three
    are correct. The tool must report them as needing a read, not as errors."""
    rows = [_row("daily-notes-2026-08-14.md", m.FINDING)]
    lines = []
    m.report(rows, 1.0, {"pizza-night": "O1"}, out=lines.append)
    assert any("not automatically wrong" in line for line in lines)


# --- the scoring guard ------------------------------------------------------

def test_scoring_a_partial_sweep_is_refused(tmp_path, monkeypatch):
    """--pages with --score would report a miss on a page the sweep was never
    going to visit, which reads as a recall failure and is a bookkeeping one."""
    vault = tmp_path / "v"
    (vault / "wiki").mkdir(parents=True)
    (vault / "RULES.md").write_text("scope", encoding="utf-8")
    for name in ("aaa", "pizza-night"):
        (vault / "wiki" / f"{name}.md").write_text("# x", encoding="utf-8")

    seed = _seed(tmp_path, [
        {"id": "O1", "kind": "out_of_scope", "pages": ["pizza-night"]},
    ])

    with pytest.raises(SystemExit) as e:
        m.main(["--vault", str(vault), "--score", str(seed), "--pages", "1"])
    assert "pizza-night" in str(e.value)
