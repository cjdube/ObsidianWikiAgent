"""Guards the definitions in tools/measure_stage2.py.

An analysis script is only useful if the number it prints means the same thing
every time it is run. These tests pin the three definitions that could quietly
drift and make two measurements incomparable: what counts as a repeat, which
max_iterations line is stage 2, and whether test-vault runs can be averaged in
with scheduled ones.
"""

import sys
from datetime import datetime
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
import measure_stage2 as m  # noqa: E402


def _line(ts, body):
    return f"2026-09-0{ts} [INFO] {body}\n"


def _log(tmp_path, name, body):
    p = tmp_path / f"wiki_ingest.{name}.log"
    p.write_text(body, encoding="utf-8")
    return str(p)


RUN = "1 09:00:00,000 [INFO] Starting wiki ingest run for vault: /v\n"


def _edit(clock, args, result):
    return (f"2026-09-01 09:00:{clock:02d},000 [INFO] tool_call "
            f"edit_wiki_page({args}) -> {result}\n")


def test_only_an_immediately_repeated_identical_call_is_a_repeat(tmp_path):
    """The defect is the model re-sending the same call because it read
    'unchanged' as a failure. The same content sent again later, after other
    work, is the model revisiting a page — not this bug, and counting it would
    inflate every future measurement."""
    same = "{'name': 'a', 'content': 'x'}"
    other = "{'name': 'b', 'content': 'y'}"
    unchanged = '{"unchanged": "a.md"}'

    body = (
        "2026-09-01 09:00:00,000 [INFO] Starting wiki ingest run for vault: /v\n"
        + _edit(1, same, '{"edited": "a.md"}')
        + _edit(2, same, unchanged)      # immediate repeat -> counts
        + _edit(3, same, unchanged)      # immediate repeat -> counts
        + _edit(4, other, '{"edited": "b.md"}')
        + _edit(5, same, unchanged)      # same content, but not immediate
    )
    (run,) = m.scan([_log(tmp_path, "v", body)])

    assert run["edits"] == 5
    assert run["unchanged"] == 3
    assert run["repeats"] == 2


def test_only_the_stage_two_cap_counts_as_an_abandoned_page(tmp_path):
    """max_iterations is per stage: 14 plan, 12 execute, 4 log. Only the
    execute cap means a page was saved mid-thought, so only 12 is counted."""
    body = (
        "2026-09-01 09:00:00,000 [INFO] Starting wiki ingest run for vault: /v\n"
        "2026-09-01 09:00:01,000 [INFO] Wrote 'p1' for 's.md': "
        "[incomplete: hit max_iterations=12 tool calls without reaching a final answer]\n"
        "2026-09-01 09:00:02,000 [INFO] Wrote 'p2' for 's.md': "
        "[incomplete: hit max_iterations=14 tool calls without reaching a final answer]\n"
        "2026-09-01 09:00:03,000 [INFO] Wrote 'p3' for 's.md': "
        "[incomplete: hit max_iterations=4 tool calls without reaching a final answer]\n"
    )
    (run,) = m.scan([_log(tmp_path, "v", body)])

    assert run["incomplete"] == 1


def test_lines_before_the_first_run_header_are_dropped(tmp_path):
    """A rotated log starts mid-run. Those calls belong to a run whose totals
    are gone, and attributing them to the next run would overstate it."""
    body = (
        _edit(1, "{'name': 'orphan'}", '{"edited": "orphan.md"}')
        + "2026-09-01 09:00:05,000 [INFO] Starting wiki ingest run for vault: /v\n"
        + _edit(6, "{'name': 'real'}", '{"edited": "real.md"}')
    )
    (run,) = m.scan([_log(tmp_path, "v", body)])

    assert run["edits"] == 1


def test_sources_are_counted_per_run(tmp_path):
    body = (
        "2026-09-01 09:00:00,000 [INFO] Starting wiki ingest run for vault: /v\n"
        "2026-09-01 09:00:01,000 [INFO] Ingesting 'a.md'\n"
        "2026-09-01 09:00:02,000 [INFO] Ingesting 'b.md'\n"
        "2026-09-02 09:00:00,000 [INFO] Starting wiki ingest run for vault: /v\n"
        "2026-09-02 09:00:01,000 [INFO] Ingesting 'c.md'\n"
    )
    first, second = m.scan([_log(tmp_path, "v", body)])

    assert (first["sources"], second["sources"]) == (2, 1)


def test_two_vaults_are_never_averaged_together(tmp_path, monkeypatch):
    """Disposable `git archive` test vaults log beside the scheduled vault.
    Their runs have different sizes and hand-picked sources, so silently
    averaging them would change what the number means between two runs of
    this tool."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "logs").mkdir()
    for name in ("llm-wiki-learnings", "vault-e2e"):
        (tmp_path / "logs" / f"wiki_ingest.{name}.log").write_text("", encoding="utf-8")

    with pytest.raises(SystemExit) as e:
        m.resolve([], None)
    assert "more than one vault" in str(e.value)
    assert "--vault llm-wiki-learnings" in str(e.value)

    # Naming one is unambiguous, and picks up only that vault's files.
    assert m.resolve([], "vault-e2e") == ["logs/wiki_ingest.vault-e2e.log"]


def test_a_single_vault_needs_no_flag(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "logs").mkdir()
    for suffix in ("log", "log.1"):
        (tmp_path / "logs" / f"wiki_ingest.only.{suffix}").write_text("", encoding="utf-8")

    assert m.resolve([], None) == [
        "logs/wiki_ingest.only.log", "logs/wiki_ingest.only.log.1",
    ]


def test_report_says_when_the_sample_is_too_thin(tmp_path):
    """Two runs is not a result. The verdict has to say so, because the table
    above it looks equally confident either way."""
    runs = [
        {"start": datetime(2026, 9, 1), "sources": 4, "edits": 9,
         "unchanged": 4, "repeats": 4, "incomplete": 2},
        {"start": datetime(2026, 9, 9), "sources": 4, "edits": 9,
         "unchanged": 0, "repeats": 0, "incomplete": 0},
    ]
    out = []
    m.report(runs, m.FIX_TIME, out=out.append)
    text = "\n".join(out)

    assert "2 before (0.50 per source), 0 after" in text
    assert "treat this as a signal, not a result" in text
