"""Tests for the on-demand query entrypoint in wiki_query.py.

run_agent is stubbed — this is about what main() does with an answer, not
about the loop, which tests/test_loop.py covers.
"""

import wiki_query as wq
from agent.loop import INCOMPLETE_PREFIX


def _main(vault, monkeypatch, answer, question="what did we learn?"):
    monkeypatch.setattr(wq, "run_agent", lambda **kw: answer)
    monkeypatch.setattr(
        "sys.argv", ["wiki_query.py", "--vault", vault.path, question]
    )
    return wq.main()


def test_an_answer_prints_and_exits_zero(vault, monkeypatch, capsys):
    assert _main(vault, monkeypatch, "Pages a and b say so.") == 0
    assert "Pages a and b say so." in capsys.readouterr().out


def test_an_incomplete_run_exits_non_zero(vault, monkeypatch, capsys):
    """The loop returns its marker as the answer, so a truncated run used to
    exit 0 and read as a real reply to anything scripting this."""
    marker = f"{INCOMPLETE_PREFIX} hit max_iterations=15 tool calls ...]"

    assert _main(vault, monkeypatch, marker) == 1

    out = capsys.readouterr()
    assert marker in out.out  # the partial result is still shown
    assert "without answering" in out.err


def test_query_raises_the_iteration_cap_above_the_loop_default(vault, monkeypatch):
    """The default 6 is index-plus-four-pages, and this workflow reads the
    index first."""
    seen = {}
    monkeypatch.setattr(wq, "run_agent", lambda **kw: seen.update(kw) or "ok")
    monkeypatch.setattr("sys.argv", ["wiki_query.py", "--vault", vault.path, "q?"])

    wq.main()

    assert seen["max_iterations"] == wq.MAX_QUERY_ITERATIONS > 6


def test_missing_rules_is_a_clean_error(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["wiki_query.py", "--vault", str(tmp_path), "q?"])
    assert wq.main() == 1
    assert "RULES.md not found" in capsys.readouterr().err
