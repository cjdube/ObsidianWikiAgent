"""Tests for the run-level control flow in wiki_ingest.py.

run_agent is mocked throughout — these cover which sources get attempted,
marked, and abandoned, not what the model produces.
"""

from pathlib import Path

import pytest

import wiki_ingest
from agent import budget
from agent.wiki_tools import get_ingested_sources


@pytest.fixture(autouse=True)
def _no_sorting(monkeypatch):
    """The sort step is exercised in its own tests; stub it out so these read
    as ingest-only."""
    monkeypatch.setattr(wiki_ingest, "sort_raw_files", lambda *a, **k: None)


@pytest.fixture(autouse=True)
def _no_real_log(monkeypatch):
    """main() would otherwise write a log file per test into the repo's logs/."""
    monkeypatch.setattr(wiki_ingest, "setup_logger", lambda name: _Logger())


class _Logger:
    """Records levels so a test can assert something was logged as an error."""

    def __init__(self):
        self.records = []

    def _log(self, level):
        return lambda msg, *a, **k: self.records.append((level, str(msg)))

    def __getattr__(self, name):
        return self._log(name)


def _writes(**kwargs):
    """A run_agent stand-in that calls one write tool, i.e. a successful ingest."""
    kwargs["dispatch"]["append_log"](entry="did the thing")
    return "done"


# --- budget stops the run --------------------------------------------------


def test_ingest_dispatch_covers_every_advertised_tool(vault):
    """A schema with no dispatch entry is an unknown-tool error mid-run, which
    costs a loop iteration and confuses the model. read_index is the deliberate
    other way round — dispatchable but unadvertised, so a RULES.md that names
    the index in prose still works."""
    from agent.wiki_tools import INGEST_TOOL_SCHEMAS

    dispatch = wiki_ingest._build_dispatch(vault.path, wiki_ingest._WriteCounter())
    advertised = {t["function"]["name"] for t in INGEST_TOOL_SCHEMAS}

    assert advertised <= set(dispatch)
    assert set(dispatch) - advertised == {"read_index"}


def test_write_counter_ignores_a_refused_write(vault):
    """A refused reserved name comes back as an error result. Counting it as
    progress would mark a source ingested on a call that wrote nothing."""
    writes = wiki_ingest._WriteCounter()
    dispatch = wiki_ingest._build_dispatch(vault.path, writes)

    assert "error" in dispatch["write_wiki_page"](name="index", content="x")
    assert not writes

    assert "written" in dispatch["write_wiki_page"](name="real", content="# Real\n")
    assert writes


def test_budget_exhaustion_propagates_out_of_ingest_vault(vault):
    vault.raw("one.md", subdir="daily-notes")
    budget.start_run(-1)

    with pytest.raises(budget.BudgetExceeded):
        wiki_ingest.ingest_vault(vault.path, _Logger())


def test_budget_exhaustion_is_not_swallowed_by_the_attempt_retry(vault, monkeypatch):
    """The generic `except Exception` around run_agent retries the source. A
    BudgetExceeded caught there would spend the next two attempts too, which
    is exactly what the budget exists to prevent."""
    calls = []

    def wedged(**kwargs):
        calls.append(1)
        raise budget.BudgetExceeded("run budget exhausted")

    vault.raw("one.md", subdir="daily-notes")
    monkeypatch.setattr(wiki_ingest, "run_agent", wedged)

    with pytest.raises(budget.BudgetExceeded):
        wiki_ingest.ingest_vault(vault.path, _Logger())
    assert calls == [1]  # not MAX_INGEST_ATTEMPTS


def test_budget_exhaustion_keeps_sources_already_done(vault, monkeypatch):
    """A run cut short must not lose the work it finished, and must leave the
    rest unmarked so tomorrow's run picks them up."""
    vault.raw("one.md", subdir="daily-notes")
    vault.raw("two.md", subdir="daily-notes")

    seen = []

    def one_then_wedge(**kwargs):
        seen.append(1)
        if len(seen) > 1:
            raise budget.BudgetExceeded("run budget exhausted")
        return _writes(**kwargs)

    monkeypatch.setattr(wiki_ingest, "run_agent", one_then_wedge)

    logger = _Logger()
    with pytest.raises(budget.BudgetExceeded):
        wiki_ingest.ingest_vault(vault.path, logger)

    ingested = get_ingested_sources(vault.path)
    assert len(ingested) == 1
    assert any(level == "error" and "Run stopped after 1/2 source(s)" in msg
               for level, msg in logger.records)


# --- retry ceiling is per source -------------------------------------------


def test_each_source_gets_a_fresh_retry_ceiling(vault, monkeypatch):
    vault.raw("one.md", subdir="daily-notes")
    vault.raw("two.md", subdir="daily-notes")

    ceilings = []

    def record(**kwargs):
        ceilings.append(budget._retries_left)
        return _writes(**kwargs)

    monkeypatch.setattr(wiki_ingest, "run_agent", record)
    wiki_ingest.ingest_vault(vault.path, _Logger())

    assert ceilings == [wiki_ingest.MAX_RETRIES_PER_SOURCE] * 2


# --- the pre-existing no-op retry still works ------------------------------


def test_source_with_no_writes_is_retried_then_left_unmarked(vault, monkeypatch):
    calls = []
    monkeypatch.setattr(
        wiki_ingest, "run_agent",
        lambda **kwargs: calls.append(1) or "answered without writing",
    )
    vault.raw("one.md", subdir="daily-notes")

    rc = wiki_ingest.ingest_vault(vault.path, _Logger())

    assert rc == 1
    assert len(calls) == wiki_ingest.MAX_INGEST_ATTEMPTS
    assert get_ingested_sources(vault.path) == []


def test_successful_source_is_marked(vault, monkeypatch):
    monkeypatch.setattr(wiki_ingest, "run_agent", lambda **kw: _writes(**kw))
    vault.raw("one.md", subdir="daily-notes")

    assert wiki_ingest.ingest_vault(vault.path, _Logger()) == 0
    assert len(get_ingested_sources(vault.path)) == 1


# --- failure notification --------------------------------------------------


def test_main_pushes_on_budget_exhaustion(vault, monkeypatch):
    pushed = []
    monkeypatch.setattr(
        wiki_ingest, "notify_failure",
        lambda job, detail, logger=None: pushed.append((job, str(detail))),
    )
    monkeypatch.setattr(
        wiki_ingest, "ingest_vault",
        lambda *a, **k: (_ for _ in ()).throw(budget.BudgetExceeded("wedged")),
    )
    monkeypatch.setattr(
        "sys.argv", ["wiki_ingest.py", "--vault", vault.path, "--budget-minutes", "45"]
    )

    assert wiki_ingest.main() == 1
    assert len(pushed) == 1
    assert pushed[0][0] == f"wiki_ingest[{Path(vault.path).name}]"
    assert "abandoned" in pushed[0][1] and "wedged" in pushed[0][1]


def test_main_pushes_on_unexpected_failure(vault, monkeypatch):
    pushed = []
    monkeypatch.setattr(
        wiki_ingest, "notify_failure",
        lambda job, detail, logger=None: pushed.append((job, str(detail))),
    )
    monkeypatch.setattr(
        wiki_ingest, "ingest_vault",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    monkeypatch.setattr("sys.argv", ["wiki_ingest.py", "--vault", vault.path])

    assert wiki_ingest.main() == 1
    assert pushed[0][1] == "boom"


def test_main_is_quiet_on_a_clean_run(vault, monkeypatch):
    pushed = []
    monkeypatch.setattr(
        wiki_ingest, "notify_failure",
        lambda job, detail, logger=None: pushed.append(1),
    )
    monkeypatch.setattr(wiki_ingest, "ingest_vault", lambda *a, **k: 0)
    monkeypatch.setattr("sys.argv", ["wiki_ingest.py", "--vault", vault.path])

    assert wiki_ingest.main() == 0
    assert pushed == []
