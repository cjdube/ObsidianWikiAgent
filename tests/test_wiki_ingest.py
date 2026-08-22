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


_ONE_PAGE = [{"name": "alpha", "action": "create", "intent": "cover x", "section": "S"}]


def _page_in_prompt(prompt: str) -> str:
    """The page name _execute_unit put in its user prompt."""
    for line in prompt.splitlines():
        if line.startswith("Page to write: "):
            return line.split("'")[1]
    raise AssertionError(f"no page name in execute prompt:\n{prompt}")


def _stages(pages=None, fail=(), dead_stage=None):
    """A run_agent stand-in that plays whichever of the three stages it was
    called for, told apart by the tools it was handed."""
    planned = _ONE_PAGE if pages is None else pages

    def run(**kwargs):
        dispatch = kwargs["dispatch"]
        if "submit_plan" in dispatch:
            if dead_stage == "plan":
                return "answered without planning"
            dispatch["submit_plan"](pages=[dict(p) for p in planned], skipped="")
            return "planned"
        if "write_wiki_page" in dispatch:
            name = _page_in_prompt(kwargs["user_prompt"])
            if dead_stage == "execute" or name in fail:
                return "answered without writing"
            dispatch["write_wiki_page"](name=name, content=f"# {name}\n")
            return "wrote"
        if dead_stage == "log":
            return "answered without logging"
        dispatch["append_log"](entry="did the thing")
        return "logged"

    return run


def _writes(**kwargs):
    """The all-three-stages happy path, for tests that only care about which
    sources were attempted."""
    return _stages()(**kwargs)


# --- stage wiring ----------------------------------------------------------


def test_every_stage_dispatches_every_tool_it_advertises(vault):
    """A schema with no dispatch entry is an unknown-tool error mid-run, which
    costs a loop iteration and confuses the model. read_index is the deliberate
    other way round — dispatchable but unadvertised, so a RULES.md that names
    the index in prose still works."""
    from agent import wiki_tools as wt

    for schemas, dispatch in (
        (wt.PLAN_TOOL_SCHEMAS, wiki_ingest._plan_dispatch(vault.path, wiki_ingest._Plan())),
        (wt.EXECUTE_TOOL_SCHEMAS, wiki_ingest._execute_dispatch(vault.path, wiki_ingest._WriteCounter())),
        (wt.LOG_TOOL_SCHEMAS, wiki_ingest._log_dispatch(vault.path, wiki_ingest._WriteCounter())),
    ):
        advertised = {t["function"]["name"] for t in schemas}
        assert advertised <= set(dispatch)
        assert set(dispatch) - advertised <= {"read_index"}


def test_only_the_stages_that_transcribe_turn_thinking_off(vault, monkeypatch):
    """Stages 2 and 3 are handed what to do and only have to carry it out, so
    they run with reasoning off — 3.4x faster, and it stops the reasoning block
    eating the reply budget and cutting a page write short. Stage 1 decides
    which pages a source touches, and with reasoning off it returned without
    calling submit_plan in half of a six-trial benchmark, so it keeps it."""
    vault.raw("src.md")
    seen = {}

    def record(**kwargs):
        dispatch = kwargs["dispatch"]
        stage = (
            "plan" if "submit_plan" in dispatch
            else "execute" if "write_wiki_page" in dispatch
            else "log"
        )
        seen[stage] = kwargs.get("think")
        return _writes(**kwargs)

    monkeypatch.setattr(wiki_ingest, "run_agent", record)
    wiki_ingest.ingest_vault(vault.path, _Logger())

    assert seen == {"plan": None, "execute": False, "log": False}


def test_planning_stage_has_no_tool_that_writes(vault):
    """What makes --plan-only safe by construction rather than by care."""
    dispatch = wiki_ingest._plan_dispatch(vault.path, wiki_ingest._Plan())
    assert not {"write_wiki_page", "update_index", "append_log"} & set(dispatch)


def test_write_counter_ignores_a_refused_write(vault):
    """A refused reserved name comes back as an error result. Counting it as
    progress would mark a page done on a call that wrote nothing."""
    writes = wiki_ingest._WriteCounter()
    dispatch = wiki_ingest._execute_dispatch(vault.path, writes)

    assert "error" in dispatch["write_wiki_page"](name="index", content="x")
    assert not writes

    assert "written" in dispatch["write_wiki_page"](name="real", content="# Real\n")
    assert writes


# --- the plan the model submits is not trusted -----------------------------


def test_plan_drops_entries_that_would_write_somewhere_reserved(vault):
    """update_index owns index.md and append_log owns log.md. A plan entry
    naming either would send a whole page body at one of them."""
    plan = wiki_ingest._Plan()
    wiki_ingest._plan_dispatch(vault.path, plan)["submit_plan"](pages=[
        {"name": "index", "action": "update", "intent": "x", "section": "S"},
        {"name": "log.md", "action": "update", "intent": "x", "section": "S"},
        {"name": "real", "action": "create", "intent": "x", "section": "S"},
    ])
    assert plan.names() == ["real"]


def test_plan_drops_a_duplicate_page(vault):
    """Two entries for one page are two stage-2 conversations racing to replace
    the same file, and the second would overwrite the first blind."""
    plan = wiki_ingest._Plan()
    wiki_ingest._plan_dispatch(vault.path, plan)["submit_plan"](pages=[
        {"name": "ollama", "action": "create", "intent": "x", "section": "S"},
        {"name": "ollama.md", "action": "update", "intent": "y", "section": "S"},
    ])
    assert plan.names() == ["ollama"]


def test_plan_defaults_an_unclear_action_to_update(vault):
    """Guessing 'update' costs a read of a page that may not exist, which comes
    back as a plain not-found. Guessing 'create' skips the read and overwrites a
    real page with a version written without it."""
    plan = wiki_ingest._Plan()
    wiki_ingest._plan_dispatch(vault.path, plan)["submit_plan"](pages=[
        {"name": "a", "action": "", "intent": "x", "section": "S"},
        {"name": "b", "action": "CREATE", "intent": "x", "section": "S"},
    ])
    assert [p["action"] for p in plan.pages] == ["update", "create"]


def test_empty_plan_is_reported_back_as_an_error(vault):
    """The model gets a chance to fix it inside the same conversation."""
    plan = wiki_ingest._Plan()
    result = wiki_ingest._plan_dispatch(vault.path, plan)["submit_plan"](pages=[])
    assert "error" in result
    assert not plan


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

    sources_started = []

    def one_then_wedge(**kwargs):
        # A source now spans three conversations, so wedge on the second
        # source's planning stage rather than the second call.
        if "submit_plan" in kwargs["dispatch"]:
            sources_started.append(1)
            if len(sources_started) > 1:
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
        # Sampled once per source, at its planning stage. The ceiling spans all
        # three of a source's stages deliberately: a spent ceiling means the
        # server is unwell, which is a property of the box and not of one page.
        if "submit_plan" in kwargs["dispatch"]:
            ceilings.append(budget._retries_left)
        return _writes(**kwargs)

    monkeypatch.setattr(wiki_ingest, "run_agent", record)
    wiki_ingest.ingest_vault(vault.path, _Logger())

    assert ceilings == [wiki_ingest.MAX_RETRIES_PER_SOURCE] * 2


# --- the pre-existing no-op retry still works ------------------------------


def test_source_that_never_plans_is_retried_then_left_unmarked(vault, monkeypatch):
    calls = []

    def counted(**kwargs):
        calls.append(1)
        return _stages(dead_stage="plan")(**kwargs)

    monkeypatch.setattr(wiki_ingest, "run_agent", counted)
    vault.raw("one.md", subdir="daily-notes")

    rc = wiki_ingest.ingest_vault(vault.path, _Logger())

    assert rc == 1
    # Stage 1 retried, and no stage-2 conversation was ever started.
    assert len(calls) == wiki_ingest.MAX_INGEST_ATTEMPTS
    assert get_ingested_sources(vault.path) == []


def test_successful_source_is_marked(vault, monkeypatch):
    monkeypatch.setattr(wiki_ingest, "run_agent", lambda **kw: _writes(**kw))
    vault.raw("one.md", subdir="daily-notes")

    assert wiki_ingest.ingest_vault(vault.path, _Logger()) == 0
    assert len(get_ingested_sources(vault.path)) == 1


# --- the split's own guarantees --------------------------------------------


def test_each_planned_page_gets_its_own_conversation(vault, monkeypatch):
    """The whole point: context is per page, so it stops growing with how many
    pages a source touches."""
    prompts = []

    def record(**kwargs):
        if "write_wiki_page" in kwargs["dispatch"]:
            prompts.append(kwargs["user_prompt"])
        return _stages(pages=[
            {"name": "alpha", "action": "create", "intent": "x", "section": "S"},
            {"name": "beta", "action": "update", "intent": "y", "section": "S"},
        ])(**kwargs)

    monkeypatch.setattr(wiki_ingest, "run_agent", record)
    vault.raw("one.md", subdir="daily-notes")
    wiki_ingest.ingest_vault(vault.path, _Logger())

    assert [_page_in_prompt(p) for p in prompts] == ["alpha", "beta"]
    assert (Path(vault.path) / "wiki" / "alpha.md").is_file()
    assert (Path(vault.path) / "wiki" / "beta.md").is_file()


def test_each_page_is_told_about_its_siblings(vault, monkeypatch):
    """Bidirectional links and the no-duplicate-page rule both need cross-page
    awareness, which separate conversations otherwise destroy."""
    prompts = {}

    def record(**kwargs):
        if "write_wiki_page" in kwargs["dispatch"]:
            prompts[_page_in_prompt(kwargs["user_prompt"])] = kwargs["user_prompt"]
        return _stages(pages=[
            {"name": "alpha", "action": "create", "intent": "x", "section": "S"},
            {"name": "beta", "action": "update", "intent": "y", "section": "S"},
        ])(**kwargs)

    monkeypatch.setattr(wiki_ingest, "run_agent", record)
    vault.raw("one.md", subdir="daily-notes")
    wiki_ingest.ingest_vault(vault.path, _Logger())

    assert "[[beta]]" in prompts["alpha"]
    assert "[[alpha]]" in prompts["beta"]
    # And never itself.
    assert "[[alpha]]" not in prompts["alpha"]


def test_a_partly_written_source_is_left_unmarked(vault, monkeypatch):
    """Stricter than the old ingest, which marked a source done as soon as any
    write landed — the half-ingested sources wiki_lint.check_source_coverage
    exists to find. Redoing tomorrow is safe: page writes overwrite by name."""
    monkeypatch.setattr(wiki_ingest, "run_agent", _stages(pages=[
        {"name": "alpha", "action": "create", "intent": "x", "section": "S"},
        {"name": "beta", "action": "create", "intent": "y", "section": "S"},
    ], fail=("beta",)))
    vault.raw("one.md", subdir="daily-notes")

    assert wiki_ingest.ingest_vault(vault.path, _Logger()) == 1
    assert get_ingested_sources(vault.path) == []
    # The page that did land stays on disk; tomorrow rewrites it.
    assert (Path(vault.path) / "wiki" / "alpha.md").is_file()


def test_no_log_entry_when_a_page_failed(vault, monkeypatch):
    """append_log is the one non-idempotent tool, so it must not run on a pass
    that will be repeated tomorrow."""
    logged = []

    def watch(**kwargs):
        if "append_log" in kwargs["dispatch"]:
            logged.append(1)
        return _stages(pages=[
            {"name": "alpha", "action": "create", "intent": "x", "section": "S"},
        ], fail=("alpha",))(**kwargs)

    monkeypatch.setattr(wiki_ingest, "run_agent", watch)
    vault.raw("one.md", subdir="daily-notes")
    wiki_ingest.ingest_vault(vault.path, _Logger())

    assert logged == []


def test_one_log_entry_per_source_not_per_page(vault, monkeypatch):
    monkeypatch.setattr(wiki_ingest, "run_agent", _stages(pages=[
        {"name": "alpha", "action": "create", "intent": "x", "section": "S"},
        {"name": "beta", "action": "create", "intent": "y", "section": "S"},
        {"name": "gamma", "action": "create", "intent": "z", "section": "S"},
    ]))
    vault.raw("one.md", subdir="daily-notes")
    wiki_ingest.ingest_vault(vault.path, _Logger())

    entries = (Path(vault.path) / "wiki" / "log.md").read_text().strip().splitlines()
    assert len(entries) == 1


def test_a_failed_page_retries_only_that_page(vault, monkeypatch):
    """A per-source retry would re-plan and rewrite the pages that already
    landed, which is the cost the split exists to avoid."""
    seen = []

    def record(**kwargs):
        if "write_wiki_page" in kwargs["dispatch"]:
            seen.append(_page_in_prompt(kwargs["user_prompt"]))
        return _stages(pages=[
            {"name": "alpha", "action": "create", "intent": "x", "section": "S"},
            {"name": "beta", "action": "create", "intent": "y", "section": "S"},
        ], fail=("beta",))(**kwargs)

    monkeypatch.setattr(wiki_ingest, "run_agent", record)
    vault.raw("one.md", subdir="daily-notes")
    wiki_ingest.ingest_vault(vault.path, _Logger())

    assert seen.count("alpha") == 1
    assert seen.count("beta") == wiki_ingest.MAX_INGEST_ATTEMPTS


# --- --plan-only -----------------------------------------------------------


def test_plan_only_writes_nothing_and_marks_nothing(vault, monkeypatch, capsys):
    monkeypatch.setattr(wiki_ingest, "run_agent", _stages(pages=[
        {"name": "alpha", "action": "create", "intent": "cover x", "section": "S"},
    ]))
    vault.raw("one.md", subdir="daily-notes")

    assert wiki_ingest.ingest_vault(vault.path, _Logger(), plan_only=True) == 0

    assert get_ingested_sources(vault.path) == []
    assert not (Path(vault.path) / "wiki" / "alpha.md").exists()
    assert not (Path(vault.path) / "wiki" / "log.md").exists()

    out = capsys.readouterr().out
    assert "alpha" in out and "cover x" in out
    assert "Nothing was written" in out


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
