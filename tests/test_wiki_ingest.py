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
def _no_sorting(vault, monkeypatch):
    """The sort step is exercised in its own tests; stub it out so these read
    as ingest-only."""
    monkeypatch.setattr(wiki_ingest, "sort_raw_files", lambda *a, **k: None)
    vault.index("# Index\n\n## S\n\n")


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
        if "write_wiki_page" in dispatch or "edit_wiki_page" in dispatch:
            name = _page_in_prompt(kwargs["user_prompt"])
            if dead_stage == "execute" or name in fail:
                return "answered without writing"
            # Whichever write tool this page's stage was given — the set depends
            # on whether the page is already on disk.
            if "edit_wiki_page" in dispatch:
                dispatch["edit_wiki_page"](
                    name=name, section="Notes", content=f"- from the source\n"
                )
            else:
                dispatch["write_wiki_page"](
                    name=name, content=f"# {name}\n\n**Summary**: covered\n"
                )
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
    the index in prose still works, and update_index joined it once filing
    became Python's job."""
    from agent import wiki_tools as wt

    counter = wiki_ingest._WriteCounter()
    for schemas, dispatch in (
        (wt.PLAN_TOOL_SCHEMAS, wiki_ingest._plan_dispatch(vault.path, wiki_ingest._Plan())),
        (wt.CREATE_PAGE_TOOL_SCHEMAS,
         wiki_ingest._execute_dispatch(vault.path, "s.md", "p", counter, exists=False)),
        (wt.UPDATE_PAGE_TOOL_SCHEMAS,
         wiki_ingest._execute_dispatch(vault.path, "s.md", "p", counter, exists=True)),
        (wt.LOG_TOOL_SCHEMAS, wiki_ingest._log_dispatch(vault.path, wiki_ingest._WriteCounter())),
    ):
        advertised = {t["function"]["name"] for t in schemas}
        assert advertised <= set(dispatch)
        assert set(dispatch) - advertised <= {"read_index", "update_index"}


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
    """A refused write comes back as an error result. Counting it as progress
    would mark a page done on a call that wrote nothing. 'index' is refused
    here by the this-page-only guard before RESERVED ever sees it — both are
    error results, and the counter must ignore either."""
    writes = wiki_ingest._WriteCounter()
    dispatch = wiki_ingest._execute_dispatch(
        vault.path, "src.md", "real", writes, exists=False
    )

    assert "error" in dispatch["write_wiki_page"](name="index", content="x")
    assert not writes

    assert "written" in dispatch["write_wiki_page"](name="real", content="# Real\n")
    assert writes


def test_edit_counter_ignores_a_refused_edit(vault):
    """The update path needs the same guarantee as the create path — a refused
    call is an error result, not progress."""
    vault.page("real", "# Real\n\n**Sources**: a.md\n**Last updated**: 2026-01-01\n")
    writes = wiki_ingest._WriteCounter()
    dispatch = wiki_ingest._execute_dispatch(
        vault.path, "src.md", "real", writes, exists=True
    )

    assert "error" in dispatch["edit_wiki_page"](
        name="ghost", section="S", content="x"
    )
    assert not writes

    assert "edited" in dispatch["edit_wiki_page"](
        name="real", section="Notes", content="- new fact\n"
    )
    assert writes


def test_the_source_filename_is_not_the_models_to_supply(vault):
    """RULES.md has a paragraph of rules about citing a source correctly — bare
    filename, no directory prefix — and the sort step files sources into
    raw/<folder>/. Binding it here means none of that can be got wrong."""
    vault.page("real", "# Real\n\n**Sources**: a.md\n**Last updated**: 2026-01-01\n")
    dispatch = wiki_ingest._execute_dispatch(
        vault.path, "daily-ai/Chat-2026-08-20.md", "real",
        wiki_ingest._WriteCounter(), True
    )
    dispatch["edit_wiki_page"](name="real", section="Notes", content="- fact\n")

    text = (vault.root / "wiki" / "real.md").read_text()
    assert "**Sources**: a.md, Chat-2026-08-20.md" in text
    assert "daily-ai/" not in text


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


def test_duplicate_raw_basenames_stop_before_either_source_is_ingested(
    vault, monkeypatch
):
    vault.raw("same.md", subdir="daily-notes")
    vault.raw("same.md", subdir="misc")
    called = []
    monkeypatch.setattr(wiki_ingest, "run_agent", lambda **kw: called.append(1))

    with pytest.raises(RuntimeError, match="duplicate raw filename 'same.md'"):
        wiki_ingest.ingest_vault(vault.path, _Logger())

    assert called == []
    assert get_ingested_sources(vault.path) == []


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


def test_read_index_is_dispatchable_in_every_stage(vault):
    """The comment in _plan_dispatch has always said "in every stage"; stage 3
    was the exception for as long as it said so. A RULES.md naming wiki/index.md
    in prose is read by all three stages, so a call arriving in the log stage
    should work there too — and that stage has the fewest iterations to spare."""
    counter = wiki_ingest._WriteCounter()
    stages = (
        wiki_ingest._plan_dispatch(vault.path, wiki_ingest._Plan()),
        wiki_ingest._execute_dispatch(vault.path, "s.md", "p", counter, exists=False),
        wiki_ingest._execute_dispatch(vault.path, "s.md", "p", counter, exists=True),
        wiki_ingest._log_dispatch(vault.path, wiki_ingest._WriteCounter()),
    )
    for dispatch in stages:
        assert "read_index" in dispatch


def test_stage_three_read_index_does_not_count_as_a_write(vault):
    """A source is marked done on the strength of the write counter, so a read
    that incremented it would mark a source done having logged nothing."""
    vault.index("# Index\n\n## Tools\n\n")
    counter = wiki_ingest._WriteCounter()

    wiki_ingest._log_dispatch(vault.path, counter)["read_index"]()

    assert counter.count == 0


# --- one step writes one page ----------------------------------------------


def test_a_create_step_cannot_overwrite_a_page_it_was_not_given(vault):
    """The hole the create/update split alone left open. The split decides
    which tool a step is handed, not which file that tool is pointed at, and
    write_wiki_page replaces whole files and takes the name as an argument. So
    a step created for a page that did not exist could name one that did and
    destroy it — through the exact path the split exists to close."""
    vault.page("colima", "# Colima\n\nThe real page.\n")
    writes = wiki_ingest._WriteCounter()
    dispatch = wiki_ingest._execute_dispatch(
        vault.path, "src.md", "podman", writes, exists=False
    )

    result = dispatch["write_wiki_page"](name="colima", content="# Clobbered\n")

    assert "error" in result
    assert (vault.root / "wiki" / "colima.md").read_text() == "# Colima\n\nThe real page.\n"
    assert not writes


def test_an_update_step_cannot_edit_a_page_it_was_not_given(vault):
    """Less destructive than an overwrite and wrong for the same reason: the
    text lands on a page nobody asked to change, cited to this source."""
    vault.page("colima", "# Colima\n\n## Notes\n\n- real\n")
    writes = wiki_ingest._WriteCounter()
    dispatch = wiki_ingest._execute_dispatch(
        vault.path, "src.md", "podman", writes, exists=True
    )

    result = dispatch["edit_wiki_page"](
        name="colima", section="Notes", content="- smuggled in\n"
    )

    assert "error" in result
    assert "smuggled in" not in (vault.root / "wiki" / "colima.md").read_text()
    assert not writes


def test_the_refusal_names_the_page_the_step_should_write(vault):
    """A model mid-workflow that is only told 'no' retries the same call — the
    2026-08-20 truncation loop is what that costs. So the error names the page
    and the call to make, the same way the RESERVED refusal names the tool."""
    dispatch = wiki_ingest._execute_dispatch(
        vault.path, "src.md", "podman", wiki_ingest._WriteCounter(), exists=False
    )

    error = dispatch["write_wiki_page"](name="colima", content="x")["error"]

    assert "podman" in error
    assert "write_wiki_page" in error


def test_the_same_page_spelled_differently_is_still_this_step_s_page(vault):
    """'Colima', 'colima' and 'colima.md' are one file, and _safe_page_path is
    the only thing that knows it. A guard comparing raw strings would refuse
    the step its own page whenever the model added the extension."""
    writes = wiki_ingest._WriteCounter()
    dispatch = wiki_ingest._execute_dispatch(
        vault.path, "src.md", "colima", writes, exists=False
    )

    assert "written" in dispatch["write_wiki_page"](
        name="Colima.md", content="# Colima\n"
    )
    assert writes


def test_an_omitted_page_name_is_filled_in_rather_than_refused(vault):
    """There is exactly one page this step can mean, so filling it in beats
    spending one of twelve iterations saying so."""
    dispatch = wiki_ingest._execute_dispatch(
        vault.path, "src.md", "colima", wiki_ingest._WriteCounter(), exists=False
    )

    assert "written" in dispatch["write_wiki_page"](content="# Colima\n")
    assert (vault.root / "wiki" / "colima.md").read_text() == "# Colima\n"


def test_a_written_page_is_filed_even_when_the_model_never_files_it(vault, monkeypatch):
    """The guarantee this stage exists to keep. Filing used to be asked of the
    model and merely *checked* here; a step whose write landed and whose index
    call did not was retried, and when the retries ran out the page stayed on
    disk with nothing in the table of contents pointing at it. The live vault
    reached 181 such pages. The section comes from the plan, so nothing about
    filing needs the model."""
    vault.raw("source.md", "source text")
    plan = wiki_ingest._Plan()
    unit = {"name": "colima", "action": "create", "intent": "cover",
            "section": "Tools"}

    def write_only(**kwargs):
        kwargs["dispatch"]["write_wiki_page"](
            name="colima", content="# Colima\n\n**Summary**: covered\n"
        )
        return "wrote"

    monkeypatch.setattr(wiki_ingest, "run_agent", write_only)

    assert wiki_ingest._execute_unit(
        vault.path, "source.md", unit, plan, "rules", _Logger()
    )
    index = (Path(vault.path) / "wiki" / "index.md").read_text()
    assert "## Tools" in index
    assert "- [[colima]]" in index
    assert "## Unfiled" not in index


def test_a_step_that_writes_nothing_is_not_marked_done(vault, monkeypatch):
    """Filing no longer stands in for the write. writes.count means "the page
    was written" and nothing else, so a step that only touched the index must
    still fail — otherwise a stray update_index call would mark a page done
    that was never written."""
    vault.raw("source.md", "source text")
    vault.page("colima", "# Colima\n\n**Summary**: covered\n")
    plan = wiki_ingest._Plan()
    unit = {"name": "colima", "action": "update", "intent": "cover",
            "section": "Tools"}

    def index_only(**kwargs):
        kwargs["dispatch"]["update_index"](page="colima", section="Tools")
        return "filed but wrote nothing"

    monkeypatch.setattr(wiki_ingest, "run_agent", index_only)

    assert not wiki_ingest._execute_unit(
        vault.path, "source.md", unit, plan, "rules", _Logger()
    )


def test_a_page_with_no_planned_section_is_still_written(vault, monkeypatch):
    """A section this cannot use is a planning problem, not a reason to throw
    the page away. Retrying would send the identical call and fail identically,
    having rewritten the page to get there. _normalize_index lists it under
    Unfiled, which is what that section is for."""
    vault.raw("source.md", "source text")
    plan = wiki_ingest._Plan()
    unit = {"name": "colima", "action": "create", "intent": "cover", "section": ""}

    def write_only(**kwargs):
        kwargs["dispatch"]["write_wiki_page"](
            name="colima", content="# Colima\n\n**Summary**: covered\n"
        )
        return "wrote"

    monkeypatch.setattr(wiki_ingest, "run_agent", write_only)

    assert wiki_ingest._execute_unit(
        vault.path, "source.md", unit, plan, "rules", _Logger()
    )
    assert (Path(vault.path) / "wiki" / "colima.md").is_file()


def test_the_page_written_is_the_one_siblings_link_to(vault):
    """Why the guard refuses rather than redirecting quietly to the model's
    name. _execute_unit puts the batch's planned names in the prompt as the
    only links this step may use, so every sibling links to this page by its
    planned name — a step that wrote itself elsewhere would leave those links
    pointing at nothing, and stage 3 would log the planned name anyway."""
    dispatch = wiki_ingest._execute_dispatch(
        vault.path, "src.md", "colima", wiki_ingest._WriteCounter(), exists=False
    )

    dispatch["write_wiki_page"](name="COLIMA", content="# Colima\n")

    assert (vault.root / "wiki" / "colima.md").is_file()


def test_the_guard_reads_the_argument_the_tool_actually_names(vault):
    """update_index spells the page 'page', not 'name'. While the guard read
    only 'name', every update_index call the model made raised TypeError on the
    duplicate keyword, so no step could ever finish — the 2026-08-29 run burned
    all twelve iterations per page and left its sources unmarked."""
    vault.page("colima", "# Colima\n\n**Summary**: covered\n")
    writes = wiki_ingest._WriteCounter()
    dispatch = wiki_ingest._execute_dispatch(
        vault.path, "src.md", "colima", writes, exists=True
    )

    assert dispatch["update_index"](page="colima", section="Tools")["filed"] == "colima"


def test_a_wrong_page_is_refused_through_the_argument_the_tool_names(vault):
    """The other half: reading the real argument must also refuse a mismatch,
    or filing an unrelated page under this source becomes possible."""
    vault.page("podman", "# Podman\n")
    writes = wiki_ingest._WriteCounter()
    dispatch = wiki_ingest._execute_dispatch(
        vault.path, "src.md", "colima", writes, exists=True
    )

    error = dispatch["update_index"](page="podman", section="Tools")["error"]

    assert "colima" in error
    assert "page='colima'" in error
    assert not writes
