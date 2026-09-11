"""Tests for tools/compare_lint_models.py and the seed pack it builds.

No model is called here. What these cover is the two ways the comparison can
lie without anyone noticing: a seeded vault whose defects leak into the
structural pass, and a scorer that credits a model for a report it did not
write.
"""

import datetime
import json
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

import compare_lint_models as clm  # noqa: E402
import wiki_lint  # noqa: E402


@pytest.fixture
def seed():
    return clm.load_seed()


# --- the seed pack ----------------------------------------------------------


def test_seeded_vault_has_no_structural_findings(tmp_path, seed):
    """The whole comparison rests on this.

    Every planted defect is meant to need judgment. One that a Python check can
    see is scored twice — once by the structural pass for free, and again
    against the model — and the judgment number stops measuring judgment. The
    pack was built to satisfy check_links, check_orphans, check_index,
    check_format, check_duplicate_titles and check_template_twins all at once;
    this is what stops a later edit quietly breaking one of them.
    """
    vault = clm.build_vault(tmp_path / "v", seed)
    findings = wiki_lint.structural_findings(
        str(vault), today=datetime.date.today()
    )
    flat = [f"{section}: {item}"
            for section, items in findings.items() for item in items]
    assert flat == []


def test_every_page_named_in_seed_json_exists(seed):
    """A defect naming a page that was renamed or deleted is unscorable, and
    the failure is silent: the slug simply never matches, so the model looks
    worse than it is."""
    pages = set(clm._seed_pages(seed))
    for defect in seed["defects"]:
        for slug in defect["pages"]:
            assert slug in pages, f"{defect['id']} names missing page {slug}"
        if superseded := defect.get("superseded_by"):
            assert superseded in pages, f"{defect['id']} -> {superseded}"


def test_defect_ids_are_unique(seed):
    ids = [d["id"] for d in seed["defects"]]
    assert len(ids) == len(set(ids))


def test_each_judgment_category_has_at_least_three_defects(seed):
    """Four defects across four categories was the 2026-09-03 design, and it
    could not separate 4-of-12 from 2-of-12. One defect per category means a
    single lucky search decides the category."""
    counts: dict[str, int] = {}
    for defect in seed["defects"]:
        counts[defect["kind"]] = counts.get(defect["kind"], 0) + 1
    assert set(counts) == {
        "contradiction", "duplicate", "out_of_scope", "outdated"
    }
    assert min(counts.values()) >= 3, counts


def test_duplicate_pairs_do_not_share_a_title(seed):
    """check_duplicate_titles catches same-title pages in code. A duplicate
    pair that shares a title is therefore not a judgment defect at all."""
    pages = clm._seed_pages(seed)
    for defect in seed["defects"]:
        if defect["kind"] != "duplicate":
            continue
        titles = {wiki_lint._title(pages[s]) for s in defect["pages"]}
        assert len(titles) == len(defect["pages"]), defect["id"]


# --- building ---------------------------------------------------------------


def test_build_writes_rules_source_and_index(tmp_path, seed):
    vault = clm.build_vault(tmp_path / "v", seed)
    assert (vault / "RULES.md").is_file()
    assert (vault / "raw" / seed["raw_subdir"] / seed["raw_source"]).is_file()
    index = (vault / "wiki" / "index.md").read_text(encoding="utf-8")
    for slug in clm._seed_pages(seed):
        assert f"[[{slug}]]" in index


def test_index_descriptions_come_from_the_pages(tmp_path, seed):
    """The index description is read off each page's Summary line rather than
    repeated in seed.json, so the two cannot drift."""
    vault = clm.build_vault(tmp_path / "v", seed)
    index = (vault / "wiki" / "index.md").read_text(encoding="utf-8")
    page = (vault / "wiki" / "prompt-caching.md").read_text(encoding="utf-8")
    assert clm._summary_of(page) in index


def _fake_host(root: Path) -> Path:
    (root / "wiki").mkdir(parents=True)
    (root / "RULES.md").write_text("# Host rules\n", encoding="utf-8")
    (root / "wiki" / "index.md").write_text("# Host Index\n", encoding="utf-8")
    return root


def test_inject_keeps_the_host_rules(tmp_path, seed, monkeypatch):
    """Scope is the host vault's policy. Overwriting RULES.md would audit that
    wiki against the fixture's rules, which is a different question."""
    host = _fake_host(tmp_path / "host")
    monkeypatch.setattr(clm, "_git_archive",
                        lambda src, dest: clm.shutil.copytree(
                            src, dest, dirs_exist_ok=True))
    vault = clm.build_vault(tmp_path / "v", seed, inject_from=host)
    assert (vault / "RULES.md").read_text(encoding="utf-8") == "# Host rules\n"


def test_inject_refuses_to_overwrite_an_existing_page(tmp_path, seed, monkeypatch):
    """Silently replacing a real page would both destroy it in the copy and
    plant a defect the host's own content was already about."""
    host = _fake_host(tmp_path / "host")
    (host / "wiki" / "prompt-caching.md").write_text("# Theirs\n", encoding="utf-8")
    monkeypatch.setattr(clm, "_git_archive",
                        lambda src, dest: clm.shutil.copytree(
                            src, dest, dirs_exist_ok=True))
    with pytest.raises(SystemExit, match="prompt-caching"):
        clm.build_vault(tmp_path / "v", seed, inject_from=host)


def test_inject_refuses_a_section_heading_the_host_already_uses(
    tmp_path, seed, monkeypatch
):
    """Two headings for one section is itself a structural finding — see
    _index_section_twins. Planting one would fail the --check gate for a reason
    that has nothing to do with the pack."""
    host = _fake_host(tmp_path / "host")
    (host / "wiki" / "index.md").write_text(
        f"# Host Index\n\n## {seed['index_section']}\n", encoding="utf-8"
    )
    monkeypatch.setattr(clm, "_git_archive",
                        lambda src, dest: clm.shutil.copytree(
                            src, dest, dirs_exist_ok=True))
    with pytest.raises(SystemExit, match="index_section"):
        clm.build_vault(tmp_path / "v", seed, inject_from=host)


# --- scoring ----------------------------------------------------------------

DEFECTS = [
    {"id": "C1", "kind": "contradiction", "pages": ["alpha", "beta"]},
    {"id": "O1", "kind": "out_of_scope", "pages": ["gamma"]},
]


def test_a_defect_scores_only_when_one_finding_names_every_page():
    report = "\n1. alpha.md and beta.md disagree about the refresh interval.\n"
    assert clm.score_report(report, DEFECTS)["found"] == ["C1"]


def test_pages_named_in_separate_findings_do_not_score():
    """The defect is the relationship between the pages. A report that
    mentions both in unrelated findings has not found it, and crediting it
    would flatter whichever model writes the longest report."""
    report = "\n1. alpha.md is thin.\n2. beta.md could use an example.\n"
    assert clm.score_report(report, DEFECTS)["found"] == []


def test_a_single_page_defect_scores_on_that_page_alone():
    report = "\n1. gamma.md is out of scope under the Scope section.\n"
    assert clm.score_report(report, DEFECTS)["found"] == ["O1"]


def test_slugs_are_matched_with_or_without_the_md_suffix():
    report = "\n1. [[alpha]] contradicts beta — one says hourly, one nightly.\n"
    assert clm.score_report(report, DEFECTS)["found"] == ["C1"]


def test_a_page_named_by_its_title_in_prose_still_scores():
    """A model writing a sentence names the page the way the sentence needs it.
    Matching only hyphenated tokens scored this as a miss, which made every
    model look worse than it was for a reason that is about the scorer."""
    report = ("\n1. Alpha and Beta disagree: one says the index refreshes "
              "hourly, the other nightly.\n")
    assert clm.score_report(report, DEFECTS)["found"] == ["C1"]


def test_prose_with_no_numbered_list_scores_nothing():
    """RULES.md and LINT_WRAPPER both ask for a numbered list naming specific
    pages. A model that answers in paragraphs has not produced the artefact the
    weekly job exists to make, so zero is the honest score rather than a
    scoring gap to paper over."""
    report = "I looked at alpha.md and beta.md and they seem to disagree."
    assert clm.score_report(report, DEFECTS)["found"] == []


def test_structural_findings_above_the_marker_are_not_scored():
    """wiki_lint prints the structural findings as a numbered list too. Scoring
    those would credit the model for work Python did before it was called."""
    stdout = (
        "## Broken and self links\n\n"
        "1. alpha.md links to beta.md, which has no page.\n\n"
        "---\n\n"
        "## Judgment pass\n\n"
        "1. gamma.md is out of scope.\n"
    )
    score = clm.score_report(clm.judgment_text(stdout), DEFECTS)
    assert score["found"] == ["O1"]


def test_a_run_with_no_judgment_section_scores_nothing():
    assert clm.judgment_text("## Broken links\n\n1. alpha.md and beta.md\n") == ""


# --- run metrics ------------------------------------------------------------


def test_metrics_read_tool_calls_and_peak_context_from_stdout():
    stdout = (
        "2026-09-10 10:00:00,000 [INFO] prompt reached 8659 tokens on "
        "iteration 4 (13% of num_ctx=65536)\n"
        "2026-09-10 10:00:00,100 [INFO] tool_call search_wiki_pages({'q': 'a'}) -> {}\n"
        "2026-09-10 10:00:01,000 [INFO] prompt reached 28020 tokens on "
        "iteration 5 (43% of num_ctx=65536)\n"
        "2026-09-10 10:00:01,100 [INFO] tool_call read_wiki_page({'name': 'a'}) -> {}\n"
        "## Judgment pass\n\n1. gamma.md is out of scope.\n"
    )
    metrics = clm.run_metrics(stdout)
    assert metrics["tool_calls"] == 2
    assert metrics["peak_pct"] == 43
    assert metrics["num_ctx"] == 65536
    assert metrics["completed"] is True


def test_an_incomplete_run_is_not_counted_as_completed():
    """An [incomplete] judgment pass produced no report at all — that is what
    Gemini did in three of three runs at a cap of 60. It is a failure to
    finish, not a recall of zero, and the table separates the two."""
    stdout = ("## Judgment pass\n\n[incomplete: hit max_iterations=120 tool "
              "calls without reaching a final answer]\n")
    assert clm.run_metrics(stdout)["completed"] is False


def test_an_empty_judgment_section_is_not_counted_as_completed():
    assert clm.run_metrics("## Judgment pass\n\n")["completed"] is False


# --- the model is passed the way the plist would pass it --------------------


def test_run_once_sets_ollama_model_and_clears_a_stray_provider(monkeypatch):
    """The winner ships as an OLLAMA_MODEL line in the lint plist, which works
    because agent/__init__.py loads config/.env without override. Passing it
    the same way here means the comparison measures the arrangement it is
    deciding about. LLM_PROVIDER is cleared so a stray export in the operator's
    shell cannot silently score Gemini under a local model's name."""
    seen = {}

    def _fake_run(cmd, **kwargs):
        seen.update(kwargs["env"])
        return subprocess.CompletedProcess(cmd, 0, stdout="")

    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setattr(clm.subprocess, "run", _fake_run)
    clm.run_once(Path("/tmp/nope"), "gemma4:31b-mlx", timeout=1)

    assert seen["OLLAMA_MODEL"] == "gemma4:31b-mlx"
    assert "LLM_PROVIDER" not in seen


# --- the table --------------------------------------------------------------


def _row(model, found, completed=True, calls=20, pct=40, secs=100.0):
    return {
        "model": model,
        "seconds": secs,
        "score": {"found": found, "recall": len(found), "total": 12, "items": 5},
        "metrics": {"tool_calls": calls, "peak_pct": pct, "num_ctx": 65536,
                    "completed": completed},
    }


def test_table_says_false_findings_are_not_scored():
    """The one number this harness cannot produce is precision. Saying so in
    the output, every time, is what stops the recall column being read as a
    verdict."""
    lines = []
    clm.report_table([_row("a", ["C1"]), _row("a", ["C1", "O1"])], out=lines.append)
    assert any("FALSE FINDINGS ARE NOT SCORED" in l for l in lines)


def test_table_warns_when_there_are_fewer_than_three_trials():
    lines = []
    clm.report_table([_row("a", ["C1"])], out=lines.append)
    assert any("not deterministic" in l for l in lines)


def test_table_does_not_warn_at_three_trials():
    lines = []
    clm.report_table([_row("a", ["C1"])] * 3, out=lines.append)
    assert not any("not deterministic" in l for l in lines)


def test_results_json_is_written_without_the_captured_stdout(tmp_path, seed, monkeypatch):
    """The reports are saved as separate files a person reads. Duplicating
    every transcript into results.json makes the summary unreadable and the
    directory twice the size for nothing."""
    monkeypatch.setattr(clm, "structural_only",
                        lambda vault: (0, '{"sections": {}}'))
    monkeypatch.setattr(clm, "run_once", lambda vault, model, timeout: {
        "model": model, "seconds": 1.0, "exit_code": 0,
        "stdout": "## Judgment pass\n\n1. gamma.md is out of scope.\n",
    })
    clm.main(["--models", "m", "--trials", "1", "--workdir", str(tmp_path)])
    saved = json.loads((tmp_path / "results.json").read_text(encoding="utf-8"))
    assert "stdout" not in saved["trials"][0]
    assert saved["trials"][0]["score"]["total"] == len(seed["defects"])
    assert (tmp_path / "reports" / "m.1.txt").is_file()


def test_a_bold_or_bulleted_number_is_still_a_finding():
    """qwen3.8:27b-mlx writes '**7. ...**'; two trials scored 0 because of it."""
    report = (
        "## Judgment pass\n\n"
        "**1. Out-of-scope page — dental-plan-enrollment**\n"
        "Entirely about dental benefits. Fix: delete the page.\n\n"
        "- 2) Duplicate concept — batch-embedding\n"
        "Covered twice.\n\n"
        "### 3. Outdated claim — quantization-builds\n"
        "Says 4-bit is the only build.\n"
    )
    items = clm.numbered_items(clm.judgment_text(report))
    assert len(items) == 3
    assert "dental-plan-enrollment" in items[0]
    assert "batch-embedding" in items[1]
    assert "quantization-builds" in items[2]


def test_prose_with_no_numbers_still_scores_zero_after_the_bold_fix():
    """The widened pattern must not turn a wall of prose into findings."""
    report = (
        "## Judgment pass\n\n"
        "The vault looks broadly consistent. The dental-plan-enrollment page "
        "is off topic and batch-embedding repeats itself.\n"
    )
    assert clm.numbered_items(clm.judgment_text(report)) == []


def test_the_run_complete_log_line_is_not_scored_as_part_of_a_finding():
    """setup_logger's trailing record lands inside the last numbered finding."""
    report = (
        "## Judgment pass\n\n"
        "2026-09-10 18:37:56,673 [INFO] tool_call search_wiki_pages({'query': 'x'})\n"
        "1. Out-of-scope page — dental-plan-enrollment. Fix: delete the page.\n"
        "2026-09-10 18:43:59,112 [INFO] Wiki lint run complete — 35 pages checked\n"
    )
    items = clm.numbered_items(clm.judgment_text(report))
    assert len(items) == 1
    assert "Wiki lint run complete" not in items[0]
    assert "tool_call" not in clm.judgment_text(report)


def test_findings_in_names_every_section_and_text():
    raw = json.dumps({"sections": {"Page format": ["n8n.md uses its slug"],
                                   "Orphan pages": [], "Escaped text": ["a", "b"]}})
    assert clm.findings_in(raw) == {
        "Page format: n8n.md uses its slug",
        "Escaped text: a",
        "Escaped text: b",
    }


def test_a_host_vaults_own_finding_is_not_blamed_on_the_pack():
    """--inject into a real vault stopped on n8n.md, which the pack never touched."""
    before = clm.findings_in(json.dumps(
        {"sections": {"Page format": ["n8n.md uses its slug as the title"]}}))
    after = clm.findings_in(json.dumps(
        {"sections": {"Page format": ["n8n.md uses its slug as the title"]}}))
    assert sorted(after - before) == []


def test_a_finding_the_pack_adds_is_still_caught():
    before = clm.findings_in(json.dumps({"sections": {"Page format": ["n8n.md"]}}))
    after = clm.findings_in(json.dumps(
        {"sections": {"Page format": ["n8n.md"], "Orphan pages": ["dental-plan-enrollment.md"]}}))
    assert sorted(after - before) == ["Orphan pages: dental-plan-enrollment.md"]
