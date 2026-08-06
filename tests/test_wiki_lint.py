"""Tests for the deterministic structural checks in wiki_lint.py.

The pure checks (check_links, check_orphans, check_duplicate_titles) take a
{slug: content} dict directly; check_index and check_format also read the
vault, so those use the fixture. The model --deep pass is not exercised here.
"""

import datetime
import re

import pytest

import wiki_lint as wl


TODAY = datetime.date(2026, 7, 24)


def good_page(title="Real Title", summary="s", sources="note.txt", updated="2026-07-01"):
    return (
        f"# {title}\n\n"
        f"**Summary**: {summary}\n"
        f"**Sources**: {sources}\n"
        f"**Last updated**: {updated}\n"
    )


# --- check_links -----------------------------------------------------------


def test_check_links_flags_self_and_broken():
    findings = wl.check_links({"a": "see [[a]] and [[missing]]"})
    assert any("links to itself" in f for f in findings)
    assert any("[[missing]]" in f for f in findings)


def test_check_links_clean():
    assert wl.check_links({"a": "[[b]]", "b": "[[a]]"}) == []


# --- check_orphans ---------------------------------------------------------


def test_check_orphans_flags_unlinked_page():
    findings = wl.check_orphans({"a": "[[b]]", "b": "[[a]]", "c": "nothing links here"})
    assert len(findings) == 1
    assert "c.md is an orphan" in findings[0]


def test_check_orphans_self_link_does_not_rescue():
    findings = wl.check_orphans({"lonely": "see [[lonely]]"})
    assert any("lonely.md is an orphan" in f for f in findings)


def test_check_orphans_exempts_dated_capture_pages():
    pages = {
        "concept": "nothing links here",
        "daily-chrome-2026-07-22": "a browsing log nothing links to",
        "ai-chat-learnings-2026-07-16": "a chat log nothing links to",
        "strategic-weekly-review-2026-05-11": "a weekly review nothing links to",
    }
    findings = wl.check_orphans(pages)
    # The concept page is still flagged; dated chronological logs are exempt.
    assert any("concept.md is an orphan" in f for f in findings)
    assert not any("daily-chrome-2026-07-22" in f for f in findings)
    assert not any("ai-chat-learnings-2026-07-16" in f for f in findings)
    assert not any("strategic-weekly-review-2026-05-11" in f for f in findings)


def test_check_orphans_dated_page_still_counts_as_a_linker():
    # A dated log is exempt from being reported, but its outbound links still
    # rescue the concept pages it points at.
    findings = wl.check_orphans(
        {"concept": "no inbound", "daily-chrome-2026-07-22": "see [[concept]]"}
    )
    assert findings == []


# --- check_index -----------------------------------------------------------


def test_check_index_missing_and_dangling(vault):
    vault.page("present", good_page())
    vault.index("# Index\n\n- [[gone]]\n")  # 'present' not listed; 'gone' has no page
    pages = wl._pages(vault.path)
    findings = wl.check_index(vault.path, pages)
    assert any("present.md is not linked from index.md" in f for f in findings)
    assert any("[[gone]]" in f and "no page" in f for f in findings)


def test_check_index_clean(vault):
    vault.page("a", good_page())
    vault.index("# Index\n\n- [[a]]\n")
    assert wl.check_index(vault.path, wl._pages(vault.path)) == []


# --- check_format ----------------------------------------------------------


def test_check_format_clean(vault):
    vault.raw("note.txt")
    findings = wl.check_format(vault.path, {"real-page": good_page()}, TODAY)
    assert findings == []


def test_check_format_missing_title(vault):
    findings = wl.check_format(vault.path, {"a": "no heading\n\n**Summary**: s\n"}, TODAY)
    assert any("no '# Title'" in f for f in findings)


def test_check_format_slug_as_title(vault):
    vault.raw("note.txt")
    page = good_page(title="my-slug")
    findings = wl.check_format(vault.path, {"my-slug": page}, TODAY)
    assert any("uses its slug as the title" in f for f in findings)


@pytest.mark.parametrize("label", ["Summary", "Sources", "Last updated"])
def test_check_format_missing_field(vault, label):
    page = good_page()
    # Drop the one field under test.
    page = "\n".join(l for l in page.splitlines() if not l.startswith(f"**{label}**"))
    findings = wl.check_format(vault.path, {"a": page + "\n"}, TODAY)
    assert any(f"missing its '**{label}**:'" in f for f in findings)


def test_check_format_non_iso_date(vault):
    vault.raw("note.txt")
    findings = wl.check_format(vault.path, {"a": good_page(updated="July 1 2026")}, TODAY)
    assert any("non-ISO 'Last updated'" in f for f in findings)


@pytest.mark.parametrize("updated", ["2026-13-45", "2026-02-30", "0000-00-00"])
def test_check_format_impossible_date_is_reported_not_raised(vault, updated):
    # Well-formed enough to pass the \d{4}-\d{2}-\d{2} shape check, but not a
    # real date — this used to abort the whole run inside fromisoformat.
    vault.raw("note.txt")
    findings = wl.check_format(vault.path, {"a": good_page(updated=updated)}, TODAY)
    assert any("non-ISO 'Last updated'" in f for f in findings)


def test_check_format_future_date(vault):
    vault.raw("note.txt")
    findings = wl.check_format(vault.path, {"a": good_page(updated="2026-12-31")}, TODAY)
    assert any("in the future" in f for f in findings)


def test_check_format_citation_with_directory_prefix(vault):
    vault.raw("note.txt")
    findings = wl.check_format(vault.path, {"a": good_page(sources="sub/note.txt")}, TODAY)
    assert any("directory prefix" in f for f in findings)


def test_check_format_citation_missing_source(vault):
    # raw/ is empty, so the cited file does not exist.
    findings = wl.check_format(vault.path, {"a": good_page(sources="ghost.txt")}, TODAY)
    assert any("not a file in raw/" in f for f in findings)


# A clipping is named after its article's title, so a comma inside the filename
# is routine. The separating commas and this one are the same character.
COMMA_NAME = "if-you're-still-hitting-the-wall,-you're-doing-it-wrong.md"


def test_check_format_citation_filename_containing_comma(vault):
    vault.raw(COMMA_NAME, subdir="clippings")
    findings = wl.check_format(vault.path, {"a": good_page(sources=COMMA_NAME)}, TODAY)
    assert findings == []


def test_check_format_citation_comma_filename_among_others(vault):
    vault.raw("note.txt")
    vault.raw(COMMA_NAME, subdir="clippings")
    vault.raw("other.md")
    sources = f"note.txt, {COMMA_NAME}, other.md"
    assert wl.check_format(vault.path, {"a": good_page(sources=sources)}, TODAY) == []


def test_cited_sources_splits_unresolvable_citation_on_its_commas():
    # Nothing in raw/ to rejoin against, so the fragments stay split and are
    # each reported — an invented citation is still caught.
    assert wl._cited_sources("ghost,-part-two.md", set()) == ["ghost", "-part-two.md"]


def test_cited_sources_prefers_the_longest_run_that_resolves():
    raw = {"a.md", "b.md", "a.md, b.md"}
    assert wl._cited_sources("a.md, b.md", raw) == ["a.md, b.md"]


# --- check_slug_typos ------------------------------------------------------


def test_check_slug_typos_flags_the_misspelled_slug():
    # The real 2026-08-05 defect: titled 'Ollama', filed as 'olloma'.
    pages = {"olloma-thread-wedges": good_page(title="Ollama Thread Wedges")}
    findings = wl.check_slug_typos(pages)
    assert any("looks like a typo" in f for f in findings)


@pytest.mark.parametrize(
    "slug,title",
    [
        # Separator-only differences, all legitimate and all seen in the vault.
        ("innovators-dilemma", "Innovator's Dilemma"),
        ("local-llm-agent", "LocalLLMAgent"),
        ("context-routing-tradeoffs", "Context Routing Trade-offs"),
        ("destructive_command_guard", "Destructive Command Guard"),
        ("osxkeychain-dark-wake-issues", "OSX Keychain Dark Wake Issues"),
        # A deliberately shortened slug is two characters off, not one.
        ("hybrid-agent-architecture", "Hybrid AI Agent Architecture"),
        # A retitle is nowhere near its slug.
        ("ai-chat-learnings-2026-07-02", "AI Chat Learnings - July 2, 2026"),
    ],
)
def test_check_slug_typos_ignores_legitimate_divergence(slug, title):
    assert wl.check_slug_typos({slug: good_page(title=title)}) == []


def test_check_slug_typos_skips_untitled_pages():
    # check_format already reports the missing heading; don't double-report.
    assert wl.check_slug_typos({"a": "no heading\n"}) == []


# --- check_duplicate_titles ------------------------------------------------


def test_check_duplicate_titles():
    pages = {"a": "# Same Thing\n", "b": "# same thing\n", "c": "# Different\n"}
    findings = wl.check_duplicate_titles(pages)
    assert len(findings) == 1
    assert "a.md and b.md" in findings[0]


# --- check_template_twins --------------------------------------------------


def twin(name, other, updated="2026-07-15"):
    """A page in the shape that fable.md and sol.md actually had: one template
    with the subject swapped, each twin linking to the other."""
    return (
        f"# {name.title()}\n\n"
        f"**Summary**: A third-party API wrapper and router.\n"
        f"**Sources**: note.txt\n"
        f"**Last updated**: {updated}\n\n---\n\n"
        f"{name.title()} acts as an intermediary between agents and providers.\n\n"
        f"## Related pages\n\n- [[claude-code]]\n- [[{other}]]\n"
    )


def test_check_template_twins_flags_name_swapped_pages():
    findings = wl.check_template_twins({"fable": twin("fable", "sol"),
                                        "sol": twin("sol", "fable")})
    assert len(findings) == 1
    assert "fable.md and sol.md" in findings[0]


def test_check_template_twins_ignores_differing_last_updated():
    # A twin edited a day later is still a twin; the date is metadata.
    findings = wl.check_template_twins({"fable": twin("fable", "sol"),
                                        "sol": twin("sol", "fable", updated="2026-08-02")})
    assert len(findings) == 1


def test_check_template_twins_leaves_genuinely_different_pages():
    pages = {"fable": twin("fable", "sol"),
             "ollama": good_page(title="Ollama", summary="Local model runner.")}
    assert wl.check_template_twins(pages) == []


def test_check_template_twins_ignores_pages_too_short_to_judge():
    # Two near-empty pages match on structure alone and prove nothing.
    assert wl.check_template_twins({"a": "# A\n", "b": "# B\n"}) == []


# --- structural_findings + exit code ---------------------------------------


def test_structural_findings_accepts_prebuilt_pages(vault):
    vault.page("orphan", good_page())
    pages = wl._pages(vault.path)
    findings = wl.structural_findings(vault.path, today=TODAY, pages=pages)
    assert findings["Orphan pages"]  # the lone page links to nothing


def test_main_header_is_dated(vault, monkeypatch, capsys):
    # Each run in the appended launchd log must be self-dating.
    monkeypatch.setattr("sys.argv", ["wiki_lint.py", "--vault", vault.path])
    wl.main()
    out = capsys.readouterr().out
    today = datetime.date.today().isoformat()
    assert f"# Wiki lint — {vault.root.name} — {today}" in out


def test_main_clean_vault_returns_zero(vault, monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["wiki_lint.py", "--vault", vault.path])
    assert wl.main() == 0
    assert "No structural problems found." in capsys.readouterr().out


def test_main_dirty_vault_returns_one(vault, monkeypatch, capsys):
    vault.page("orphan", good_page())
    monkeypatch.setattr("sys.argv", ["wiki_lint.py", "--vault", vault.path])
    assert wl.main() == 1
    assert "orphan" in capsys.readouterr().out


# --- run logging (what LocalLLMAgent's dashboard parses) -------------------


def _log_lines(vault, tmp_path):
    return (tmp_path / f"wiki_lint.{vault.root.name}.log").read_text().splitlines()


def test_main_logs_run_boundaries(vault, monkeypatch, capsys, tmp_path):
    # LocalLLMAgent parses run history out of these two markers: a "Starting …
    # run" line opens a run, a "… run complete" line closes it successfully.
    monkeypatch.setattr("sys.argv", ["wiki_lint.py", "--vault", vault.path])
    wl.main()

    lines = _log_lines(vault, tmp_path)
    assert any(f"Starting wiki lint run for vault: {vault.path}" in ln for ln in lines)
    assert any("Wiki lint run complete" in ln for ln in lines)


def test_findings_log_at_info_not_warning(vault, monkeypatch, capsys, tmp_path):
    # A weekly lint result is a dashboard read, not an alert — and WARNING is
    # what LocalLLMAgent's log_inspector pushes to the phone each morning.
    vault.page("orphan", good_page())
    monkeypatch.setattr("sys.argv", ["wiki_lint.py", "--vault", vault.path])
    assert wl.main() == 1

    lines = _log_lines(vault, tmp_path)
    assert any("Orphan pages: " in ln for ln in lines)
    assert not any("[WARNING]" in ln or "[ERROR]" in ln for ln in lines)


def test_dirty_run_still_logs_as_complete(vault, monkeypatch, capsys, tmp_path):
    # Findings mean the lint WORKED. Only a crash is a failed run; logging them
    # as failures would paint every week red on the dashboard.
    vault.page("orphan", good_page())
    monkeypatch.setattr("sys.argv", ["wiki_lint.py", "--vault", vault.path])
    wl.main()

    complete = [ln for ln in _log_lines(vault, tmp_path) if "run complete" in ln]
    assert len(complete) == 1
    # One orphan page trips three checks (orphan, index, invented citation).
    assert "1 pages checked, 3 structural findings" in complete[0]


def test_each_log_record_is_one_line(vault, monkeypatch, capsys, tmp_path):
    # A multi-line record is read as a traceback continuation and lands in the
    # run's error field, painting a clean run as a failure.
    vault.page("orphan", good_page())
    monkeypatch.setattr("sys.argv", ["wiki_lint.py", "--vault", vault.path])
    wl.main()

    stamp = re.compile(r"^\d{4}-\d\d-\d\d \d\d:\d\d:\d\d,\d+ \[\w+\] ")
    assert all(stamp.match(ln) for ln in _log_lines(vault, tmp_path))


# --- --fix (safe, mechanical auto-fixes only) ------------------------------


def test_apply_safe_fixes_strips_self_links(vault):
    vault.page("a", "# A\n\nSee [[a]] and [[a|myself]] and [[b]].\n")
    vault.page("b", "# B\n")
    pages = wl._pages(vault.path)
    changes = wl.apply_safe_fixes(vault.path, pages)
    content = (vault.root / "wiki" / "a.md").read_text()
    assert "[[a]]" not in content            # self-link removed
    assert "[[a|myself]]" not in content
    assert "myself" in content               # alias preserved as plain text
    assert "[[b]]" in content                # link to another real page kept
    assert any("a.md" in c for c in changes)


def test_apply_safe_fixes_delinks_dead_index_links(vault):
    vault.page("real", good_page())
    vault.index("# Index\n\n- [[real]] keep me\n- [[ghost]] dead entry\n")
    pages = wl._pages(vault.path)
    wl.apply_safe_fixes(vault.path, pages)
    index = (vault.root / "wiki" / "index.md").read_text()
    assert "[[real]]" in index               # valid TOC link kept
    assert "[[ghost]]" not in index          # dead link removed
    assert "dead entry" in index             # description text preserved


def test_apply_safe_fixes_leaves_judgment_items_alone(vault):
    # Broken page-body links are create-vs-delink calls; orphans are judgment.
    # --fix must touch neither.
    vault.raw("note.txt")
    vault.page("orphan", good_page())
    vault.page("linker", good_page() + "\nsee [[missing-concept]]\n")
    pages = wl._pages(vault.path)
    changes = wl.apply_safe_fixes(vault.path, pages)
    linker = (vault.root / "wiki" / "linker.md").read_text()
    assert "[[missing-concept]]" in linker   # broken body link untouched
    assert changes == []                     # nothing mechanical to fix


def test_apply_safe_fixes_clean_vault_no_changes(vault):
    vault.page("a", "# A\n\n[[b]]\n")
    vault.page("b", "# B\n\n[[a]]\n")
    assert wl.apply_safe_fixes(vault.path, wl._pages(vault.path)) == []


def test_main_fix_applies_then_reports_remaining(vault, monkeypatch, capsys):
    vault.raw("note.txt")
    vault.page("a", good_page(title="A") + "\nsee [[a]]\n")   # self-link: fixable
    vault.page("orphan", good_page(title="Orphan"))            # orphan: judgment
    vault.index("# Index\n\n- [[a]]\n- [[orphan]]\n")
    monkeypatch.setattr("sys.argv", ["wiki_lint.py", "--vault", vault.path, "--fix"])
    rc = wl.main()
    out = capsys.readouterr().out
    # The self-link was fixed on disk...
    assert "[[a]]" not in (vault.root / "wiki" / "a.md").read_text()
    # ...and the fix was reported...
    assert "self-link" in out.lower()
    # ...while the judgment item still surfaces and drives a nonzero exit.
    assert "orphan.md is an orphan" in out
    assert rc == 1


# --- lens frontmatter ------------------------------------------------------

_LENS_BODY = (
    "# AI Slop\n\n**Summary**: The lens applied when a draft is judged "
    "(consumed by LocalLLMAgent's `evaluate_against`).\n"
)


def test_check_lens_frontmatter_flags_stripped_marker():
    findings = wl.check_lens_frontmatter({"ai-slop": _LENS_BODY})
    assert len(findings) == 1
    assert "no 'lens: true' frontmatter" in findings[0]


def test_check_lens_frontmatter_accepts_intact_marker():
    page = "---\nlens: true\ndescription: standards\n---\n\n" + _LENS_BODY
    assert wl.check_lens_frontmatter({"ai-slop": page}) == []


def test_check_lens_frontmatter_flags_frontmatter_without_the_marker():
    """Frontmatter surviving isn't enough — the marker itself must be there."""
    page = "---\ndescription: standards\n---\n\n" + _LENS_BODY
    assert len(wl.check_lens_frontmatter({"ai-slop": page})) == 1


def test_check_lens_frontmatter_ignores_ordinary_pages():
    page = "# Colima\n\n**Summary**: A container runtime.\n"
    assert wl.check_lens_frontmatter({"colima": page}) == []


# --- main(): run boundaries, alerting, and the --deep budget ----------------


def _lint_argv(vault, *extra):
    return ["wiki_lint.py", "--vault", vault.path, *extra]


def test_main_is_quiet_when_it_only_finds_problems(vault, monkeypatch):
    """Findings are a weekly read, not a 7am phone push. The run still exits
    non-zero for a human or a CI check, but it must not alert."""
    pushed = []
    monkeypatch.setattr(wl, "notify_failure", lambda *a, **k: pushed.append(1))
    vault.page("orphan", good_page())
    vault.index("# Index\n")
    monkeypatch.setattr("sys.argv", _lint_argv(vault))

    assert wl.main() == 1  # findings -> non-zero
    assert pushed == []


def test_main_pushes_an_alert_when_the_run_crashes(vault, monkeypatch):
    # A weekly unattended audit that dies leaves no trace anyone reads: the
    # report just doesn't appear, which looks the same as a clean week.
    pushed = []
    monkeypatch.setattr(
        wl, "notify_failure",
        lambda job, detail, logger=None: pushed.append((job, str(detail))),
    )
    monkeypatch.setattr(wl, "_pages", lambda _: (_ for _ in ()).throw(OSError("disk")))
    monkeypatch.setattr("sys.argv", _lint_argv(vault))

    assert wl.main() == 1
    assert len(pushed) == 1
    assert pushed[0][0] == f"wiki_lint[{vault.root.name}]"
    assert "disk" in pushed[0][1]


def test_main_starts_a_budget_only_for_the_deep_pass(vault, monkeypatch):
    """The structural pass is local Python and consults no budget, so starting
    one for it would bound nothing. --deep is the pass that can hang."""
    from agent import budget

    vault.page("a", good_page())
    vault.index("# Index\n\n- [[a]] s.\n")

    monkeypatch.setattr("sys.argv", _lint_argv(vault))
    wl.main()
    assert budget.remaining() is None

    monkeypatch.setattr(wl, "run_agent", lambda **kw: "no findings")
    monkeypatch.setattr("sys.argv", _lint_argv(vault, "--deep"))
    wl.main()
    assert budget.remaining() is not None
    assert budget.remaining() <= wl.DEEP_RUN_BUDGET_MINUTES * 60


def test_main_turns_an_exhausted_budget_into_an_alert(vault, monkeypatch):
    from agent import budget

    pushed = []
    monkeypatch.setattr(
        wl, "notify_failure",
        lambda job, detail, logger=None: pushed.append((job, str(detail))),
    )

    def _wedged(**kwargs):
        raise budget.BudgetExceeded("run budget exhausted before model request")

    monkeypatch.setattr(wl, "run_agent", _wedged)
    vault.page("a", good_page())
    vault.index("# Index\n\n- [[a]] s.\n")
    monkeypatch.setattr("sys.argv", _lint_argv(vault, "--deep"))

    assert wl.main() == 1
    assert len(pushed) == 1
    assert "abandoned" in pushed[0][1]


def test_check_format_accepts_a_citation_to_a_binary_source(vault):
    """A PNG in raw/ is hidden from the model but is still a real file, so a
    page citing it has not invented anything."""
    vault.raw("scan.png", content="\x00\x01binary")
    findings = wl.check_format(vault.path, {"a": good_page(sources="scan.png")}, TODAY)
    assert findings == []
