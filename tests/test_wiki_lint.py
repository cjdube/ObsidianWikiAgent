"""Tests for the deterministic structural checks in wiki_lint.py.

The pure checks (check_links, check_orphans, check_duplicate_titles) take a
{slug: content} dict directly; check_index and check_format also read the
vault, so those use the fixture. The model --deep pass is not exercised here.
"""

import datetime
import json
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


def test_check_index_reports_the_unfiled_backlog_as_one_finding(vault):
    """The hand-edit shape: a section heading is deleted, and _normalize_index
    silently recomputes its pages under Unfiled on the next update_index."""
    for slug in ("a", "b"):
        vault.page(slug, good_page())
    vault.index("# Index\n\n## Unfiled\n\n- [[a]] x\n- [[b]] y\n")
    findings = wl.check_index(vault.path, wl._pages(vault.path))
    assert len(findings) == 1
    assert findings[0].startswith("2 page(s) sit under the index's Unfiled heading (a, b)")


def test_check_index_truncates_a_long_unfiled_backlog(vault):
    slugs = [f"p{i}" for i in range(7)]
    for slug in slugs:
        vault.page(slug, good_page())
    entries = "".join(f"- [[{slug}]] x\n" for slug in slugs)
    vault.index(f"# Index\n\n## Unfiled\n\n{entries}")
    findings = wl.check_index(vault.path, wl._pages(vault.path))
    assert len(findings) == 1
    assert "(p0, p1, p2, p3, p4, and 2 more)" in findings[0]


def test_check_index_stops_counting_at_the_next_heading(vault):
    """Unfiled is not always last: a later heading ends it, and its pages are
    filed, not part of the backlog."""
    for slug in ("stray", "filed"):
        vault.page(slug, good_page())
    vault.index(
        "# Index\n\n## Unfiled\n\n- [[stray]] x\n\n## Tools\n\n- [[filed]] y\n"
    )
    findings = wl.check_index(vault.path, wl._pages(vault.path))
    assert len(findings) == 1
    assert findings[0].startswith("1 page(s) sit under the index's Unfiled heading (stray)")


def test_check_index_reports_two_headings_that_name_one_section(vault):
    """The real shape, from the live vault 2026-08-26 to 2026-09-07: a hyphen
    made a second 'AI & Agent Development', and the page under it was stranded
    while every other check stayed clean."""
    for slug in ("a", "b"):
        vault.page(slug, good_page())
    vault.index(
        "# Index\n\n## AI & Agent Development\n\n- [[a]] x\n"
        "\n## AI & Agent-Development\n\n- [[b]] y\n"
    )
    findings = wl.check_index(vault.path, wl._pages(vault.path))
    assert len(findings) == 1
    assert "index.md has 2 headings for one section" in findings[0]
    assert "'AI & Agent Development', 'AI & Agent-Development'" in findings[0]


def test_check_index_accepts_sections_that_are_genuinely_different(vault):
    for slug in ("a", "b"):
        vault.page(slug, good_page())
    vault.index("# Index\n\n## Tools\n\n- [[a]] x\n\n## Projects\n\n- [[b]] y\n")
    assert wl.check_index(vault.path, wl._pages(vault.path)) == []


def test_check_index_ignores_headings_that_are_not_sections(vault):
    """'# Index' is the title and '### Sub' is not a section — _section_bounds
    only recognises '## ', so neither can strand a page."""
    vault.page("a", good_page())
    vault.index("# Index\n\n## Tools\n\n### Tools\n\n- [[a]] x\n")
    assert wl.check_index(vault.path, wl._pages(vault.path)) == []


# --- check_source_coverage -------------------------------------------------


def ingested(vault, *names):
    (vault.root / "wiki" / ".ingested.json").write_text(json.dumps(list(names)))


def test_check_source_coverage_flags_a_dated_source_with_no_page(vault):
    """The 2026-08-03 shape: marked done having written only topic pages."""
    vault.raw("AI-Chat-Learnings-2026-08-03.md", "notes", subdir="daily-ai")
    vault.page("ollama", good_page())
    ingested(vault, "AI-Chat-Learnings-2026-08-03.md")
    findings = wl.check_source_coverage(vault.path, wl._pages(vault.path))
    assert len(findings) == 1
    assert "AI-Chat-Learnings-2026-08-03.md" in findings[0]


def test_check_source_coverage_accepts_the_page_however_it_is_spelled(vault):
    vault.raw("Strategic-Weekly-Review-2026-05-11.md", "notes")
    vault.page("strategic-weekly-review-2026-05-11", good_page())
    ingested(vault, "Strategic-Weekly-Review-2026-05-11.md")
    assert wl.check_source_coverage(vault.path, wl._pages(vault.path)) == []


def test_check_source_coverage_ignores_a_source_still_in_the_queue(vault):
    """Not yet ingested is not a gap — it is waiting its turn."""
    vault.raw("AI-Chat-Learnings-2026-08-15.md", "notes")
    ingested(vault)
    assert wl.check_source_coverage(vault.path, wl._pages(vault.path)) == []


def test_check_source_coverage_ignores_an_undated_source(vault):
    """RULES.md step 4 gives an article no page of its own, by design."""
    vault.raw("landing-the-plane.md", "notes", subdir="clippings")
    vault.page("zeigarnik-effect", good_page())
    ingested(vault, "landing-the-plane.md")
    assert wl.check_source_coverage(vault.path, wl._pages(vault.path)) == []


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


def test_structural_findings_walks_raw_only_once(vault, monkeypatch):
    """check_source_coverage and check_format both need the raw/ listing, and
    check_format needs it twice over (readable plus binary). Each used to fetch
    its own, so a lint probed every file in raw/ for a NUL byte three times."""
    vault.raw("note.txt")
    vault.page("a", good_page())
    calls = []
    real = wl.scan_raw
    monkeypatch.setattr(wl, "scan_raw", lambda p: calls.append(p) or real(p))

    wl.structural_findings(vault.path, today=TODAY)

    assert len(calls) == 1


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


def test_check_and_fix_agree_on_a_link_inside_a_code_span(vault):
    """A complete [[a]] inside backticks is a link to both of them. The
    backtick exclusion stops a stray `[[` from *opening* a runaway match; it
    does not exempt a well-formed link that happens to sit in a code span, and
    the link body here contains no backtick to exclude.

    So the assertion is agreement, not exemption: the check reports it and the
    fix repairs it. Those two used to run different patterns."""
    vault.page("a", "# A\n\nThe `[[a]]` syntax links to a page.\n")
    pages = wl._pages(vault.path)

    assert any("links to itself" in f for f in wl.check_links(pages))

    changes = wl.apply_safe_fixes(vault.path, pages)

    assert any("a.md" in c for c in changes)
    assert wl.check_links(wl._pages(vault.path)) == []


def test_apply_safe_fixes_strips_a_self_link_after_a_stray_open_bracket(vault):
    """The other half of the same disagreement: a stray `[[` in a code span
    earlier in the body used to swallow the real self-link that followed, so
    --fix reported no changes while check_links kept flagging it every run."""
    vault.page("a", "# A\n\nUnclosed `[[` brackets are bad.\n\nSee [[a]] too.\n")
    pages = wl._pages(vault.path)

    assert any("links to itself" in f for f in wl.check_links(pages))

    changes = wl.apply_safe_fixes(vault.path, pages)
    content = (vault.root / "wiki" / "a.md").read_text()

    assert "[[a]]" not in content
    assert "`[[` brackets are bad" in content  # the code span is untouched
    assert any("a.md" in c for c in changes)
    # What the check reports and what the fix repairs now agree.
    assert wl.check_links(wl._pages(vault.path)) == []


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


def test_apply_safe_fixes_decodes_escaped_text(vault):
    vault.page("vibe", '# Vibe\n\n**Summary**: A \\"vibe\\" shift.\n')
    pages = wl._pages(vault.path)

    changes = wl.apply_safe_fixes(vault.path, pages)

    assert (vault.root / "wiki" / "vibe.md").read_text() == (
        '# Vibe\n\n**Summary**: A "vibe" shift.\n'
    )
    assert changes == ["vibe.md: decoded escaped text on 1 line"]


def test_apply_safe_fixes_reports_the_line_count(vault):
    vault.page("x", '# X\n\nA \\"one\\".\n\nB \\"two\\".\n')
    changes = wl.apply_safe_fixes(vault.path, wl._pages(vault.path))
    assert changes == ["x.md: decoded escaped text on 2 lines"]


def test_apply_safe_fixes_honours_the_decode_exemptions(vault):
    """The exemptions come from the shared function, so --fix inherits them and
    cannot corrupt what the check deliberately stays quiet about."""
    fence = '# Hooks\n\n```json\n{"c": "prettier \\"$FILE\\""}\n```\n'
    prose = "# OWA\n\nFixed JSON escape sequences (`\\u2019`) in the vault.\n"
    vault.page("hooks", fence)
    vault.page("owa", prose)

    changes = wl.apply_safe_fixes(vault.path, wl._pages(vault.path))

    assert changes == []
    assert (vault.root / "wiki" / "hooks.md").read_text() == fence
    assert (vault.root / "wiki" / "owa.md").read_text() == prose


def test_apply_safe_fixes_writes_a_page_once_for_both_fixes(vault):
    """A page needing the decode and a self-link strip gets one write and two
    lines in the log, not two writes."""
    vault.page("a", '# A\n\nSee [[a]] about \\"scope\\".\n')
    pages = wl._pages(vault.path)

    changes = wl.apply_safe_fixes(vault.path, pages)
    content = (vault.root / "wiki" / "a.md").read_text()

    assert '"scope"' in content and "[[a]]" not in content
    assert changes == [
        "a.md: decoded escaped text on 1 line",
        "a.md: removed 1 self-link",
    ]
    # pages is updated in place so the checks that run next see the fixed text.
    assert pages["a"] == content


def test_check_and_fix_agree_on_escaped_text(vault):
    """What check_escaped_text reports is exactly what --fix repairs — the two
    read the same function, so neither can name a page the other ignores."""
    vault.page("damaged", '# D\n\nA \\"quote\\".\n')
    vault.page("fenced", '# F\n\n```json\n{"a": "\\"b\\""}\n```\n')
    pages = wl._pages(vault.path)

    flagged = {f.split(".md")[0] for f in wl.check_escaped_text(pages)}
    fixed = {c.split(".md")[0] for c in wl.apply_safe_fixes(vault.path, pages)}

    assert flagged == fixed == {"damaged"}
    assert wl.check_escaped_text(pages) == []   # nothing left to report


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


def test_check_lens_frontmatter_ignores_a_dated_log_that_mentions_the_tool():
    """A day's record naming the tool is a note, not a self-description."""
    page = (
        "# AI Chat Learnings 2026-08-03\n\n"
        "- Updated `docs/tool-loading.md` to include `evaluate_against`.\n"
    )
    assert wl.check_lens_frontmatter({"ai-chat-learnings-2026-08-03": page}) == []


# --- check_escaped_text -----------------------------------------------------


def test_check_escaped_text_flags_an_escaped_quote():
    page = '# Vibe Coding\n\n**Summary**: A \\"vibe\\" shift.\n'
    findings = wl.check_escaped_text({"vibe-coding": page})
    assert len(findings) == 1
    assert "JSON escaping as literal text" in findings[0]
    assert "line 3" in findings[0]


def test_check_escaped_text_counts_the_lines_it_found():
    page = '# X\n\nA \\"one\\".\n\nB \\"two\\".\n'
    assert "2 lines, first at 3" in wl.check_escaped_text({"x": page})[0]


def test_check_escaped_text_flags_a_unicode_escape():
    page = "# FPL\n\nTypically 5\\u201310 hours a week.\n"
    assert len(wl.check_escaped_text({"fpl": page})) == 1


def test_check_escaped_text_passes_a_clean_page():
    page = '# Colima\n\n**Summary**: A "container" runtime.\n'
    assert wl.check_escaped_text({"colima": page}) == []


def test_check_escaped_text_exempts_a_json_fence():
    """Inside a JSON string the backslash is syntax, not damage. The check gets
    this from the guard it shares rather than restating the rule."""
    page = (
        '# Hooks\n\n```json\n'
        '{"command": "prettier --write \\"$FILE\\""}\n'
        '```\n'
    )
    assert wl.check_escaped_text({"claude-code-hooks": page}) == []


def test_check_escaped_text_exempts_a_line_about_escaping():
    page = "# OWA\n\nFixed JSON escape sequences (`\\u2019`) across the vault.\n"
    assert wl.check_escaped_text({"obsidian-wiki-agent": page}) == []


def test_check_escaped_text_agrees_with_the_ingest_guard():
    """The check and the code that repairs it must not be able to disagree
    about what counts — a report naming pages the fix then leaves alone is
    worse than no report."""
    from agent.wiki_tools import _quotes_outside_json

    pages = {
        "damaged": '# X\n\nA \\"quote\\".\n',
        "clean": '# Y\n\nA "quote".\n',
        "json-fence": '# Z\n\n```json\n{"a": "\\"b\\""}\n```\n',
    }
    flagged = {f.split(".md")[0] for f in wl.check_escaped_text(pages)}
    would_change = {s for s, c in pages.items() if _quotes_outside_json(c) != c}
    assert flagged == would_change == {"damaged"}


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


def _clean_sweep(monkeypatch):
    """Stub the scope sweep's one model call so a --deep test exercises the
    judgment pass without 569 HTTP requests. It answers clean, because these
    tests are about the pass around it, not about the verdict."""
    monkeypatch.setattr(wl, "complete_text", lambda **kw: "VERDICT: CLEAN")


def test_deep_pass_gives_the_judgment_agent_its_full_iteration_allowance(vault, monkeypatch):
    """A literal here regressed once already: at 60 the pass ran out of tool
    calls and returned [incomplete] instead of a report, so the allowance is a
    named constant and this asserts the agent actually receives it."""
    seen = {}
    vault.page("a", good_page())
    vault.index("# Index\n\n- [[a]] s.\n")

    def _capture(**kw):
        seen.update(kw)
        return "no findings"

    _clean_sweep(monkeypatch)
    monkeypatch.setattr(wl, "run_agent", _capture)
    monkeypatch.setattr("sys.argv", _lint_argv(vault, "--deep"))
    wl.main()

    assert seen["max_iterations"] == wl.MAX_DEEP_ITERATIONS > 60


def test_main_starts_a_budget_only_for_the_deep_pass(vault, monkeypatch):
    """The structural pass is local Python and consults no budget, so starting
    one for it would bound nothing. --deep is the pass that can hang."""
    from agent import budget

    vault.page("a", good_page())
    vault.index("# Index\n\n- [[a]] s.\n")

    monkeypatch.setattr("sys.argv", _lint_argv(vault))
    wl.main()
    assert budget.remaining() is None

    _clean_sweep(monkeypatch)
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

    _clean_sweep(monkeypatch)
    monkeypatch.setattr(wl, "run_agent", _wedged)
    vault.page("a", good_page())
    vault.index("# Index\n\n- [[a]] s.\n")
    monkeypatch.setattr("sys.argv", _lint_argv(vault, "--deep"))

    assert wl.main() == 1
    assert len(pushed) == 1
    assert "abandoned" in pushed[0][1]


def test_main_treats_incomplete_deep_judgment_as_failure(vault, monkeypatch):
    pushed = []
    monkeypatch.setattr(wl, "notify_failure", lambda *a, **k: pushed.append(1))
    vault.page("a", good_page())
    vault.index("# Index\n\n- [[a]] s.\n")
    _clean_sweep(monkeypatch)
    monkeypatch.setattr(wl, "run_agent", lambda **kw: "[incomplete: hit max_iterations=60 tool calls without reaching a final answer]")
    monkeypatch.setattr("sys.argv", _lint_argv(vault, "--deep"))

    assert wl.main() == 1
    assert pushed == [1]


# --- the scope sweep (Pass A) -----------------------------------------------
#
# The sweep exists because the judgment pass samples. Measured 2026-09-11 over
# a 70-page slice of the live vault it opened 24 of 70, reported the wiki
# clean, and stopped with 20 minutes of budget unspent. These pin the parts
# that make "every page was judged" true rather than merely claimed.


def _sweep_vault(vault, n=3):
    names = [chr(ord("a") + i) for i in range(n)]
    for name in names:
        vault.page(name, good_page())
    vault.index("# Index\n\n" + "".join(f"- [[{x}]] s.\n" for x in names))
    return names


def test_the_sweep_judges_every_page_not_a_sample(vault, monkeypatch):
    """The whole point. One call per page, driven by a for-loop, so coverage
    cannot depend on the model deciding it has seen enough."""
    names = _sweep_vault(vault, 5)
    judged = []

    def _fake(*, user_prompt, **kw):
        judged.append(user_prompt.splitlines()[0])
        return "VERDICT: CLEAN"

    monkeypatch.setattr(wl, "complete_text", _fake)
    monkeypatch.setattr(wl, "run_agent", lambda **kw: "no findings")
    monkeypatch.setattr("sys.argv", _lint_argv(vault, "--deep"))
    wl.main()

    assert judged == [f"Page: {n}" for n in names]


def test_the_sweep_turns_thinking_off(vault, monkeypatch):
    """Measured 2026-09-11 on qwen3.8:27b-mlx: 46s a page with reasoning, 0.4s
    without — 7.3 hours against 72 minutes over this vault. The tokens never
    reach the usage ledger, so a regression here shows up only as wall clock."""
    _sweep_vault(vault, 1)
    seen = {}

    def _fake(**kw):
        seen.update(kw)
        return "VERDICT: CLEAN"

    monkeypatch.setattr(wl, "complete_text", _fake)
    monkeypatch.setattr(wl, "run_agent", lambda **kw: "no findings")
    monkeypatch.setattr("sys.argv", _lint_argv(vault, "--deep"))
    wl.main()

    assert seen["think"] is False


def test_an_unreadable_verdict_is_never_counted_as_clean(vault, monkeypatch, capsys):
    """The failure this design exists to kill. A reply the parser cannot read
    is a page nobody judged, and folding it into the clean count prints the
    same false all-clear as not reading the page at all — so the report has to
    name it."""
    _sweep_vault(vault, 2)
    replies = iter(["VERDICT: CLEAN", "The page looks fine to me."])

    monkeypatch.setattr(wl, "complete_text", lambda **kw: next(replies))
    monkeypatch.setattr(wl, "run_agent", lambda **kw: "no findings")
    monkeypatch.setattr("sys.argv", _lint_argv(vault, "--deep"))
    wl.main()

    out = capsys.readouterr().out
    assert "No readable verdict: b" in out
    assert "1 gave no readable verdict and were NOT judged." in out


def test_the_sweep_reports_the_pages_it_flagged(vault, monkeypatch, capsys):
    _sweep_vault(vault, 2)
    replies = iter([
        "VERDICT: CLEAN",
        "The Scope section excludes recipes.\n1. b is a recipe; delete it.\n"
        "VERDICT: FINDING",
    ])

    monkeypatch.setattr(wl, "complete_text", lambda **kw: next(replies))
    monkeypatch.setattr(wl, "run_agent", lambda **kw: "no findings")
    monkeypatch.setattr("sys.argv", _lint_argv(vault, "--deep"))
    wl.main()

    out = capsys.readouterr().out
    assert "## Scope sweep" in out
    assert "1. b — b is a recipe; delete it." in out
    # Neither the bookkeeping line nor the reasoning is part of the finding a
    # human reads weekly.
    assert "VERDICT:" not in out
    assert "excludes recipes" not in out


def test_the_finding_is_the_last_numbered_item_not_the_first():
    """The model reasons on its way to the answer and that reasoning is often
    itself a numbered list. Taken from the fixture, where this page's reasoning
    numbered three topics before naming the finding."""
    reply = (
        "The page covers three topics.\n"
        "1. Quantized build: in scope.\n"
        "2. Reranker swap: in scope.\n"
        "3. Volunteer day: excluded.\n"
        "\n"
        "1. Page b includes a volunteer day section; remove it.\n"
        "VERDICT: FINDING"
    )
    assert wl._sweep_body(reply) == "Page b includes a volunteer day section; remove it."


def test_a_finding_with_no_numbered_item_is_kept():
    """A parser that goes looking for the finding can come back empty, and an
    empty finding reads as a clean page. Verbose beats missing."""
    reply = "This page is about catering, which the Scope excludes.\nVERDICT: FINDING"
    assert wl._sweep_body(reply) == (
        "This page is about catering, which the Scope excludes."
    )


def test_reasoning_numbered_above_a_prose_conclusion_is_not_the_finding():
    """The shape that broke the 569-page run. The model worked through the
    Scope lines as a numbered checklist, then concluded in prose underneath, so
    reading the last numbered line of the whole reply put a fragment of the
    checklist at the head of the report:

        9. google-cloud-platform — Is it "Meetings, social..."? No.

    A conclusion below a blank line is the finding. The checklist is not."""
    reply = (
        '1. Is it "AARP..."? No.\n'
        '2. Is it "Meetings, social..."? No.\n'
        "\n"
        "The page records administrative tasks.\n"
        "\n"
        "Therefore the subject falls under the exclusion of "
        '"Enterprise productivity administration".\n'
        "VERDICT: FINDING"
    )
    assert wl._sweep_body(reply) == (
        "Therefore the subject falls under the exclusion of "
        '"Enterprise productivity administration".'
    )


def test_the_numbered_item_still_wins_inside_the_last_paragraph():
    """The common shape: one sentence of reasoning and the numbered item under
    it, no blank line between. Only the numbered item belongs in the report, so
    the paragraph rule must not undo the numbered-item rule."""
    reply = (
        "Scope has no line about cooking.\n"
        "\n"
        "The Scope section excludes recipes.\n"
        "1. b is a recipe; delete it.\n"
        "VERDICT: FINDING"
    )
    assert wl._sweep_body(reply) == "b is a recipe; delete it."


def test_a_flagged_page_is_never_left_without_text():
    """scope_sweep prints the page name beside this body, but an empty body
    still reads as an all-clear at a glance. Whatever the model wrote, some of
    it reaches the report."""
    for reply in ("Out of scope.\nVERDICT: FINDING",
                  "a\n\nb\n\nc\nVERDICT: FINDING",
                  "1. x\nVERDICT: FINDING"):
        assert wl._sweep_body(reply)


def test_a_page_past_the_ceiling_abandons_the_run(vault, monkeypatch):
    """A per-run deadline cannot protect the shared Ollama slot across 569
    separate calls: a server that wedges on call three sits inside the whole
    budget without finishing anything. The ceiling catches that at one page."""
    from agent import budget

    _sweep_vault(vault, 2)
    clock = iter([0.0, wl.SWEEP_PAGE_CEILING_SECONDS + 1])
    monkeypatch.setattr(wl.time, "monotonic", lambda: next(clock))
    monkeypatch.setattr(wl, "complete_text", lambda **kw: "VERDICT: CLEAN")

    with pytest.raises(budget.BudgetExceeded) as e:
        wl.scope_sweep({"a": "x", "b": "y"}, "rules")
    assert "'a'" in str(e.value)


def test_each_page_gets_its_own_retry_ceiling(vault, monkeypatch):
    """Counting retries across the whole sweep would let a wedged server stay
    under every per-call cap while the total ran to hundreds — the exact
    product agent/budget.py was written to stop."""
    from agent import budget

    started = []
    monkeypatch.setattr(
        budget, "start_source",
        lambda name, retries: started.append((name, retries)),
    )
    monkeypatch.setattr(wl, "complete_text", lambda **kw: "VERDICT: CLEAN")

    wl.scope_sweep({"a": "x", "b": "y"}, "rules")

    assert started == [
        ("scope sweep: a", wl.MAX_SWEEP_RETRIES),
        ("scope sweep: b", wl.MAX_SWEEP_RETRIES),
    ]


def test_the_sweep_logs_every_page_as_it_goes(monkeypatch):
    """This pass can run for hours and the log is the only thing anyone can
    look at while it does. With no progress line a working sweep and a wedged
    one are both just silence."""
    logged = []

    class _Log:
        def info(self, line):
            logged.append(line)

    monkeypatch.setattr(wl, "complete_text", lambda **kw: "VERDICT: CLEAN")
    wl.scope_sweep({"a": "x", "b": "y"}, "rules", _Log())

    assert len(logged) == 2
    assert logged[0].startswith("Scope sweep 1/2 CLEAN a")
    assert logged[1].startswith("Scope sweep 2/2 CLEAN b")


def test_the_sweep_line_says_it_covers_one_category(vault):
    """"Swept 569 of 569" alone reads as all four judgment checks. Three of the
    four need a second page, and this pass holds one."""
    line = wl._sweep_line({wl.SWEEP_CLEAN: 8, wl.SWEEP_FINDING: 1,
                           wl.SWEEP_UNPARSED: 0})
    assert "9 of 9 pages" in line
    assert "Out-of-scope check only." in line
    assert "NOT judged" not in line


def test_the_sweep_line_admits_pages_it_could_not_judge(vault):
    line = wl._sweep_line({wl.SWEEP_CLEAN: 7, wl.SWEEP_FINDING: 0,
                           wl.SWEEP_UNPARSED: 2})
    assert "8 of 8" not in line
    assert "9 of 9 pages" in line
    assert "2 gave no readable verdict and were NOT judged." in line


def test_check_format_accepts_a_citation_to_a_binary_source(vault):
    """A PNG in raw/ is hidden from the model but is still a real file, so a
    page citing it has not invented anything."""
    vault.raw("scan.png", content="\x00\x01binary")
    findings = wl.check_format(vault.path, {"a": good_page(sources="scan.png")}, TODAY)
    assert findings == []


# --- --json (the UI path) --------------------------------------------------


def _json_out(capsys) -> dict:
    return json.loads(capsys.readouterr().out)


def test_json_reports_findings_as_sections(vault, monkeypatch, capsys):
    vault.raw("note.txt")
    vault.page("orphan", good_page(title="Orphan"))
    vault.index("# Index\n\n- [[orphan]]\n")
    monkeypatch.setattr("sys.argv", ["wiki_lint.py", "--vault", vault.path, "--json"])

    assert wl.main() == 1                     # 1 means findings, not failure
    payload = _json_out(capsys)
    assert payload["pages"] == 1
    assert payload["fixes"] == []
    assert any("orphan.md is an orphan" in f for f in payload["sections"]["Orphan pages"])


def test_json_clean_vault_returns_zero_with_every_section_present(vault, monkeypatch, capsys):
    vault.raw("note.txt")
    vault.page("a", good_page(title="A") + "\n[[b]]\n")
    vault.page("b", good_page(title="B") + "\n[[a]]\n")
    vault.index("# Index\n\n- [[a]]\n- [[b]]\n")
    monkeypatch.setattr("sys.argv", ["wiki_lint.py", "--vault", vault.path, "--json"])

    assert wl.main() == 0
    sections = _json_out(capsys)["sections"]
    # Clean sections are still keys, so a UI can show what was checked.
    assert sections["Orphan pages"] == []
    assert set(sections) == set(wl.structural_findings(vault.path))


def test_json_writes_no_log(vault, monkeypatch, capsys, tmp_path):
    """The whole reason --json exists as a separate path: a button press must not
    fabricate a run in LocalLLMAgent's dashboard."""
    vault.page("a", good_page())
    monkeypatch.setattr("sys.argv", ["wiki_lint.py", "--vault", vault.path, "--json"])
    wl.main()
    capsys.readouterr()
    assert not list(tmp_path.glob("wiki_lint.*.log"))


def test_json_fix_applies_then_reports_the_fixed_state(vault, monkeypatch, capsys):
    vault.raw("note.txt")
    vault.page("a", good_page(title="A") + "\nsee [[a]]\n")
    vault.page("orphan", good_page(title="Orphan"))
    vault.index("# Index\n\n- [[a]]\n- [[orphan]]\n")
    monkeypatch.setattr(
        "sys.argv", ["wiki_lint.py", "--vault", vault.path, "--json", "--fix"]
    )

    assert wl.main() == 1
    payload = _json_out(capsys)
    assert any("self-link" in c for c in payload["fixes"])
    assert "[[a]]" not in (vault.root / "wiki" / "a.md").read_text()
    # The self-link is gone from the report; the judgment call remains.
    assert payload["sections"]["Broken and self links"] == []
    assert payload["sections"]["Orphan pages"]


def test_json_refuses_deep(vault, monkeypatch, capsys):
    monkeypatch.setattr(
        "sys.argv", ["wiki_lint.py", "--vault", vault.path, "--json", "--deep"]
    )
    assert wl.main() == 2
    assert "cannot be combined" in capsys.readouterr().err


def test_json_missing_vault_is_an_error_object_not_a_crash(vault, monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(
        "sys.argv", ["wiki_lint.py", "--vault", str(tmp_path / "gone"), "--json"]
    )
    assert wl.main() == 2
    assert "error" in _json_out(capsys)


# --- judgment-pass coverage ------------------------------------------------


def test_coverage_line_reports_the_share_read():
    assert wl._coverage_line(27, 582) == (
        "Judgment pass read 27 of 582 pages (4.6%). "
        "This is a sample, not a sweep."
    )


def test_coverage_line_survives_an_empty_vault():
    assert wl._coverage_line(0, 0).startswith("Judgment pass read 0 of 0 pages.")


def test_pages_read_counts_each_page_once(vault):
    vault.page("a", good_page(title="A"))
    dispatch = wl.query_dispatch(vault.path)
    counter = wl._PagesRead(dispatch)

    assert "content" in dispatch["read_wiki_page"](name="a")
    dispatch["read_wiki_page"](name="A.md")

    assert len(counter) == 1


def test_pages_read_ignores_a_page_that_does_not_exist(vault):
    dispatch = wl.query_dispatch(vault.path)
    counter = wl._PagesRead(dispatch)

    assert "error" in dispatch["read_wiki_page"](name="missing")

    assert len(counter) == 0


# --- the judgment pass's own tool set ---------------------------------------


def test_lint_advertises_list_wiki_pages_and_can_dispatch_it(vault):
    """Under measurement: the pass used to be told no such tool existed, so it
    guessed its search terms. Advertising a tool nothing dispatches is an
    unknown-tool error and a wasted iteration, so the two are asserted together."""
    advertised = {t["function"]["name"] for t in wl.LINT_TOOL_SCHEMAS}
    assert "list_wiki_pages" in advertised
    assert advertised <= set(wl.lint_dispatch(vault.path))


def test_the_query_path_is_not_given_the_page_list(vault):
    """The deviation is the judgment pass's alone. wiki_query.py answers one
    question at a time and has no reason for a result sized by the vault."""
    from agent.wiki_tools import QUERY_TOOL_SCHEMAS

    assert "list_wiki_pages" not in {
        t["function"]["name"] for t in QUERY_TOOL_SCHEMAS
    }


def test_the_prompt_no_longer_denies_the_page_list():
    """LINT_WRAPPER told the model 'You have no tool that returns the index'.
    Leaving that in place beside the tool is how a stage ends up never calling
    what it was given."""
    assert "no tool that returns the index" not in wl.LINT_WRAPPER
    assert "list_wiki_pages" in wl.LINT_WRAPPER
