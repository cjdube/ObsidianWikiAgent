"""Tests for the deterministic structural checks in wiki_lint.py.

The pure checks (check_links, check_orphans, check_duplicate_titles) take a
{slug: content} dict directly; check_index and check_format also read the
vault, so those use the fixture. The model --deep pass is not exercised here.
"""

import datetime

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
