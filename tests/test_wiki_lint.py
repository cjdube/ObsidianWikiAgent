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


# --- check_duplicate_titles ------------------------------------------------


def test_check_duplicate_titles():
    pages = {"a": "# Same Thing\n", "b": "# same thing\n", "c": "# Different\n"}
    findings = wl.check_duplicate_titles(pages)
    assert len(findings) == 1
    assert "a.md and b.md" in findings[0]


# --- structural_findings + exit code ---------------------------------------


def test_structural_findings_accepts_prebuilt_pages(vault):
    vault.page("orphan", good_page())
    pages = wl._pages(vault.path)
    findings = wl.structural_findings(vault.path, today=TODAY, pages=pages)
    assert findings["Orphan pages"]  # the lone page links to nothing


def test_main_clean_vault_returns_zero(vault, monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["wiki_lint.py", "--vault", vault.path])
    assert wl.main() == 0
    assert "No structural problems found." in capsys.readouterr().out


def test_main_dirty_vault_returns_one(vault, monkeypatch, capsys):
    vault.page("orphan", good_page())
    monkeypatch.setattr("sys.argv", ["wiki_lint.py", "--vault", vault.path])
    assert wl.main() == 1
    assert "orphan" in capsys.readouterr().out
