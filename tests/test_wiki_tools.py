"""Tests for the pure file-I/O and parsing core in agent/wiki_tools.py."""

import json
import os
import time

import pytest

from agent import wiki_tools as wt


# --- path safety -----------------------------------------------------------


def test_safe_page_path_accepts_plain_name(vault):
    path = wt._safe_page_path(vault.path, "speakers-bureau")
    assert path.name == "speakers-bureau.md"
    assert path.parent == (vault.root / "wiki").resolve()


def test_safe_page_path_appends_md_only_when_missing(vault):
    assert wt._safe_page_path(vault.path, "foo.md").name == "foo.md"
    assert wt._safe_page_path(vault.path, "foo").name == "foo.md"


@pytest.mark.parametrize("bad", ["../evil", "../../etc/passwd", "/etc/passwd"])
def test_safe_page_path_rejects_escape(vault, bad):
    with pytest.raises(ValueError):
        wt._safe_page_path(vault.path, bad)


def test_safe_raw_path_accepts_plain_and_subdir(vault):
    assert wt._safe_raw_path(vault.path, "note.txt").name == "note.txt"
    # A sort subdirectory is still inside raw/, so it's allowed.
    assert wt._safe_raw_path(vault.path, "daily-notes/note.txt").name == "note.txt"


@pytest.mark.parametrize("bad", ["../secret", "../../etc/passwd", "/etc/passwd"])
def test_safe_raw_path_rejects_escape(vault, bad):
    with pytest.raises(ValueError):
        wt._safe_raw_path(vault.path, bad)


def test_read_raw_file_rejects_traversal(vault):
    result = wt.read_raw_file(vault.path, "../../etc/passwd")
    assert "error" in result
    assert "outside" in result["error"]


def test_read_raw_file_reads_and_falls_back_to_subdir(vault):
    vault.raw("top.txt", "top-level content")
    vault.raw("nested.txt", "nested content", subdir="daily-notes")
    assert wt.read_raw_file(vault.path, "top.txt")["content"] == "top-level content"
    # Not at the top level anymore — found by basename under raw/.
    assert wt.read_raw_file(vault.path, "nested.txt")["content"] == "nested content"


# --- link parsing ----------------------------------------------------------


@pytest.mark.parametrize(
    "content,expected",
    [
        ("see [[foo]]", {"foo"}),
        ("see [[foo.md]]", {"foo"}),
        ("see [[foo|alias]]", {"foo"}),
        ("see [[foo#section]]", {"foo"}),
        ("[[a]] and [[b|x]] and [[c.md]]", {"a", "b", "c"}),
        ("no links here", set()),
    ],
)
def test_linked_page_names(content, expected):
    assert wt._linked_page_names(content) == expected


# --- index maintenance -----------------------------------------------------


def test_strip_unfiled_removes_only_that_section():
    content = (
        "# Index\n\n## Topics\n\n- [[a]]\n\n"
        f"{wt.UNFILED_HEADING}\n\n- [[b]]\n"
    )
    stripped = wt._strip_unfiled(content)
    assert "[[a]]" in stripped
    assert wt.UNFILED_HEADING not in stripped
    assert "[[b]]" not in stripped
    assert "## Topics" in stripped


def test_page_summary_flattens_wiki_links(vault):
    vault.page("a", "# A\n\n**Summary**: relates to [[b|Big B]] and [[c]].\n")
    assert wt._page_summary(vault.path, "a") == "relates to Big B and c."


def test_page_summary_missing_returns_empty(vault):
    vault.page("a", "# A\n\nNo summary line here.\n")
    assert wt._page_summary(vault.path, "a") == ""


# --- _normalize_index: the guarantees every index write carries -------------


def test_normalize_appends_missing_pages_under_unfiled(vault):
    vault.page("linked", "# Linked\n\n**Summary**: a linked page.\n")
    vault.page("dropped", "# Dropped\n\n**Summary**: model forgot me.\n")

    index, unfiled, _ = wt._normalize_index(
        vault.path, "# Index\n\n## Topics\n\n- [[linked]]\n")

    assert wt.UNFILED_HEADING in index
    assert "[[dropped]]" in index
    # The already-linked page is not duplicated into Unfiled.
    assert "[[linked]]" not in index.split(wt.UNFILED_HEADING, 1)[1]
    assert unfiled == 1


def test_normalize_no_unfiled_when_all_linked(vault):
    vault.page("a", "# A\n\n**Summary**: s.\n")
    index, unfiled, _ = wt._normalize_index(vault.path, "## Topics\n\n- [[a]]\n")
    assert wt.UNFILED_HEADING not in index
    assert unfiled == 0


def test_normalize_delinks_broken_links(vault):
    vault.page("real", "# Real\n\n**Summary**: s.\n")
    # One valid link and one garbled/dead link.
    index, _, delinked = wt._normalize_index(
        vault.path,
        "# Index\n\n## Topics\n\n- [[real]] the good one\n"
        "- [[ai-chat--2026-07-15]] a garbled slug\n",
    )
    assert "[[real]]" in index                       # valid link kept
    assert "[[ai-chat--2026-07-15]]" not in index    # dead link removed
    assert "a garbled slug" in index                 # description preserved
    assert "ai-chat--2026-07-15" in index            # de-linked to plain text
    assert delinked == 1


def test_normalize_delink_keeps_alias_text(vault):
    index, _, delinked = wt._normalize_index(vault.path, "## T\n\n- [[gone|Old Name]] desc\n")
    assert "[[gone|Old Name]]" not in index
    assert "Old Name" in index      # the alias the reader saw is kept as text
    assert delinked == 1


def test_normalize_keeps_valid_links_and_unfiled_intact(vault):
    # De-linking must not disturb valid links or the Unfiled guarantee.
    vault.page("linked", "# Linked\n\n**Summary**: s.\n")
    vault.page("dropped", "# Dropped\n\n**Summary**: s.\n")
    index, unfiled, delinked = wt._normalize_index(
        vault.path, "## T\n\n- [[linked]]\n- [[ghost]] not a page\n")
    assert "[[linked]]" in index
    assert "[[ghost]]" not in index
    assert "[[dropped]]" in index          # still appended under Unfiled
    assert unfiled == 1
    assert delinked == 1


def test_normalize_backfills_a_bare_entry_from_the_page(vault):
    # How a stripped index repairs itself: the model wrote index.md down to bare
    # links on 2026-08-04, and the descriptions come back from the pages.
    vault.page("a", "# A\n\n**Summary**: what the page is about.\n")
    index, _, _ = wt._normalize_index(vault.path, "## Topics\n\n- [[a]]\n")
    assert "- [[a]] what the page is about." in index


def test_normalize_does_not_overwrite_an_existing_description(vault):
    # Filling blanks only, so a description curated by hand in Obsidian
    # survives the next ingest.
    vault.page("a", "# A\n\n**Summary**: the page's own wording.\n")
    index, _, _ = wt._normalize_index(vault.path, "## Topics\n\n- [[a]] my own wording\n")
    assert "- [[a]] my own wording" in index
    assert "the page's own wording" not in index


# --- update_index: one page, one section ------------------------------------


def test_update_index_files_one_page_under_a_section(vault):
    vault.page("ollama", "# Ollama\n\n**Summary**: runs local models.\n")
    result = wt.update_index(vault.path, "ollama", "Tools")

    index = (vault.root / "wiki" / "index.md").read_text()
    assert "## Tools" in index
    assert "- [[ollama]] runs local models." in index
    assert result["filed"] == "ollama"


def test_update_index_leaves_the_rest_of_the_file_alone(vault):
    """The whole point: one call edits one entry, it does not rewrite the file.

    The old whole-document signature is what let a truncated generation
    overwrite 91 curated descriptions in a single call.
    """
    vault.page("old", "# Old\n\n**Summary**: s.\n")
    vault.page("new", "# New\n\n**Summary**: s.\n")
    vault.index("# My Wiki\n\n## Established\n\n- [[old]] a description I wrote\n")

    wt.update_index(vault.path, "new", "Established")

    index = (vault.root / "wiki" / "index.md").read_text()
    assert "# My Wiki" in index                        # preamble intact
    assert "- [[old]] a description I wrote" in index  # neighbour untouched
    assert "- [[new]] s." in index


def test_update_index_refiling_moves_rather_than_duplicates(vault):
    vault.page("a", "# A\n\n**Summary**: s.\n")
    wt.update_index(vault.path, "a", "First")
    wt.update_index(vault.path, "a", "Second")

    index = (vault.root / "wiki" / "index.md").read_text()
    assert index.count("[[a]]") == 1
    assert "- [[a]] s." in index.split("## Second", 1)[1]


def test_update_index_sorts_within_a_section(vault):
    for name in ("zebra", "apple", "mango"):
        vault.page(name, f"# {name}\n\n**Summary**: s.\n")
        wt.update_index(vault.path, name, "Fruit")

    section = (vault.root / "wiki" / "index.md").read_text().split("## Fruit", 1)[1]
    assert [wt._entry_name(ln) for ln in section.splitlines() if wt._entry_name(ln)] \
        == ["apple", "mango", "zebra"]


def test_update_index_new_section_goes_before_unfiled(vault):
    vault.page("filed", "# Filed\n\n**Summary**: s.\n")
    vault.page("loose", "# Loose\n\n**Summary**: s.\n")  # never filed -> Unfiled

    wt.update_index(vault.path, "filed", "Topics")

    index = (vault.root / "wiki" / "index.md").read_text()
    assert index.index("## Topics") < index.index(wt.UNFILED_HEADING)


def test_update_index_rejects_a_page_that_does_not_exist(vault):
    result = wt.update_index(vault.path, "ghost", "Topics")
    assert "error" in result
    assert not (vault.root / "wiki" / "index.md").exists()


def test_update_index_rejects_filing_into_unfiled(vault):
    # Unfiled is computed, not authored — filing into it would be undone by the
    # next normalize and reads as "leave this uncategorised".
    vault.page("a", "# A\n\n**Summary**: s.\n")
    result = wt.update_index(vault.path, "a", "Unfiled")
    assert "error" in result


def test_update_index_matches_an_existing_heading_case_insensitively(vault):
    vault.page("a", "# A\n\n**Summary**: s.\n")
    vault.page("b", "# B\n\n**Summary**: s.\n")
    vault.index("## Tools & Runtimes\n\n- [[a]] s.\n")

    wt.update_index(vault.path, "b", "tools & runtimes")

    index = (vault.root / "wiki" / "index.md").read_text()
    assert index.count("## Tools & Runtimes") == 1
    assert index.lower().count("## tools & runtimes") == 1


# --- move / sort -----------------------------------------------------------


def test_move_raw_file_valid(vault):
    vault.raw("note.txt")
    result = wt.move_raw_file(vault.path, "note.txt", "daily-notes")
    assert result == {"moved": "daily-notes/note.txt"}
    assert (vault.root / "raw" / "daily-notes" / "note.txt").is_file()
    assert not (vault.root / "raw" / "note.txt").exists()


@pytest.mark.parametrize("folder", ["a/b", "..", "", ".", "a\\b"])
def test_move_raw_file_rejects_bad_folder(vault, folder):
    vault.raw("note.txt")
    assert "error" in wt.move_raw_file(vault.path, "note.txt", folder)


def test_move_raw_file_rejects_filename_escape(vault):
    # A traversing filename must not let the model relocate an outside file.
    outside = vault.root / "outside.txt"
    outside.write_text("secret", encoding="utf-8")
    result = wt.move_raw_file(vault.path, "../outside.txt", "misc")
    assert "error" in result
    assert outside.is_file()  # untouched


# --- RULES.md parsing ------------------------------------------------------


def test_parse_raw_folders(vault):
    folders = wt.parse_raw_folders(vault.path)
    names = [f["name"] for f in folders]
    assert names == ["daily-notes", "misc"]
    assert folders[0]["description"] == "Notes captured day to day."


def test_parse_raw_folders_absent_section(tmp_path):
    (tmp_path / "RULES.md").write_text("# Rules\n\nNo raw folders here.\n")
    assert wt.parse_raw_folders(str(tmp_path)) == []


# --- listing / dotfile handling --------------------------------------------


def test_list_wiki_pages_skips_reserved_and_dotfiles(vault):
    vault.page("real", "# Real\n")
    vault.page("index", "# Index\n")
    vault.page("log", "log\n")
    (vault.root / "wiki" / "._real.md").write_text("appledouble", encoding="utf-8")
    assert wt.list_wiki_pages(vault.path)["pages"] == ["real.md"]


def test_list_raw_files_recurses_and_skips_dotfiles(vault):
    vault.raw("a.txt")
    vault.raw("b.txt", subdir="daily-notes")
    (vault.root / "raw" / ".hidden").write_text("x", encoding="utf-8")
    assert wt.list_raw_files(vault.path)["files"] == ["a.txt", "b.txt"]


def test_list_raw_files_is_oldest_first_not_alphabetical(vault):
    """The queue order decides who starves when a run hits its budget.

    Alphabetical looks neutral and is not: feeds drop files under stable
    prefixes, so the same prefix sits at the tail every day. Daily-YouTube-*
    went unfiled from 2026-07-31 behind Daily-Chrome-* and AI-Chat-Learnings-*.
    """
    vault.raw("AI-Chat-Learnings.md")   # sorts first alphabetically, newest
    vault.raw("Daily-YouTube.md")       # sorts last alphabetically, oldest
    old = time.time() - 3 * 86400
    os.utime(vault.root / "raw" / "Daily-YouTube.md", (old, old))

    assert wt.list_raw_files(vault.path)["files"] == [
        "Daily-YouTube.md", "AI-Chat-Learnings.md"]


def test_list_raw_files_ties_break_on_name(vault):
    # Same mtime must still give a deterministic order.
    vault.raw("b.md")
    vault.raw("a.md")
    same = time.time() - 60
    for name in ("a.md", "b.md"):
        os.utime(vault.root / "raw" / name, (same, same))

    assert wt.list_raw_files(vault.path)["files"] == ["a.md", "b.md"]


def test_list_raw_files_dates_a_duplicate_name_by_its_older_copy(vault):
    # One basename is one queue entry (that's the .ingested.json identity), and
    # how long it has waited is measured from the older copy.
    vault.raw("recent.md")
    vault.raw("dupe.md")
    vault.raw("dupe.md", subdir="daily-notes")
    old = time.time() - 5 * 86400
    os.utime(vault.root / "raw" / "dupe.md", (old, old))

    assert wt.list_raw_files(vault.path)["files"] == ["dupe.md", "recent.md"]


def test_list_unsorted_raw_files_top_level_only(vault):
    vault.raw("top.txt")
    vault.raw("nested.txt", subdir="daily-notes")
    assert wt.list_unsorted_raw_files(vault.path)["files"] == ["top.txt"]


def _png(vault, name="shot.png", subdir=""):
    """A file with a NUL byte in its header, like any real binary."""
    d = vault.root / "raw" / subdir if subdir else vault.root / "raw"
    d.mkdir(parents=True, exist_ok=True)
    p = d / name
    p.write_bytes(b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR" + b"\xff" * 400)
    return p


def test_list_raw_files_excludes_binaries(vault):
    # A binary decodes to replacement characters rather than failing, so the
    # model receives noise plus a filename and writes what the name implies.
    vault.raw("real.txt")
    _png(vault)
    assert wt.list_raw_files(vault.path)["files"] == ["real.txt"]


def test_list_binary_raw_files_reports_what_was_skipped(vault):
    vault.raw("real.txt")
    _png(vault, "a.png")
    _png(vault, "b.pdf", subdir="misc")
    assert wt.list_binary_raw_files(vault.path)["files"] == ["a.png", "b.pdf"]


def test_list_unsorted_raw_files_excludes_binaries(vault):
    # The sorter classifies by reading the file; a PNG makes it guess.
    vault.raw("top.txt")
    _png(vault)
    assert wt.list_unsorted_raw_files(vault.path)["files"] == ["top.txt"]


def test_binary_detection_ignores_extension(vault):
    # Extension allowlists miss mislabelled files; the NUL byte does not.
    (vault.root / "raw" / "sneaky.md").write_bytes(b"# Title\x00\x00binary")
    assert wt.list_raw_files(vault.path)["files"] == []
    assert wt.list_binary_raw_files(vault.path)["files"] == ["sneaky.md"]


# --- ingested ledger -------------------------------------------------------


def test_get_ingested_sources_missing_file(vault):
    assert wt.get_ingested_sources(vault.path) == []


def test_get_ingested_sources_corrupt_json(vault):
    (vault.root / "wiki" / ".ingested.json").write_text("{not json", encoding="utf-8")
    assert wt.get_ingested_sources(vault.path) == []


def test_mark_ingested_dedups(vault):
    wt.mark_ingested(vault.path, "a.txt")
    wt.mark_ingested(vault.path, "a.txt")
    wt.mark_ingested(vault.path, "b.txt")
    stored = json.loads((vault.root / "wiki" / ".ingested.json").read_text())
    assert stored == ["a.txt", "b.txt"]


# --- write_wiki_page -------------------------------------------------------

FRONTMATTER = (
    "---\nlens: true\ndescription: writing standards\n"
    "max_em_dashes_per_sentence: 1\n---"
)
LENS_PAGE = f"{FRONTMATTER}\n\n# AI Slop\n\nOriginal body.\n"


def test_write_wiki_page_creates_a_new_page(vault):
    assert wt.write_wiki_page(vault.path, "colima", "# Colima\n") == {"written": "colima.md"}
    assert (vault.root / "wiki" / "colima.md").read_text() == "# Colima\n"


def test_write_wiki_page_overwrites_a_plain_page(vault):
    vault.page("colima", "# Colima\n\nOld.\n")
    wt.write_wiki_page(vault.path, "colima", "# Colima\n\nNew.\n")
    assert (vault.root / "wiki" / "colima.md").read_text() == "# Colima\n\nNew.\n"


def test_write_wiki_page_preserves_frontmatter_the_model_dropped(vault):
    """The regression that cost ai-slop.md its lens marker: the model rewrites
    the body from RULES.md's template and omits the YAML block entirely."""
    vault.page("ai-slop", LENS_PAGE)
    wt.write_wiki_page(vault.path, "ai-slop", "# AI Slop\n\nRewritten body.\n")

    result = (vault.root / "wiki" / "ai-slop.md").read_text()
    assert result.startswith(FRONTMATTER)
    assert "lens: true" in result
    assert "max_em_dashes_per_sentence: 1" in result
    assert "Rewritten body." in result
    assert "Original body." not in result


def test_write_wiki_page_respects_frontmatter_the_caller_supplies(vault):
    """A caller that sends its own block is making a deliberate change — the
    old one must not be prepended on top of it."""
    vault.page("ai-slop", LENS_PAGE)
    new = "---\nlens: true\ndescription: revised\n---\n\n# AI Slop\n\nBody.\n"
    wt.write_wiki_page(vault.path, "ai-slop", new)

    assert (vault.root / "wiki" / "ai-slop.md").read_text() == new


def test_write_wiki_page_leaves_pages_without_frontmatter_alone(vault):
    """The no-op path for the ~99% of pages that never had a block."""
    vault.page("colima", "# Colima\n\nOld.\n")
    wt.write_wiki_page(vault.path, "colima", "# Colima\n\nNew.\n")
    assert not (vault.root / "wiki" / "colima.md").read_text().startswith("---")


def test_write_wiki_page_does_not_mistake_a_body_separator_for_frontmatter(vault):
    """This vault's page format puts '---' between the header block and the
    body. Only a block at position 0 counts."""
    vault.page("colima", "# Colima\n\n**Summary**: s\n\n---\n\nBody.\n")
    wt.write_wiki_page(vault.path, "colima", "# Colima\n\nNew.\n")
    assert (vault.root / "wiki" / "colima.md").read_text() == "# Colima\n\nNew.\n"


def test_write_wiki_page_rejects_traversal(vault):
    result = wt.write_wiki_page(vault.path, "../../escape", "x")
    assert "error" in result
    assert not (vault.root.parent / "escape.md").exists()
