"""Tests for the pure file-I/O and parsing core in agent/wiki_tools.py."""

import json
import os
import time
from datetime import date
from pathlib import Path

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
        # A `[[` in a code span must not open a match that runs through the
        # real link after it, nor may an unclosed [[ cross a line break.
        ("Unclosed `[[` brackets break the [[owa]] linter", {"owa"}),
        ("stray [[ opener\nnext line has [[foo]]", {"foo"}),
    ],
)
def test_linked_page_names(content, expected):
    assert wt.linked_page_names(content) == expected


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


# --- list_index_sections ---------------------------------------------------


def test_list_index_sections_returns_headings_only(vault):
    vault.index(
        "# Index\n\nSome intro prose.\n\n"
        "## AI & Agent Development\n\n- [[ollama]] Local models.\n"
        "- [[colima]] Containers.\n\n"
        "### A sub-heading that is not a section\n\n"
        "## Product\n\n- [[prd]] How we write them.\n"
    )
    assert wt.list_index_sections(vault.path) == {
        "sections": ["AI & Agent Development", "Product"]
    }


def test_list_index_sections_omits_unfiled(vault):
    """Unfiled is computed by _normalize_index and update_index rejects filing
    into it, so offering it as a destination only buys a round-trip."""
    vault.index("# Index\n\n## Tools\n\n- [[colima]] x\n\n## Unfiled\n\n- [[stray]] y\n")
    assert wt.list_index_sections(vault.path) == {"sections": ["Tools"]}


def test_list_index_sections_on_a_missing_index(vault):
    assert wt.list_index_sections(vault.path) == {"sections": []}


def test_list_index_sections_agrees_with_section_bounds(vault):
    """The two must use one rule for what counts as a section — the model picks
    a name from this list and update_index has to find it."""
    vault.index("# Index\n\n## Tools\n\n- [[colima]] x\n\n##  Spaced  \n\n- [[y]] z\n")
    for name in wt.list_index_sections(vault.path)["sections"]:
        lines = wt.read_index(vault.path)["content"].splitlines()
        assert wt._section_bounds(lines, name) is not None


def test_no_ingest_stage_offers_the_whole_index():
    """The token saving only lands if read_index is un-advertised."""
    for schemas in (wt.PLAN_TOOL_SCHEMAS, wt.CREATE_PAGE_TOOL_SCHEMAS,
                    wt.UPDATE_PAGE_TOOL_SCHEMAS, wt.LOG_TOOL_SCHEMAS):
        assert "read_index" not in [t["function"]["name"] for t in schemas]
    assert "list_index_sections" in [
        t["function"]["name"] for t in wt.PLAN_TOOL_SCHEMAS
    ]
    # The read side still gets the full table of contents.
    assert "read_index" in [t["function"]["name"] for t in wt.QUERY_TOOL_SCHEMAS]


def test_planning_stage_cannot_read_or_write_pages():
    """Stage 1 exists to decide *which* pages to touch. A page read here would
    pull back the per-page bulk the split removes, and a write would make the
    plan-only path write to the vault."""
    names = [t["function"]["name"] for t in wt.PLAN_TOOL_SCHEMAS]
    assert "read_wiki_page" not in names
    assert "write_wiki_page" not in names
    assert "update_index" not in names
    assert "append_log" not in names


def test_execute_stage_does_not_relist_every_page():
    """418 page names is the largest item in the old context. Re-listing it once
    per planned page would be worse than the problem being fixed."""
    for schemas in (wt.CREATE_PAGE_TOOL_SCHEMAS, wt.UPDATE_PAGE_TOOL_SCHEMAS):
        names = [t["function"]["name"] for t in schemas]
        assert "list_wiki_pages" not in names
        assert "submit_plan" not in names
        # append_log is stage 3's alone — it is the one non-idempotent tool, and
        # a per-page stage would append once per page instead of once per source.
        assert "append_log" not in names


def test_a_page_that_exists_is_offered_no_way_to_be_overwritten():
    """The whole point of the split. write_wiki_page replaces the file, which is
    how updates lost lines and baked in escaping; an update never sees it, and a
    create never sees the edit tool that would only fail on a missing page."""
    update = [t["function"]["name"] for t in wt.UPDATE_PAGE_TOOL_SCHEMAS]
    create = [t["function"]["name"] for t in wt.CREATE_PAGE_TOOL_SCHEMAS]

    assert "write_wiki_page" not in update
    assert "edit_wiki_page" in update
    assert "edit_wiki_page" not in create
    assert "write_wiki_page" in create
    # Both still read the source, read the page, and file it.
    for names in (update, create):
        assert {"read_raw_file", "read_wiki_page", "update_index"} <= set(names)


def test_query_dispatch_covers_exactly_the_query_schemas(vault):
    """The pairing wiki_query.py and wiki_lint.py both rely on."""
    dispatch = wt.query_dispatch(vault.path)
    assert set(dispatch) == {t["function"]["name"] for t in wt.QUERY_TOOL_SCHEMAS}


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


def test_update_index_keeps_hand_written_content_in_a_section(vault):
    """A section is not just its entry lines — a human edits this file too.

    Rebuilding the body from `- [[page]]` lines alone deleted prose and
    sub-headings on the next ingest that touched the section, silently, with the
    call still reporting success.
    """
    vault.page("foo", "# Foo\n\n**Summary**: a summary.\n")
    vault.index(
        "# Index\n\n## Tools\n\n"
        "Hand-written note about this section.\n\n"
        "### Sub-heading\n\n"
        "- [[foo]] a summary.\n"
    )

    wt.update_index(vault.path, "foo", "Tools")

    index = (vault.root / "wiki" / "index.md").read_text()
    assert "Hand-written note about this section." in index
    assert "### Sub-heading" in index
    assert index.count("[[foo]]") == 1


def test_update_index_sorts_into_the_trailing_run_not_across_sub_headings(vault):
    # Entries grouped under sub-headings keep their groups: the new entry joins
    # the last run rather than being flattened into one alphabetical list.
    for name in ("alpha", "zulu", "mike"):
        vault.page(name, f"# {name}\n\n**Summary**: s.\n")
    vault.index(
        "## Tools\n\n### CLI\n\n- [[zulu]] s.\n\n### GUI\n\n- [[alpha]] s.\n"
    )

    wt.update_index(vault.path, "mike", "Tools")

    index = (vault.root / "wiki" / "index.md").read_text()
    assert "### CLI" in index and "### GUI" in index
    # zulu stays put under CLI; mike sorts into the GUI run alongside alpha.
    cli, gui = index.split("### GUI", 1)
    assert "[[zulu]]" in cli and "[[mike]]" not in cli
    assert [wt._entry_name(ln) for ln in gui.splitlines() if wt._entry_name(ln)] \
        == ["alpha", "mike"]


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


def test_move_raw_file_refuses_to_overwrite_a_name_already_there(vault):
    """Path.rename replaces the destination silently on POSIX, and raw/ is the
    one tree whose contents are never supposed to change."""
    vault.raw("notes.md", "the copy already filed", subdir="daily-notes")
    vault.raw("notes.md", "the copy just dropped in")

    result = wt.move_raw_file(vault.path, "notes.md", "daily-notes")

    assert "error" in result
    assert "already exists" in result["error"]
    # Both survive: neither the filed copy nor the dropped one is lost.
    assert (vault.root / "raw" / "daily-notes" / "notes.md").read_text() == "the copy already filed"
    assert (vault.root / "raw" / "notes.md").read_text() == "the copy just dropped in"


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


def test_scan_raw_classifies_every_way_in_one_walk(vault):
    """The three listings are three views of one walk, not three walks."""
    vault.raw("top.txt")
    vault.raw("nested.txt", subdir="daily-notes")
    _png(vault, "a.png")
    _png(vault, "b.pdf", subdir="misc")

    scan = wt.scan_raw(vault.path)

    assert set(scan.text) == {"top.txt", "nested.txt"}
    assert scan.binary == ["a.png", "b.pdf"]
    assert scan.unsorted == ["top.txt"]
    # And the views agree with taking each listing on its own.
    assert wt.list_raw_files(vault.path, scan) == wt.list_raw_files(vault.path)
    assert wt.list_binary_raw_files(vault.path, scan) == wt.list_binary_raw_files(vault.path)
    assert wt.list_unsorted_raw_files(vault.path, scan) == wt.list_unsorted_raw_files(vault.path)


def test_scan_raw_probes_each_file_once(vault):
    vault.raw("a.txt")
    vault.raw("b.txt", subdir="daily-notes")
    _png(vault, "c.png")
    probed = []
    original = wt._is_text_source
    wt._is_text_source = lambda p: probed.append(p.name) or original(p)
    try:
        wt.scan_raw(vault.path)
    finally:
        wt._is_text_source = original
    assert sorted(probed) == ["a.txt", "b.txt", "c.png"]


def test_binary_detection_ignores_extension(vault):
    # Extension allowlists miss mislabelled files; the NUL byte does not.
    (vault.root / "raw" / "sneaky.md").write_bytes(b"# Title\x00\x00binary")
    assert wt.list_raw_files(vault.path)["files"] == []
    assert wt.list_binary_raw_files(vault.path)["files"] == ["sneaky.md"]


# --- ingested ledger -------------------------------------------------------


def test_get_ingested_sources_missing_file(vault):
    assert wt.get_ingested_sources(vault.path) == []


def test_get_ingested_sources_refuses_corrupt_json(vault):
    """[] here would mean "nothing has ever been ingested" and re-ingest the
    whole of raw/. A truncated ledger is exactly what a crash mid-write leaves."""
    (vault.root / "wiki" / ".ingested.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(RuntimeError, match="unreadable"):
        wt.get_ingested_sources(vault.path)


def test_get_ingested_sources_refuses_a_non_list(vault):
    (vault.root / "wiki" / ".ingested.json").write_text('{"a": 1}', encoding="utf-8")
    with pytest.raises(RuntimeError, match="not a JSON list"):
        wt.get_ingested_sources(vault.path)


def test_mark_ingested_dedups(vault):
    wt.mark_ingested(vault.path, "a.txt")
    wt.mark_ingested(vault.path, "a.txt")
    wt.mark_ingested(vault.path, "b.txt")
    stored = json.loads((vault.root / "wiki" / ".ingested.json").read_text())
    assert stored == ["a.txt", "b.txt"]


def test_mark_ingested_leaves_no_temp_file_behind(vault):
    wt.mark_ingested(vault.path, "a.txt")
    assert [p.name for p in (vault.root / "wiki").iterdir()] == [".ingested.json"]


def test_mark_ingested_never_leaves_a_half_written_ledger(vault, monkeypatch):
    """The ledger must be the old list or the new one, never a truncated file
    that parses as nothing. write_text truncates first, so the rename is what
    buys this."""
    wt.mark_ingested(vault.path, "a.txt")

    real_write = Path.write_text

    def die_mid_write(self, *args, **kwargs):
        real_write(self, *args, **kwargs)
        raise RuntimeError("watchdog fired")

    monkeypatch.setattr(Path, "write_text", die_mid_write)
    with pytest.raises(RuntimeError, match="watchdog"):
        wt.mark_ingested(vault.path, "b.txt")

    monkeypatch.undo()
    assert wt.get_ingested_sources(vault.path) == ["a.txt"]


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


def test_write_wiki_page_keeps_frontmatter_when_the_body_opens_with_a_rule(vault):
    """'---' at the top of a body is a horizontal rule, not the caller
    supplying frontmatter. Reading it as one dropped the very block this guard
    exists to preserve — the `lens: true` marker nothing else notices is gone."""
    vault.page("ai-slop", LENS_PAGE)
    wt.write_wiki_page(vault.path, "ai-slop", "---\n\n# AI Slop\n\nNew body.\n")
    written = (vault.root / "wiki" / "ai-slop.md").read_text()
    assert written.startswith(FRONTMATTER)
    assert "lens: true" in written
    assert "New body." in written


# --- edit_wiki_page ---------------------------------------------------------

PAGE = """\
# Ollama

**Summary**: A tool for running local models.
**Sources**: AI-Chat-2026-07-01.md
**Last updated**: 2026-07-01

---

## Context Management

- num_ctx is pinned (source: AI-Chat-2026-07-01.md).

### Overflow

- Ollama drops the oldest messages.

## Related pages

- [[gemma-4]]
"""


def _edit(vault, **kw):
    kw.setdefault("source", "Chat-2026-08-21.md")
    kw.setdefault("name", "ollama")
    return wt.edit_wiki_page(vault.path, kw.pop("source"), **kw)


def test_edit_appends_to_an_existing_section(vault):
    vault.page("ollama", PAGE)
    result = _edit(vault, section="Context Management", content="- New fact.")

    assert result == {"edited": "ollama.md", "section": "Context Management"}
    text = (vault.root / "wiki" / "ollama.md").read_text()
    # Lands after the subsection that belongs to the parent, not in front of it.
    assert text.index("- New fact.") > text.index("Ollama drops the oldest")
    assert text.index("- New fact.") < text.index("## Related pages")


def test_edit_leaves_every_other_line_byte_identical(vault):
    """The whole reason this tool exists. Text the model never re-emits is text
    it cannot lose or bake JSON escaping into."""
    vault.page("ollama", PAGE)
    _edit(vault, section="Context Management", content="- New fact.")

    before = [l for l in PAGE.split("\n") if l.strip()]
    after = (vault.root / "wiki" / "ollama.md").read_text()
    untouched = [
        l for l in before
        if not l.startswith(("**Sources**:", "**Last updated**:"))
    ]
    for line in untouched:
        assert line in after, f"lost: {line!r}"


def test_edit_creates_a_missing_section_before_related_pages(vault):
    """'## Related pages' is the page's last word per RULES.md."""
    vault.page("ollama", PAGE)
    _edit(vault, section="Truncation", content="- done_reason is 'length'.")

    text = (vault.root / "wiki" / "ollama.md").read_text()
    assert "## Truncation" in text
    assert text.index("## Truncation") < text.index("## Related pages")


def test_edit_separates_a_new_section_with_exactly_one_blank_line(vault):
    """Obsidian renders it either way, but a page that grows a blank line per
    ingest drifts away from the format every other page keeps."""
    vault.page("ollama", PAGE)
    _edit(vault, section="Truncation", content="- done_reason is 'length'.")

    text = (vault.root / "wiki" / "ollama.md").read_text()
    assert "\n\n\n" not in text
    assert "\n\n## Truncation\n\n- done_reason" in text
    assert "\n\n## Related pages\n" in text


def test_edit_creates_a_missing_section_at_the_end_without_related_pages(vault):
    vault.page("plain", "# Plain\n\n**Sources**: a.md\n**Last updated**: 2026-01-01\n")
    _edit(vault, name="plain", section="Notes", content="- A note.")

    text = (vault.root / "wiki" / "plain.md").read_text()
    assert text.rstrip().endswith("- A note.")


def test_edit_adds_a_related_link_through_the_same_mechanism(vault):
    """A back-link is the commonest update in a plan, and it needs no special
    case — 'Related pages' is just a section."""
    vault.page("ollama", PAGE)
    _edit(vault, section="Related pages", content="- [[num-predict]]")

    text = (vault.root / "wiki" / "ollama.md").read_text()
    assert "- [[gemma-4]]" in text
    assert "- [[num-predict]]" in text


def test_edit_records_the_source_and_stamps_the_date(vault):
    vault.page("ollama", PAGE)
    _edit(vault, section="Context Management", content="- New fact.")

    text = (vault.root / "wiki" / "ollama.md").read_text()
    assert "**Sources**: AI-Chat-2026-07-01.md, Chat-2026-08-21.md" in text
    assert f"**Last updated**: {date.today().isoformat()}" in text


def test_edit_does_not_list_the_same_source_twice(vault):
    vault.page("ollama", PAGE)
    _edit(vault, source="AI-Chat-2026-07-01.md",
          section="Context Management", content="- New fact.")

    # Counted on the Sources line alone — the body legitimately cites the same
    # file inline, and that is not a duplicate.
    text = (vault.root / "wiki" / "ollama.md").read_text()
    sources = next(l for l in text.split("\n") if l.startswith("**Sources**:"))
    assert sources == "**Sources**: AI-Chat-2026-07-01.md"


def test_edit_keeps_a_source_filename_that_contains_a_comma(vault):
    """This vault really has one. Splitting the Sources list on ',' to rejoin it
    turned that single citation into two invented ones, which wiki_lint then
    reported as sources that are not files in raw/."""
    awkward = "if-you’re-still-hitting-the-5-hour-wall,-you’re-doing-it-wrong.md"
    vault.page("ollama", PAGE.replace(
        "**Sources**: AI-Chat-2026-07-01.md",
        f"**Sources**: AI-Chat-2026-07-01.md, {awkward}",
    ))

    _edit(vault, section="Context Management", content="- New fact.")

    sources = next(
        l for l in (vault.root / "wiki" / "ollama.md").read_text().split("\n")
        if l.startswith("**Sources**:")
    )
    assert awkward in sources
    assert sources.endswith("Chat-2026-08-21.md")


def test_edit_does_not_reflow_the_sources_line_it_appends_to(vault):
    """Whatever spacing the line already had is the page's, not this tool's."""
    vault.page("ollama", PAGE.replace(
        "**Sources**: AI-Chat-2026-07-01.md",
        "**Sources**: a.md,b.md, c.md",
    ))

    _edit(vault, section="Context Management", content="- New fact.")

    text = (vault.root / "wiki" / "ollama.md").read_text()
    assert "**Sources**: a.md,b.md, c.md, Chat-2026-08-21.md" in text


def test_edit_replaces_the_summary_only_when_given_one(vault):
    vault.page("ollama", PAGE)
    _edit(vault, section="Context Management", content="- A.")
    assert "**Summary**: A tool for running local models." in (
        vault.root / "wiki" / "ollama.md").read_text()

    _edit(vault, section="Context Management", content="- B.", summary="Revised.")
    assert "**Summary**: Revised." in (vault.root / "wiki" / "ollama.md").read_text()


def test_edit_is_idempotent_on_content_already_present(vault):
    """Replacing a file is naturally idempotent; appending is not. A retried
    conversation must not leave the same paragraph on the page twice."""
    vault.page("ollama", PAGE)
    _edit(vault, section="Context Management", content="- New fact.")
    first = (vault.root / "wiki" / "ollama.md").read_text()

    result = _edit(vault, section="Context Management", content="- New fact.")

    assert "unchanged" in result
    assert (vault.root / "wiki" / "ollama.md").read_text() == first


def test_edit_preserves_frontmatter(vault):
    vault.page("ai-slop", FRONTMATTER + "\n\n# AI Slop\n\n**Sources**: a.md\n"
                                        "**Last updated**: 2026-01-01\n")
    _edit(vault, name="ai-slop", section="Notes", content="- A note.")

    text = (vault.root / "wiki" / "ai-slop.md").read_text()
    assert text.startswith(FRONTMATTER)
    assert "lens: true" in text


def test_edit_decodes_content_the_model_double_encoded(vault):
    """Same guard write_wiki_page has. It is the escaping that leaked into 25
    pages, and it must not survive on the path that replaced it."""
    vault.page("ollama", PAGE)
    _edit(vault, section="Notes", content='- A \\"quoted\\" line.\\n- Second.')

    text = (vault.root / "wiki" / "ollama.md").read_text()
    assert '- A "quoted" line.' in text
    assert "\\n" not in text


def test_edit_refuses_page_content_passed_as_a_section_name(vault):
    """Seen once in 13 pages on 2026-08-21: the model put a whole bullet in
    'section', and the page grew a 319-character heading with the real material
    filed under it. Markdown accepts that, so only this can refuse it."""
    vault.page("ollama", PAGE)
    bullet = "- **LocalLLMAgent**: " + "Continued progress with servers. " * 8

    result = _edit(vault, section=bullet, content="- A fact.")

    assert "not a section heading" in result["error"]
    assert "## - **LocalLLMAgent**" not in (
        vault.root / "wiki" / "ollama.md").read_text()


def test_edit_still_appends_to_a_long_heading_the_page_already_has(vault):
    """The shape test guards heading *creation*. A heading already on the page
    is the page's own, whatever it looks like, and must stay reachable."""
    long_heading = "A Heading That Someone Wrote By Hand And Made Much Longer Than Sixty Characters"
    vault.page("ollama", PAGE.replace("## Context Management", f"## {long_heading}"))

    result = _edit(vault, section=long_heading, content="- A fact.")

    assert "edited" in result
    text = (vault.root / "wiki" / "ollama.md").read_text()
    assert text.count(f"## {long_heading}") == 1


def test_edit_refuses_a_page_that_does_not_exist(vault):
    result = _edit(vault, name="ghost", section="Notes", content="- x")
    assert "does not exist" in result["error"]
    assert not (vault.root / "wiki" / "ghost.md").exists()


def test_edit_refuses_empty_content(vault):
    vault.page("ollama", PAGE)
    assert "empty" in _edit(vault, section="Notes", content="   ")["error"]


@pytest.mark.parametrize("name,tool", [("index", "update_index"), ("log", "append_log")])
def test_edit_refuses_the_files_python_owns(vault, name, tool):
    """Same two names write_wiki_page refuses, for the same reasons — the index
    rebuilds itself, the log does not come back."""
    vault.page(name, "# X\n")
    result = _edit(vault, name=name, section="Notes", content="- x")
    assert tool in result["error"]


def test_edit_refuses_a_nested_name(vault):
    result = _edit(vault, name="topics/foo", section="Notes", content="- x")
    assert "nest" in result["error"]


def test_write_wiki_page_rejects_a_nested_name(vault):
    """It used to succeed, creating wiki/topics/ — and then nothing could see
    the page: list_wiki_pages uses iterdir, so it was absent from the index and
    from every lint check, while update_index refused to file it."""
    result = wt.write_wiki_page(vault.path, "topics/foo", "# Foo\n")
    assert "nest" in result["error"]
    assert not (vault.root / "wiki" / "topics").exists()


def test_read_wiki_page_rejects_a_nested_name(vault):
    assert "nest" in wt.read_wiki_page(vault.path, "topics/foo")["error"]


def test_safe_page_path_allows_a_name_that_resolves_back_to_flat(vault):
    """'sub/../foo' names wiki/foo.md, which is where pages go."""
    assert wt._safe_page_path(vault.path, "sub/../foo").name == "foo.md"


def test_write_wiki_page_decodes_an_escaped_body(vault):
    """A body the model json-encoded one time too many arrives as one line of
    literal \\n. Written through, it costs the page its title, its header
    fields and its links (observed 2026-08-16 on ollama.md)."""
    wt.write_wiki_page(
        vault.path,
        "ollama",
        "# Ollama\\n\\n**Summary**: Runs models \\\"locally\\\".\\n\\nSee [[agent-tools]].\\n",
    )
    written = (vault.root / "wiki" / "ollama.md").read_text()
    assert written == '# Ollama\n\n**Summary**: Runs models "locally".\n\nSee [[agent-tools]].\n'


def test_write_wiki_page_decodes_a_doubly_escaped_body(vault):
    wt.write_wiki_page(vault.path, "ollama", "# Ollama\\\\n\\\\nBody.\\\\n")
    assert (vault.root / "wiki" / "ollama.md").read_text() == "# Ollama\n\nBody.\n"


def test_write_wiki_page_decodes_escaped_frontmatter_before_the_carry_over(vault):
    """An escaped body opens with a literal '---', so the frontmatter check
    would read it as supplying its own block and skip the carry-over."""
    vault.page("owa", "---\nproject: true\n---\n\n# OWA\n")
    wt.write_wiki_page(vault.path, "owa", "# Obsidian Wiki Agent\\n\\nBody.\\n")
    written = (vault.root / "wiki" / "owa.md").read_text()
    assert written.startswith("---\nproject: true\n---\n")
    assert "# Obsidian Wiki Agent" in written


def test_write_wiki_page_leaves_a_code_block_backslash_n_alone(vault):
    """A page may legitimately show \\n in a snippet — just never more often
    than it starts a new line."""
    body = '# Regex\n\n**Summary**: s\n\nUse `re.split("\\n", text)` here.\n'
    wt.write_wiki_page(vault.path, "regex", body)
    assert (vault.root / "wiki" / "regex.md").read_text() == body


def test_write_wiki_page_decodes_quotes_when_the_newlines_are_fine(vault):
    """The gap the ratio test never covered. A body whose newlines survived
    still carries \\" through every quotation, passes the ratio check untouched,
    and lands on disk looking almost right — which is how 41 lines across 23
    pages went unnoticed until they were swept by hand on 2026-08-21."""
    wt.write_wiki_page(
        vault.path,
        "vibe-coding",
        '# Vibe Coding\n\n**Summary**: A \\"vibe\\" shift.\n\nSee the \\"end\\" of it.\n',
    )
    written = (vault.root / "wiki" / "vibe-coding.md").read_text()
    assert written == (
        '# Vibe Coding\n\n**Summary**: A "vibe" shift.\n\nSee the "end" of it.\n'
    )


def test_write_wiki_page_decodes_unicode_escapes(vault):
    wt.write_wiki_page(
        vault.path, "fpl", "# FPL\n\nTypically 5\\u201310 hours\\u2014weekly.\n"
    )
    assert (vault.root / "wiki" / "fpl.md").read_text() == (
        "# FPL\n\nTypically 5–10 hours—weekly.\n"
    )


def test_write_wiki_page_keeps_escaped_quotes_inside_a_json_fence(vault):
    """Inside a JSON string, \\" is how a quote is spelled. Decoding it breaks
    the example — claude-code-hooks.md has two such lines."""
    body = (
        '# Hooks\n\n**Summary**: s\n\n```json\n'
        '{"command": "npx prettier --write \\"$CLAUDE_FILE_PATH\\""}\n'
        '```\n'
    )
    wt.write_wiki_page(vault.path, "hooks", body)
    assert (vault.root / "wiki" / "hooks.md").read_text() == body


def test_write_wiki_page_decodes_escaped_quotes_inside_a_bash_fence(vault):
    """Only json is exempt. The shell needs no backslash before a quote, so one
    there is the same damage as anywhere else."""
    wt.write_wiki_page(
        vault.path,
        "tools",
        '# Tools\n\n```bash\npython -m weather --location \\"Boston,MA,US\\"\n```\n',
    )
    assert 'python -m weather --location "Boston,MA,US"' in (
        vault.root / "wiki" / "tools.md").read_text()


def test_write_wiki_page_leaves_a_line_about_escaping_alone(vault):
    """A sentence about escape sequences is showing one on purpose. Two pages
    in this vault narrate an earlier repair of exactly this damage."""
    body = '# OWA\n\nFixed JSON escape sequences (`\\u2019`) across the vault.\n'
    wt.write_wiki_page(vault.path, "owa", body)
    assert (vault.root / "wiki" / "owa.md").read_text() == body


def test_edit_wiki_page_gets_the_same_guard(vault):
    """Both write paths enter the vault through _decode_if_escaped, so neither
    can be the one that lets escaping back in."""
    vault.page("ollama", PAGE)
    _edit(vault, section="Context Management", content='- A \\"quoted\\" fact.')

    assert '- A "quoted" fact.' in (vault.root / "wiki" / "ollama.md").read_text()


def test_write_wiki_page_rejects_traversal(vault):
    result = wt.write_wiki_page(vault.path, "../../escape", "x")
    assert "error" in result
    assert not (vault.root.parent / "escape.md").exists()


# --- reserved names --------------------------------------------------------
#
# index.md and log.md live inside wiki/, so _safe_page_path admits them and
# list_wiki_pages only hides them from view. write_wiki_page opens "w", so
# either name reaching it discards a file Python is supposed to own.


@pytest.mark.parametrize("name", ["index", "index.md", "log", "log.md"])
def test_write_wiki_page_refuses_reserved_names(vault, name):
    original = "# Index\n\n## Tools\n\n- [[colima]] A container runtime.\n"
    vault.index(original)
    (vault.root / "wiki" / "log.md").write_text("- 2026-08-01: first entry\n")
    before = {p.name: p.read_text() for p in (vault.root / "wiki").iterdir()}

    result = wt.write_wiki_page(vault.path, name, "# Clobbered\n")

    assert "error" in result
    assert "write_wiki_page" in result["error"]
    after = {p.name: p.read_text() for p in (vault.root / "wiki").iterdir()}
    assert after == before


def test_write_wiki_page_reserved_error_names_the_right_tool(vault):
    """The model is mid-workflow — a bare refusal just gets retried."""
    assert "update_index" in wt.write_wiki_page(vault.path, "index", "x")["error"]
    assert "append_log" in wt.write_wiki_page(vault.path, "log", "x")["error"]


def test_write_wiki_page_still_allows_ordinary_names(vault):
    """The guard is two filenames, not a prefix match — 'indexing' is a page."""
    assert "written" in wt.write_wiki_page(vault.path, "indexing", "# Indexing\n")
    assert "written" in wt.write_wiki_page(vault.path, "logging", "# Logging\n")
