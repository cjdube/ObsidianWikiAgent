"""Tests for the pure file-I/O and parsing core in agent/wiki_tools.py."""

import json

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


def test_update_index_appends_missing_pages_under_unfiled(vault):
    vault.page("linked", "# Linked\n\n**Summary**: a linked page.\n")
    vault.page("dropped", "# Dropped\n\n**Summary**: model forgot me.\n")

    result = wt.update_index(vault.path, "# Index\n\n## Topics\n\n- [[linked]]\n")

    index = (vault.root / "wiki" / "index.md").read_text()
    assert wt.UNFILED_HEADING in index
    assert "[[dropped]]" in index
    # The already-linked page is not duplicated into Unfiled.
    unfiled_section = index.split(wt.UNFILED_HEADING, 1)[1]
    assert "[[linked]]" not in unfiled_section
    assert result["unfiled"] == 1


def test_update_index_no_unfiled_when_all_linked(vault):
    vault.page("a", "# A\n\n**Summary**: s.\n")
    result = wt.update_index(vault.path, "## Topics\n\n- [[a]]\n")
    index = (vault.root / "wiki" / "index.md").read_text()
    assert wt.UNFILED_HEADING not in index
    assert result["unfiled"] == 0


def test_update_index_delinks_broken_links(vault):
    vault.page("real", "# Real\n\n**Summary**: s.\n")
    # The model authored one valid link and one garbled/dead link.
    result = wt.update_index(
        vault.path,
        "# Index\n\n## Topics\n\n- [[real]] the good one\n"
        "- [[ai-chat--2026-07-15]] a garbled slug\n",
    )
    index = (vault.root / "wiki" / "index.md").read_text()
    assert "[[real]]" in index                      # valid link kept
    assert "[[ai-chat--2026-07-15]]" not in index    # dead link removed
    assert "a garbled slug" in index                 # description preserved
    assert "ai-chat--2026-07-15" in index            # de-linked to plain text
    assert result["delinked"] == 1


def test_update_index_delink_keeps_alias_text(vault):
    result = wt.update_index(vault.path, "## T\n\n- [[gone|Old Name]] desc\n")
    index = (vault.root / "wiki" / "index.md").read_text()
    assert "[[gone|Old Name]]" not in index
    assert "Old Name" in index      # the alias the reader saw is kept as text
    assert result["delinked"] == 1


def test_update_index_keeps_valid_links_and_unfiled_intact(vault):
    # De-linking must not disturb valid links or the Unfiled guarantee.
    vault.page("linked", "# Linked\n\n**Summary**: s.\n")
    vault.page("dropped", "# Dropped\n\n**Summary**: s.\n")
    result = wt.update_index(
        vault.path, "## T\n\n- [[linked]]\n- [[ghost]] not a page\n"
    )
    index = (vault.root / "wiki" / "index.md").read_text()
    assert "[[linked]]" in index
    assert "[[ghost]]" not in index
    assert "[[dropped]]" in index          # still appended under Unfiled
    assert result["unfiled"] == 1
    assert result["delinked"] == 1


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


def test_list_unsorted_raw_files_top_level_only(vault):
    vault.raw("top.txt")
    vault.raw("nested.txt", subdir="daily-notes")
    assert wt.list_unsorted_raw_files(vault.path)["files"] == ["top.txt"]


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
