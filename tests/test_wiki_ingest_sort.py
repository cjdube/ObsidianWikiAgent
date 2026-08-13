"""Tests for the pre-ingest sort step in wiki_ingest.py.

This is the only place a model's answer decides a filesystem mutation: the
reply names a folder and the file is moved there. complete_text is mocked
throughout — no Ollama, no network.

The vault fixture's RULES.md declares two folders, 'daily-notes' and 'misc'
(see tests/conftest.py), so 'misc' is available as the fallback except where a
test replaces the rules to remove it.
"""

import pytest

from agent import budget
from wiki_ingest import _classify_raw_file, sort_raw_files


class _Recorder:
    """Captures what the sort step logged, per level."""

    def __init__(self):
        self.records = {"info": [], "warning": [], "error": []}

    def info(self, msg):
        self.records["info"].append(msg)

    def warning(self, msg):
        self.records["warning"].append(msg)

    def error(self, msg):
        self.records["error"].append(msg)

    exception = error

    def said(self, level, needle):
        return any(needle in m for m in self.records[level])


FOLDERS = [
    {"name": "daily-notes", "description": "Notes captured day to day."},
    {"name": "misc", "description": "Anything that doesn't fit elsewhere."},
]


def _reply(monkeypatch, text):
    """Make the classifier's model return exactly `text`."""
    import wiki_ingest

    monkeypatch.setattr(wiki_ingest, "complete_text", lambda **kw: text)


# --- _classify_raw_file ----------------------------------------------------


def test_classify_exact_match(monkeypatch):
    _reply(monkeypatch, "daily-notes")
    assert _classify_raw_file("n.md", "body", FOLDERS) == "daily-notes"


def test_classify_is_case_insensitive(monkeypatch):
    _reply(monkeypatch, "Daily-Notes")
    assert _classify_raw_file("n.md", "body", FOLDERS) == "daily-notes"


@pytest.mark.parametrize("reply", ["daily-notes.", "daily-notes,", "  daily-notes  "])
def test_classify_tolerates_trailing_punctuation(monkeypatch, reply):
    """Small models routinely add a full stop to a one-word answer."""
    _reply(monkeypatch, reply)
    assert _classify_raw_file("n.md", "body", FOLDERS) == "daily-notes"


def test_classify_falls_back_to_a_substring(monkeypatch):
    """The last resort, for a model that explains itself instead of answering."""
    _reply(monkeypatch, "This looks like it belongs in daily-notes to me.")
    assert _classify_raw_file("n.md", "body", FOLDERS) == "daily-notes"


def test_classify_returns_none_when_nothing_matches(monkeypatch):
    _reply(monkeypatch, "I am not sure about this one.")
    assert _classify_raw_file("n.md", "body", FOLDERS) is None


def test_classify_sends_only_the_head_of_a_large_file(monkeypatch):
    """A source can be tens of KB; the sorter only ever needs the opening."""
    import wiki_ingest

    seen = {}
    monkeypatch.setattr(
        wiki_ingest, "complete_text",
        lambda **kw: seen.update(kw) or "misc",
    )
    _classify_raw_file("n.md", "x" * 50_000, FOLDERS)
    assert len(seen["user_prompt"]) < 3000


# --- sort_raw_files --------------------------------------------------------


def test_sort_moves_a_dropped_file_into_its_folder(vault, monkeypatch):
    vault.raw("monday.md", "Notes from Monday.")
    _reply(monkeypatch, "daily-notes")
    logger = _Recorder()

    sort_raw_files(vault.path, logger)

    assert (vault.root / "raw" / "daily-notes" / "monday.md").is_file()
    assert not (vault.root / "raw" / "monday.md").exists()
    assert logger.said("info", "monday.md")


def test_sort_defaults_to_misc_when_it_cannot_classify(vault, monkeypatch):
    vault.raw("mystery.md", "???")
    _reply(monkeypatch, "no idea")
    logger = _Recorder()

    sort_raw_files(vault.path, logger)

    assert (vault.root / "raw" / "misc" / "mystery.md").is_file()
    assert logger.said("warning", "defaulting to 'misc'")


def test_sort_leaves_the_file_alone_when_no_misc_is_declared(vault, monkeypatch):
    """The fallback is opt-in per vault — without it, an unclassifiable file
    stays put rather than being invented a home."""
    (vault.root / "RULES.md").write_text(
        "# Rules\n\n## Raw folders\n\n- daily-notes Notes captured day to day.\n",
        encoding="utf-8",
    )
    vault.raw("mystery.md", "???")
    _reply(monkeypatch, "no idea")
    logger = _Recorder()

    sort_raw_files(vault.path, logger)

    assert (vault.root / "raw" / "mystery.md").is_file()
    assert logger.said("warning", "leaving it in place")


def test_sort_is_a_noop_without_a_raw_folders_section(vault, monkeypatch):
    (vault.root / "RULES.md").write_text("# Rules\n\nNo folders here.\n", encoding="utf-8")
    vault.raw("monday.md", "Notes.")
    _reply(monkeypatch, "daily-notes")
    logger = _Recorder()

    sort_raw_files(vault.path, logger)

    assert (vault.root / "raw" / "monday.md").is_file()
    assert logger.said("info", "skipping raw sort step")


def test_sort_survives_a_classifier_that_raises(vault, monkeypatch):
    """One bad file must not abandon the rest of the queue."""
    import wiki_ingest

    vault.raw("a-boom.md", "x")
    vault.raw("b-fine.md", "y")

    def flaky(**kw):
        if "a-boom.md" in kw["user_prompt"]:
            raise RuntimeError("model blew up")
        return "daily-notes"

    monkeypatch.setattr(wiki_ingest, "complete_text", flaky)
    logger = _Recorder()

    sort_raw_files(vault.path, logger)

    assert (vault.root / "raw" / "misc" / "a-boom.md").is_file()
    assert (vault.root / "raw" / "daily-notes" / "b-fine.md").is_file()


def test_sort_does_not_demote_a_budget_exception_to_a_bad_guess(vault, monkeypatch):
    """BudgetExceeded means the run is over. Catching it as 'couldn't classify
    this one' would file the file under misc and keep going, which is exactly
    what the budget exists to prevent."""
    import wiki_ingest

    vault.raw("monday.md", "Notes.")

    def exhausted(**kw):
        raise budget.BudgetExceeded("run budget exhausted")

    monkeypatch.setattr(wiki_ingest, "complete_text", exhausted)

    with pytest.raises(budget.BudgetExceeded):
        sort_raw_files(vault.path, _Recorder())

    assert (vault.root / "raw" / "monday.md").is_file()
    assert not (vault.root / "raw" / "misc").exists()


def test_sort_reports_a_collision_instead_of_overwriting(vault, monkeypatch):
    """move_raw_file refuses to replace a name already taken; the sort step
    logs that and leaves both copies alone."""
    vault.raw("notes.md", "already filed", subdir="daily-notes")
    vault.raw("notes.md", "just dropped in")
    _reply(monkeypatch, "daily-notes")
    logger = _Recorder()

    sort_raw_files(vault.path, logger)

    assert (vault.root / "raw" / "daily-notes" / "notes.md").read_text() == "already filed"
    assert (vault.root / "raw" / "notes.md").read_text() == "just dropped in"
    assert logger.said("warning", "already exists")


def test_sort_ignores_files_already_in_subdirectories(vault, monkeypatch):
    """Only the top level is unsorted input; a filed file is not re-sorted."""
    vault.raw("filed.md", "x", subdir="daily-notes")
    _reply(monkeypatch, "misc")
    logger = _Recorder()

    sort_raw_files(vault.path, logger)

    assert (vault.root / "raw" / "daily-notes" / "filed.md").is_file()
    assert logger.said("info", "nothing to sort")
