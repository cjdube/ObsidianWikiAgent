"""File I/O against an Obsidian vault, parameterized by vault_path so the
same functions work against any vault (one today, others later).

Every function takes vault_path explicitly rather than reading it from env —
that's what lets one script serve N vaults with zero per-vault code.

Vault layout expected:
    <vault>/RULES.md         -- the wiki's own rules (folder structure, page
                                 format, citation rules) — read as-is by the
                                 caller, not touched here.
    <vault>/raw/              -- source documents (content never modified;
                                 the pre-ingest sort step may move them into
                                 raw/<folder>/ subdirectories per RULES.md)
    <vault>/wiki/             -- maintained pages
    <vault>/wiki/index.md     -- table of contents
    <vault>/wiki/log.md       -- append-only operation log
    <vault>/wiki/.ingested.json -- list of raw filenames already processed

Usage:
    python -m agent.wiki_tools list-raw --vault ~/Vaults/llm-wiki-learnings
    python -m agent.wiki_tools list-wiki --vault ~/Vaults/llm-wiki-learnings
"""

import argparse
import functools
import glob
import json
import os
import re
import sys
from datetime import date
from pathlib import Path
from typing import NamedTuple

from agent.wikilinks import (
    delink_broken,
    flatten_links,
    link_target,
    linked_page_names,
)


# The two files in wiki/ that Python owns rather than the model. Both are
# reachable by name through write_wiki_page — _safe_page_path only checks the
# path stays inside wiki/, and these are inside it — so naming them here is
# what keeps them out of that tool's reach. See write_wiki_page.
RESERVED = ("index.md", "log.md")


def _raw_dir(vault_path: str) -> Path:
    return Path(vault_path) / "raw"


def _wiki_dir(vault_path: str) -> Path:
    return Path(vault_path) / "wiki"


def _safe_page_path(vault_path: str, name: str) -> Path:
    """Resolve a wiki page name to a path directly inside <vault>/wiki.

    Two things are refused. Escaping the directory (via '../', an absolute
    path, or a symlink out of it) is the obvious one — the comparison is
    against .resolve()d paths, so all three are caught.

    Nesting is the other, and containment alone allowed it: 'topics/foo'
    resolves *inside* wiki/, so write_wiki_page created wiki/topics/ and
    reported success — and then nothing could see the page. list_wiki_pages
    uses iterdir rather than rglob, so the file was missing from the index,
    from every structural check, and from the orphan scan, while update_index
    refused it with "no wiki page named 'topics/foo'". A write that lands
    where nothing reads is worse than a refused one, which at least tells the
    model to pick another name. wiki/ is flat by design; this makes it so.
    """
    filename = name if name.endswith(".md") else f"{name}.md"
    wiki_dir = _wiki_dir(vault_path).resolve()
    candidate = (wiki_dir / filename).resolve()
    if wiki_dir not in candidate.parents:
        raise ValueError(f"page name '{name}' resolves outside the wiki directory")
    if candidate.parent != wiki_dir:
        raise ValueError(
            f"page name '{name}' would nest the page in a subdirectory of "
            f"wiki/, where nothing lists it — wiki pages are flat, so use a "
            f"hyphenated name like '{filename.replace('/', '-')}'"
        )
    return candidate


def _safe_raw_path(vault_path: str, name: str) -> Path:
    """Resolve a raw source name to a path inside <vault>/raw, rejecting any
    attempt to escape that directory (e.g. via '../'). The raw-side mirror of
    _safe_page_path — filenames here come from the model too, so they get the
    same containment guard."""
    raw_dir = _raw_dir(vault_path).resolve()
    candidate = (raw_dir / name).resolve()
    if raw_dir not in candidate.parents:
        raise ValueError(f"raw file name '{name}' resolves outside the raw directory")
    return candidate


def _is_text_source(path: Path) -> bool:
    """Whether a raw file is something the ingest can actually read.

    A binary dropped into raw/ is not merely useless, it is dangerous.
    read_raw_file decodes with errors="replace", so a PNG never fails — it
    returns hundreds of thousands of replacement characters. The model is then
    holding a filename and no legible content, and it writes what the filename
    implies, citing the file for every claim.

    Observed 2026-08-02: WhatPMsGetPaidFor2026.png returned 366k characters of
    noise, of which 156k were replacement chars, and produced roughly
    twenty-five specific assertions about product-management compensation
    across three pages — PRDs, JIRA, standups, moats, activation and retention
    — each tagged (source: WhatPMsGetPaidFor2026.png). None of it appears in
    the file. check_format cannot catch this: it verifies the cited file
    exists, and it did. Fabrication that carries a citation is worse than
    fabrication that does not, because it reads as sourced.

    Git's heuristic, for the same reason git needs one: a NUL byte in the first
    8 KB means binary. Cheap, and it does not care about the extension, so a
    mislabelled file is still caught."""
    try:
        with path.open("rb") as f:
            return b"\x00" not in f.read(8192)
    except OSError:
        return False


class RawScan(NamedTuple):
    """One walk of raw/, classified every way anything here asks about it.

    text:     readable sources by basename, OLDEST FIRST — the ingest queue.
    binary:   the basenames _is_text_source refused, sorted.
    unsorted: readable sources sitting directly in raw/, sorted — the sort
              step's input.

    These were three separate functions doing three separate walks, and each
    walk probes every file (_is_text_source opens it and reads 8 KB looking for
    a NUL byte). A lint asks for all three and so paid for the walk three times;
    an ingest asks for all three too. Answering them from one pass is the whole
    of this type's job.
    """

    text: list[str]
    binary: list[str]
    unsorted: list[str]


def scan_raw(vault_path: str) -> RawScan:
    """Walk raw/ once and classify what is in it. See RawScan.

    Not cached: sort_raw_files moves files mid-run, so a scan taken before the
    sort step does not describe the tree after it. Callers take a fresh one on
    either side of a move and share it everywhere in between.
    """
    raw_dir = _raw_dir(vault_path)
    if not raw_dir.exists():
        return RawScan([], [], [])

    # Keyed by basename (the .ingested.json identity), so a name appearing in
    # two directories is one queue entry — dated by the older copy, which is how
    # long that source has actually been waiting.
    oldest: dict[str, float] = {}
    binary: set[str] = set()
    unsorted: list[str] = []

    for p in raw_dir.rglob("*"):
        if not p.is_file() or p.name.startswith("."):
            continue
        if not _is_text_source(p):
            binary.add(p.name)
            continue
        if p.parent == raw_dir:
            unsorted.append(p.name)
        try:
            mtime = p.stat().st_mtime
        except OSError:
            continue
        if p.name not in oldest or mtime < oldest[p.name]:
            oldest[p.name] = mtime

    return RawScan(
        text=[n for n, _ in sorted(oldest.items(), key=lambda kv: (kv[1], kv[0]))],
        binary=sorted(binary),
        unsorted=sorted(unsorted),
    )


def list_raw_files(vault_path: str, scan: RawScan = None) -> dict:
    """Every raw source the ingest can read, by basename, wherever it sits
    under raw/.

    Recurses into the sort subdirectories (daily-*, misc, ...) so files moved
    there by the pre-ingest sort step are still seen. Returns bare basenames —
    not relative paths — so the identity a file is tracked by in
    .ingested.json is unchanged by sorting and nothing gets re-ingested.

    Binaries are excluded (see _is_text_source). They are reported separately
    by list_binary_raw_files rather than dropped in silence, because a file
    dropped into raw/ was put there to be read.

    OLDEST FIRST, by mtime — this is the ingest queue, and the order decides who
    starves when a run hits its budget. Alphabetical order looks neutral and is
    not: feeds drop files under stable prefixes, so the same prefix sits at the
    tail every single day. Daily-YouTube-* went unfiled from 2026-07-31 onward
    behind Daily-Chrome-* and AI-Chat-Learnings-*, because each day added two
    sources that sorted ahead of it and the run only ever reached two. FIFO
    cannot do that: waiting longest is exactly what earns a source its turn.
    Ties break on name so the order stays deterministic.

    `scan` lets a caller that has already walked raw/ hand that work over
    instead of paying for it twice; it is taken here when omitted."""
    return {"files": (scan or scan_raw(vault_path)).text}


def list_binary_raw_files(vault_path: str, scan: RawScan = None) -> dict:
    """The raw files list_raw_files refuses to hand the model. Surfaced so a
    run can say which sources it ignored — a screenshot or PDF put in raw/ was
    meant to be a source, and needs OCR or a vision model, not silence."""
    return {"files": (scan or scan_raw(vault_path)).binary}


def read_raw_file(vault_path: str, filename: str) -> dict:
    raw_dir = _raw_dir(vault_path)
    try:
        path = _safe_raw_path(vault_path, filename)
    except ValueError as e:
        return {"error": str(e)}
    if not path.is_file():
        # Sorted into a subdirectory since it was last at the top level; find
        # it by basename anywhere under raw/. rglob stays under raw_dir, so the
        # fallback can't escape even though the guard above only vetted the
        # direct path. glob.escape because the name comes from the model: an
        # unescaped '*' or '[' would make this match some *other* file and
        # return it as though it were the one asked for.
        matches = [p for p in raw_dir.rglob(glob.escape(filename)) if p.is_file()]
        if not matches:
            return {"error": f"raw file '{filename}' not found"}
        path = matches[0]
    try:
        return {"content": path.read_text(encoding="utf-8", errors="replace")}
    except Exception as e:
        return {"error": str(e)}


def list_unsorted_raw_files(vault_path: str, scan: RawScan = None) -> dict:
    """Files sitting directly in raw/, i.e. dropped in but not yet sorted into
    one of the vault's subdirectories. These are the sort step's input.

    Binaries are excluded here too. The sorter classifies a file by reading the
    start of it, so handing it a PNG asks the model to pick a folder from
    binary noise — the same guess-from-the-filename that fabricated content
    downstream, just earlier in the run."""
    return {"files": (scan or scan_raw(vault_path)).unsorted}


def move_raw_file(vault_path: str, filename: str, folder: str) -> dict:
    """Move raw/<filename> into the raw/<folder>/ subdirectory, creating it if
    needed. Rejects any folder that isn't a plain subdirectory name inside
    raw/ (no path separators, no escaping via '..'), and refuses to land on a
    name already taken in the destination.

    That last guard matters because Path.rename replaces an existing
    destination silently on POSIX, and raw/ is the one tree whose contents are
    never supposed to change (see the module docstring). Duplicate basenames
    are not hypothetical: list_raw_files exists partly to reconcile "a name
    appearing in two directories". A collision leaves the file where it is and
    reports why — sort_raw_files logs that and moves on, which is the right
    outcome for something only a human can untangle."""
    if "/" in folder or "\\" in folder or folder in ("", ".", ".."):
        return {"error": f"invalid destination folder '{folder}'"}
    raw_dir = _raw_dir(vault_path).resolve()
    try:
        src = _safe_raw_path(vault_path, filename)
    except ValueError as e:
        return {"error": str(e)}
    if not src.is_file():
        return {"error": f"raw file '{filename}' not found"}
    dest_dir = (raw_dir / folder).resolve()
    if raw_dir not in dest_dir.parents:
        return {"error": f"destination '{folder}' resolves outside raw/"}
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / src.name
    if dest.exists():
        return {
            "error": f"'{src.name}' already exists in {folder}/ — "
                     f"leaving it in raw/ rather than overwriting."
        }
    src.rename(dest)
    return {"moved": f"{folder}/{src.name}"}


def parse_raw_folders(vault_path: str) -> list[dict]:
    """The sort destinations declared in the vault's RULES.md '## Raw folders'
    section, as [{"name", "description"}]. Each bullet is `- <name> <desc>`;
    the first whitespace-delimited token is the folder name (so hyphenated
    names like daily-youtube are fine) and the rest is the description used to
    tell the model what belongs there. Returns [] when the vault declares no
    such section — the sort step is opt-in per vault."""
    rules_path = Path(vault_path) / "RULES.md"
    if not rules_path.is_file():
        return []
    folders: list[dict] = []
    in_section = False
    for line in rules_path.read_text(encoding="utf-8").splitlines():
        heading = re.match(r"^#{1,6}\s+(.*)$", line)
        if heading:
            in_section = heading.group(1).strip().lower() == "raw folders"
            continue
        if not in_section:
            continue
        bullet = re.match(r"^\s*[-*]\s+(\S+)\s*(.*)$", line)
        if not bullet:
            continue
        name = bullet.group(1).strip().rstrip(":")
        desc = bullet.group(2).strip().lstrip("—-:").strip()
        if "/" in name or name in (".", ".."):
            continue
        folders.append({"name": name, "description": desc})
    return folders


def list_wiki_pages(vault_path: str) -> dict:
    """Every real wiki page. Skips dotfiles: a vault on any filesystem without
    native extended attributes (exFAT/NTFS drives, network shares) accumulates
    macOS '._name.md' AppleDouble sidecars, which match *.md and would
    otherwise be listed and indexed as pages."""
    wiki_dir = _wiki_dir(vault_path)
    if not wiki_dir.exists():
        return {"pages": []}
    pages = sorted(
        p.name for p in wiki_dir.iterdir()
        if p.is_file()
        and p.suffix == ".md"
        and not p.name.startswith(".")
        and p.name not in RESERVED
    )
    return {"pages": pages}


def read_wiki_page(vault_path: str, name: str) -> dict:
    try:
        path = _safe_page_path(vault_path, name)
    except ValueError as e:
        return {"error": str(e)}
    if not path.is_file():
        return {"error": f"wiki page '{name}' not found"}
    return {"content": path.read_text(encoding="utf-8")}


def page_exists(vault_path: str, name: str) -> bool:
    """Whether `name` is already a page in wiki/.

    Not a model tool — this is how the ingest decides which stage-2 tool set a
    page gets, and a name that would resolve outside wiki/ is answered False
    rather than raising, because to that question it is simply not a page here.
    """
    try:
        return _safe_page_path(vault_path, name).is_file()
    except ValueError:
        return False


_FRONTMATTER_RE = re.compile(r"^---\s*\n.*?\n---", re.DOTALL)

_ESCAPES = {"n": "\n", "t": "\t", "r": "\r", '"': '"', "\\": "\\", "/": "/"}
_ESCAPE_RE = re.compile(r"\\(u[0-9a-fA-F]{4}|.)", re.DOTALL)


def _unescape_once(content: str) -> str:
    def repl(m: re.Match) -> str:
        seq = m.group(1)
        if seq.startswith("u"):
            return chr(int(seq[1:], 16))
        return _ESCAPES.get(seq, m.group(0))

    return _ESCAPE_RE.sub(repl, content)


_ESCAPED_QUOTE_RE = re.compile(r'\\+"')
_ESCAPED_UNICODE_RE = re.compile(r"\\+u([0-9a-fA-F]{4})")


def _quotes_outside_json(content: str) -> str:
    """Undo \\" and \\uXXXX everywhere they are damage rather than syntax.

    Two exemptions, and both are load-bearing — the 2026-08-21 sweep hit both
    and would have corrupted four places without them.

    A ```json fence is the one context where \\" is required rather than
    wrong: inside a JSON string it *is* how a quote is spelled, and decoding it
    breaks the example. Only json — a ```bash fence showing
    `--location \\"Boston,MA,US\\"` is damage like any other, since the shell
    needs no backslash there.

    A line that talks about escaping is showing an escape on purpose. This
    vault has two pages narrating an earlier repair of this very damage
    ("Fixed JSON escape sequences (`\\u2019`)…"), and decoding those deletes
    the subject of the sentence.

    Deliberately narrower than _unescape_once: only quotes and unicode escapes.
    Extending it to \\n or \\t would mangle every code block that legitimately
    shows one, which is the case the ratio test in _decode_if_escaped exists to
    protect.
    """
    out, lang = [], None
    for line in content.split("\n"):
        stripped = line.lstrip()
        if stripped.startswith("```"):
            lang = None if lang is not None else (stripped[3:].strip().lower() or "plain")
            out.append(line)
            continue
        if lang == "json" or "escape" in line.lower():
            out.append(line)
            continue
        line = _ESCAPED_QUOTE_RE.sub('"', line)
        out.append(_ESCAPED_UNICODE_RE.sub(lambda m: chr(int(m.group(1), 16)), line))
    return "\n".join(out)


def _decode_if_escaped(content: str) -> str:
    """Undo JSON escaping the model applied to a page body one time too many.

    The model composes the page, runs it through json.dumps to build the tool
    call, and intermittently emits a body that had already been encoded — so
    what arrives here is the two characters \\ and n where a newline belongs,
    plus \\" for quotes and \\u2019 for a curly apostrophe. It reads fine in the
    model's own transcript and lands on disk as one enormous line. Observed
    2026-08-16 across four pages: every '# Title', '**Summary**:' and
    '**Sources**:' line vanished from the linter's view at once, because there
    were no lines, and one of those bodies then flooded index.md through the
    Summary lookup that reads a line.

    Nothing downstream can recover from it — Obsidian renders the wall of text,
    and a later ingest re-reads the damage as the page's true content — so it is
    caught at the boundary where the page enters the vault.

    Two passes, because the damage arrives in two severities.

    The first is the collapse. The test there is the ratio, not the presence: a
    page may legitimately show \\n inside a fenced code block, but never more
    often than it starts a new line. Decoding repeats because bodies arrive
    doubly and triply encoded.

    The second is _quotes_outside_json, and it exists because the ratio test
    only ever caught the loud half. A body whose newlines survived intact can
    still carry \\" through every quotation on the page, and that passes the
    ratio check untouched — which is how 41 lines across 23 pages came to hold
    escaped quotes that nothing reported and nothing repaired, until they were
    swept by hand on 2026-08-21.
    """
    for _ in range(5):
        if content.count("\\n") <= content.count("\n"):
            break
        content = _unescape_once(content)
    # After the collapse is undone, not before: the line-by-line pass below
    # needs real lines to find the fences it must leave alone.
    content = _quotes_outside_json(content)
    return content


def write_wiki_page(vault_path: str, name: str, content: str) -> dict:
    """Write a page, carrying over any YAML frontmatter the caller left out.

    A page's frontmatter is machine-readable config a human wrote — the marker
    that makes a page an evaluation lens, and the lens's own check settings.
    The model never reproduces it: it reads the page, rebuilds it from the
    template in RULES.md (which starts at '# Title' and never mentions
    frontmatter), and writes back a full body with the block missing. Because
    this tool replaces whole files, that silently deletes the config, and
    nothing downstream reads frontmatter to notice it went — the page still
    looks right in Obsidian. Observed 2026-07-27: ai-slop.md lost `lens: true`
    plus its em-dash and banned-phrase settings in a single ingest, dropping
    out of the consuming agent's lens list and taking its deterministic prose
    checks with it.

    Preserving it here rather than asking for it in the prompt follows
    update_index: the local model does not reliably honour instructions about
    parts of a file it cannot see the purpose of, and a guarantee belongs in
    code. Content that *does* carry frontmatter is written through untouched,
    so this restores what was dropped without making frontmatter immutable.

    index.md and log.md are refused (see RESERVED). Both live inside wiki/, so
    _safe_page_path admits them, and list_wiki_pages hiding them only means the
    model is never *shown* them — nothing stopped it naming one. That left this
    tool as a way around the two guarantees the rest of the module exists to
    provide: update_index owns index.md precisely so a whole-document rewrite
    can't truncate it (the 2026-08-04 stub that stripped 91 descriptions), and
    append_log opens log.md in append mode so the operation history only ever
    grows. This function opens "w", so either name reaching it discards the
    file. The index would partly rebuild itself on the next update_index call;
    the log would not come back at all.
    """
    try:
        path = _safe_page_path(vault_path, name)
    except ValueError as e:
        return {"error": str(e)}
    if path.name in RESERVED:
        # Named tools, not just a refusal: the model is mid-workflow and needs
        # to know where to put this, or it will retry the same call.
        tool = "update_index" if path.name == "index.md" else "append_log"
        return {
            "error": f"'{path.name}' is not writable with write_wiki_page — "
                     f"use {tool} instead."
        }
    path.parent.mkdir(parents=True, exist_ok=True)
    # Before the frontmatter check, which an escaped body would pass falsely:
    # "---\nproject: true..." starts with '---' without holding a real block.
    content = _decode_if_escaped(content)
    # Matched, not startswith: a body that opens with a '---' horizontal rule
    # is not the caller supplying frontmatter, and reading it as one dropped
    # the block this guard exists to preserve.
    if path.is_file() and not _FRONTMATTER_RE.match(content):
        block = _FRONTMATTER_RE.match(path.read_text(encoding="utf-8"))
        if block:
            content = f"{block.group(0)}\n\n{content.lstrip()}"
    path.write_text(content, encoding="utf-8")
    return {"written": path.name}


_RELATED_SECTION = "Related pages"


def _section_title(line: str) -> str | None:
    """The heading text of a `## Section` line, or None for anything else."""
    if not line.startswith("## "):
        return None
    return line[3:].strip()


def _section_bounds(lines: list[str], section: str) -> tuple[int, int] | None:
    """(heading index, end index) of `## section`, or None if it isn't there.

    Shared by index.md and by wiki pages, which is why '## ' is the only thing
    that counts as a section here. On a page that means a '### Subsection' does
    not end its parent — appending to '## Tools' lands after the subsections
    that belong to it, not in front of them — and it keeps one answer to "what
    is a section" for the model, which picks these names out of
    list_index_sections and out of the page it just read.
    """
    want = section.strip().lstrip("#").strip().casefold()
    for i, line in enumerate(lines):
        title = _section_title(line)
        if title is not None and title.casefold() == want:
            for j in range(i + 1, len(lines)):
                if _section_title(lines[j]) is not None:
                    return i, j
            return i, len(lines)
    return None


# The longest real '## ' heading across this vault's 831 of them is 60
# characters, and the median is 13. 80 leaves room for a longer one than any
# page has yet without admitting a paragraph.
_MAX_SECTION_CHARS = 80


def _heading_like(section: str) -> bool:
    """Whether `section` could plausibly be a new '## ' heading.

    Guards the one direction that damages a page — creating a heading. Markdown
    has no illegal heading text, so this is a shape test, not a validity one: a
    heading is one short line, and it is not a list item.
    """
    return (
        "\n" not in section
        and len(section) <= _MAX_SECTION_CHARS
        and not section.startswith(("-", "*", "+", ">", "|"))
    )


def _append_to_section(body: str, section: str, content: str) -> str:
    """`body` with `content` added at the end of `section`, creating it if new.

    A new section is placed before '## Related pages' rather than at the end,
    because that list is the page's last word by the format in RULES.md. When
    the page has no such list — or when the new section *is* it — the content
    goes at the end, which is the same rule with nothing to sit in front of.
    """
    lines = body.split("\n")
    bounds = _section_bounds(lines, section)
    block = content.strip().split("\n")

    if bounds:
        start, end = bounds
        # Back over the blank lines that separate this section from the next,
        # so the addition joins the section's own content rather than the gap.
        at = end
        while at > start + 1 and not lines[at - 1].strip():
            at -= 1
        return "\n".join(lines[:at] + [""] + block + lines[at:])

    related = _section_bounds(lines, _RELATED_SECTION)
    new = [f"## {section.strip()}", ""] + block
    before = lines[:related[0]] if related else lines
    # One blank line before the new heading, however many the split left behind.
    while before and not before[-1].strip():
        before.pop()
    rest = [""] + lines[related[0]:] if related else []
    return "\n".join(before + [""] + new + rest)


def _record_source(line: str, source: str) -> str:
    """The '**Sources**:' line with `source` appended if it is not already named.

    The bare filename, always. RULES.md asks for one with no directory prefix
    and the sort step files sources into raw/<folder>/, so a caller that passes
    the path it walked would write exactly the prefixed citation the rules call
    out ('weekly-learnings/Strategic-Weekly-Review-2026-05-11.md'). Stripping it
    here means no caller can get that wrong.

    The existing text is appended to, never parsed and rebuilt. A raw filename
    may itself contain a comma — this vault has
    'if-you’re-still-hitting-the-claude-code-5-hour-wall,-you’re-doing-it-wrong.md'
    — and splitting the list on ',' to rejoin it turned that one citation into
    two invented ones, which is how wiki_lint found this. The separator is
    comma-*space*, which that filename does not contain, so it is safe to split
    on for the membership test; it is the rejoining that was never safe.
    """
    name = Path(source).name
    prefix = "**Sources**:"
    existing = line[len(prefix):]
    if name in [s.strip() for s in existing.split(", ")]:
        return line
    if not existing.strip():
        return f"{prefix} {name}"
    return f"{line.rstrip()}, {name}"


def edit_wiki_page(
    vault_path: str,
    source: str,
    name: str,
    section: str,
    content: str,
    summary: str = "",
) -> dict:
    """Add `content` to one section of an existing page, leaving the rest alone.

    This is the update path. write_wiki_page replaces whole files, which made
    every update cost a full regeneration of the page — and charged for that
    twice over. Once in time: the 2026-08-21 run regenerated all 119 lines of
    wiki/ollama.md to change 22 of them, and ~16,600 tokens of page body bought
    59 insertions and 45 deletions across the whole run. Once in accuracy: text
    the model re-emits is text it can damage, and it did, in both directions.
    Whole lines went missing (wiki/local-llm-agent.md, 22 insertions and 78
    deletions on 2026-08-20), and lines it was not even editing came back with
    their JSON escaping baked in — the '\\u2014' and '\\"' that 25 pages now
    carry are all survivors of a rewrite that had no reason to touch them.

    Text that is never re-emitted cannot be lost or mangled, so the fix is to
    stop re-emitting it. What the model supplies here is only the new material.

    Three parts of the page are Python's rather than the model's, for the same
    reason update_index owns index descriptions and write_wiki_page preserves
    frontmatter — a guarantee belongs in code:

      - '**Sources**' gains this ingest's source file, as a bare filename
      - '**Last updated**' is set to today, so it can never be a placeholder,
        a parenthetical, or the future date RULES.md forbids
      - everything outside `section` is copied through untouched

    `summary` is optional and replaces the '**Summary**:' line when given: a
    page's one-line description does drift as the page grows, and it is the one
    piece of existing text an update has a real reason to revise.

    Returns 'unchanged' rather than appending when the content is already on the
    page. Replacing a file is naturally idempotent and appending to one is not,
    so a stage that runs twice — a retried conversation, a source re-ingested
    after a failure — would otherwise leave the same paragraph on the page
    twice.
    """
    try:
        path = _safe_page_path(vault_path, name)
    except ValueError as e:
        return {"error": str(e)}
    if path.name in RESERVED:
        tool = "update_index" if path.name == "index.md" else "append_log"
        return {
            "error": f"'{path.name}' is not writable with edit_wiki_page — "
                     f"use {tool} instead."
        }
    if not path.is_file():
        return {
            "error": f"wiki page '{name}' does not exist, so there is nothing "
                     f"to add to. Check the name — this step updates a page "
                     f"that is already in the wiki."
        }

    content = _decode_if_escaped(content)
    if not content.strip():
        return {"error": "content is empty — nothing to add to the page."}

    original = path.read_text(encoding="utf-8")
    block = _FRONTMATTER_RE.match(original)
    head = block.group(0) if block else ""
    body = original[len(head):]

    if content.strip() in body:
        return {"unchanged": path.name, "reason": "that content is already on the page"}

    section = section.strip().lstrip("#").strip()
    if not section:
        return {
            "error": "section is required — name the '## ' heading this goes "
                     "under, e.g. 'Context Management'."
        }
    # A name that is already a heading on the page is always fine, however it
    # reads. The check is only on the ones that would create a heading: the
    # model sometimes passes a whole bullet here, and a page then grows a 319
    # character '## - **LocalLLMAgent**: Included refactoring of...' with the
    # real content filed under it. Observed once in 13 pages on 2026-08-21.
    if _section_bounds(body.split("\n"), section) is None and not _heading_like(section):
        return {
            "error": f"'{section[:40]}…' is page content, not a section "
                     f"heading. Pass a short heading like 'Context Management' "
                     f"as 'section', and put the material in 'content'."
        }

    body = _append_to_section(body, section, content)

    lines = body.split("\n")
    for i, line in enumerate(lines):
        if line.startswith("**Sources**:"):
            lines[i] = _record_source(line, source)
        elif line.startswith("**Last updated**:"):
            lines[i] = f"**Last updated**: {date.today().isoformat()}"
        elif summary.strip() and line.startswith("**Summary**:"):
            lines[i] = f"**Summary**: {_decode_if_escaped(summary).strip()}"

    path.write_text(head + "\n".join(lines), encoding="utf-8")
    return {"edited": path.name, "section": section.strip()}


def read_index(vault_path: str) -> dict:
    path = _wiki_dir(vault_path) / "index.md"
    if not path.is_file():
        return {"content": ""}
    return {"content": path.read_text(encoding="utf-8")}


UNFILED_HEADING = "## Unfiled"


def list_index_sections(vault_path: str) -> dict:
    """The section headings in wiki/index.md — the whole of what a caller needs
    in order to decide where a page goes.

    read_index returns the entire table of contents, which on a 336-page vault
    is 57 KB: roughly 14k tokens, a quarter of the default context window, and
    growing linearly with the vault. Nearly all of it is entry lines the model
    has no use for. update_index takes a page and a section, so the only
    question left to answer is which heading, and there are a few dozen of
    those.

    This is update_index's own trade applied to the read side: the model names
    a destination, Python owns the document. Naming one of forty headings
    cannot be truncated into a valid-looking wrong answer either.

    'Unfiled' is omitted — it is computed by _normalize_index and update_index
    rejects filing into it, so offering it as a destination only buys a
    round-trip.
    """
    unfiled = UNFILED_HEADING[3:].casefold()
    sections = []
    for line in read_index(vault_path)["content"].splitlines():
        title = _section_title(line)
        if title and title.casefold() != unfiled:
            sections.append(title)
    return {"sections": sections}


def _strip_unfiled(content: str) -> str:
    """Drop any previously appended Unfiled section so it is recomputed rather
    than accumulating. A page the model has since filed under a real heading
    just stops reappearing here."""
    kept, skipping = [], False
    for line in content.splitlines():
        if line.strip() == UNFILED_HEADING:
            skipping = True
            continue
        if skipping and line.startswith("## "):
            skipping = False
        if not skipping:
            kept.append(line)
    return "\n".join(kept)


def _page_summary(vault_path: str, name: str) -> str:
    """The page's own '**Summary**:' line, used as its index description.

    Summaries contain wiki-links of their own, which are flattened to plain
    text: carried through verbatim they would add links to the index for
    concepts that have no page (breaking them), and would let a page count as
    linked merely because another entry's description mentions it."""
    try:
        path = _safe_page_path(vault_path, name)
    except ValueError:
        return ""
    if not path.is_file():
        return ""
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("**Summary**:"):
            summary = line[len("**Summary**:"):].strip()
            # Every link, whatever it points at — this text is about to be
            # re-embedded in index.md, where a surviving [[x]] would render as
            # a real entry-description link.
            return flatten_links(summary, lambda target: True)[0]
    return ""


_ENTRY_RE = re.compile(r"^- \[\[([^\]]+)\]\]\s*(.*)$")


def _entry_name(line: str) -> str | None:
    """The page a `- [[name]] description` line points at, or None.

    _ENTRY_RE stays its own pattern rather than reusing LINK_RE: it matches the
    *shape of an index entry line* — anchored, one link, description after —
    not a link anywhere in a body. Only the target extraction is shared."""
    m = _ENTRY_RE.match(line)
    if not m:
        return None
    return link_target(m.group(1))


def _ensure_descriptions(vault_path: str, lines: list[str]) -> list[str]:
    """Give any bare `- [[page]]` entry the page's own Summary line.

    Fills blanks only — an existing description is left exactly as written, so
    curating one by hand in Obsidian survives the next ingest. This is what
    repairs an index the model previously stripped: on 2026-08-04 it rewrote
    index.md down to bare links, and the 91 curated descriptions came back from
    the pages themselves rather than from a git restore, which would only have
    recovered the 91 and left the other 200 as they were.
    """
    out = []
    for line in lines:
        m = _ENTRY_RE.match(line)
        if m and not m.group(2).strip():
            name = link_target(m.group(1))
            if name and (summary := _page_summary(vault_path, name)):
                line = f"- [[{name}]] {summary}"
        out.append(line)
    return out


def _normalize_index(
    vault_path: str, body: str, pages: list[str] = None
) -> tuple[str, int, int]:
    """The guarantees every index write carries, whoever wrote it.

    The model does not reliably honour instructions about a file's overall
    shape, so completeness lives in code: any page not linked from `body` is
    appended under Unfiled rather than lost (observed 2026-07-14, 73 of 160
    pages orphaned by a rewrite from memory). Dead links are flattened, and
    bare entries pick up their page's summary.

    `pages` lets a caller that has already listed the vault hand that work over
    instead of paying for the scan twice; it is derived here when omitted, so
    this stays callable on its own.
    """
    pages = list_wiki_pages(vault_path)["pages"] if pages is None else pages
    valid = {page[: -len(".md")] for page in pages}

    body = _strip_unfiled(body).rstrip("\n")
    body, delinked = delink_broken(body, valid)
    body = "\n".join(_ensure_descriptions(vault_path, body.splitlines()))

    linked = linked_page_names(body)
    unfiled = [name for name in valid if name not in linked]
    if unfiled:
        entries = "\n".join(
            f"- [[{name}]] {_page_summary(vault_path, name)}".rstrip()
            for name in sorted(unfiled)
        )
        body += f"\n\n{UNFILED_HEADING}\n\n{entries}"
    return body, len(unfiled), delinked


def _trimmed(lines: list[str]) -> list[str]:
    """`lines` without leading or trailing blanks — blank lines around a block
    are formatting this module re-adds itself, not content worth preserving."""
    out = list(lines)
    while out and not out[0].strip():
        out.pop(0)
    while out and not out[-1].strip():
        out.pop()
    return out


def _file_into_section(body: list[str], entry: str) -> list[str]:
    """Add `entry` to one section's body, sorted into the run of entries that
    ends it, and leave every other line exactly where its author put it.

    This used to rebuild the body from its entry lines alone, which silently
    deleted anything else under the heading — intro prose, '### ' sub-headings,
    any hand-written note. That is the same loss of human curation that
    write_wiki_page's frontmatter guard and _ensure_descriptions were each
    written to stop, one level up in the document: the index is edited by a
    person in Obsidian as well as by this code, and only one of the two can
    notice something went missing.

    Sorting is therefore scoped to the trailing run rather than the section. A
    section that is nothing but entries — the common case — is one such run, so
    it still sorts whole; a section whose entries are grouped under sub-headings
    keeps its groups instead of being flattened into one alphabetical list.
    """
    tail = len(body)
    while tail and not body[tail - 1].strip():
        tail -= 1
    head = tail
    while head and _entry_name(body[head - 1]):
        head -= 1

    run = sorted(
        body[head:tail] + [entry],
        key=lambda ln: (_entry_name(ln) or "").casefold(),
    )
    kept = _trimmed(body[:head])
    return kept + ([""] if kept else []) + run


def update_index(vault_path: str, page: str, section: str) -> dict:
    """File ONE page under ONE section heading. Python owns the document.

    This used to take the whole of index.md as a string, and that was the
    single most expensive thing in an ingest. Regenerating a 45 KB table of
    contents is ~12k output tokens against a 32k context already holding the
    source and several pages; on 2026-08-04 the model spent 30 of one run's 45
    budgeted minutes on five attempts (45426, 45460, 45450, 12224, 3108 chars),
    timed out three times, and each retry came back shorter until a truncated
    3 KB stub was written over the real index, stripping 91 descriptions. The
    run then hit its budget having filed 2 of 5 sources.

    None of that was the model failing at its job — it was being asked to do
    Python's. Naming a page and a section is ~15 tokens and cannot be truncated
    into a valid-looking wrong answer.
    """
    wiki_dir = _wiki_dir(vault_path)
    wiki_dir.mkdir(parents=True, exist_ok=True)

    # Listed once and handed to _normalize_index below: this used to scan the
    # whole wiki directory twice per call, and the ingest calls it once per page
    # it writes.
    pages = list_wiki_pages(vault_path)["pages"]

    name = page.strip().removesuffix(".md")
    if name not in {p[: -len(".md")] for p in pages}:
        return {"error": f"no wiki page named '{name}' — write the page first"}
    section = section.strip().lstrip("#").strip()
    if not section:
        return {"error": "section is required, e.g. 'AI & Agent Development'"}
    if section.casefold() == UNFILED_HEADING[3:].casefold():
        # Unfiled is computed, not authored: filing INTO it would be undone by
        # the next normalize and reads as "leave this page uncategorised".
        return {"error": "'Unfiled' is maintained automatically — name a real section"}

    lines = read_index(vault_path)["content"].splitlines()
    # Drop any existing entry for this page, wherever it currently sits, so
    # re-filing moves it instead of listing it twice.
    lines = [ln for ln in lines if _entry_name(ln) != name]
    entry = f"- [[{name}]] {_page_summary(vault_path, name)}".rstrip()

    bounds = _section_bounds(lines, section)
    if bounds is None:
        # A new section goes before Unfiled, which always stays last.
        unfiled_at = next(
            (i for i, ln in enumerate(lines) if ln.strip() == UNFILED_HEADING),
            len(lines))
        block = [f"## {section}", "", entry, ""]
        lines = lines[:unfiled_at] + block + lines[unfiled_at:]
    else:
        start, end = bounds
        body = _file_into_section(lines[start + 1:end], entry)
        lines = lines[:start + 1] + [""] + body + [""] + lines[end:]

    body, unfiled, delinked = _normalize_index(
        vault_path, "\n".join(lines), pages=pages
    )
    (wiki_dir / "index.md").write_text(body + "\n", encoding="utf-8")
    return {"filed": name, "section": section, "unfiled": unfiled, "delinked": delinked}


def append_log(vault_path: str, entry: str) -> dict:
    wiki_dir = _wiki_dir(vault_path)
    wiki_dir.mkdir(parents=True, exist_ok=True)
    path = wiki_dir / "log.md"
    with path.open("a", encoding="utf-8") as f:
        f.write(entry.rstrip("\n") + "\n")
    return {"appended": True}


def get_ingested_sources(vault_path: str) -> list[str]:
    """The raw filenames already processed, from the vault's ledger.

    A missing ledger is an empty one — that is simply a vault nothing has
    ingested yet. An unreadable one is not, and used to be treated as the same
    thing. That reading is the worst available: [] means "no source has ever
    been ingested", so the next run re-ingests every file in raw/ — hundreds of
    model calls, every page rewritten from scratch, and a duplicate log.md entry
    for each. check_source_coverage cannot flag it either, since it reads this
    same empty list. Refuse instead, and let the caller's failure alert say so.
    """
    path = _wiki_dir(vault_path) / ".ingested.json"
    if not path.is_file():
        return []
    try:
        sources = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        raise RuntimeError(
            f"{path} is unreadable ({e}) — refusing to read it as an empty "
            f"ledger, which would re-ingest every source in raw/. Restore it "
            f"from a vault snapshot, or delete it if a full re-ingest is "
            f"genuinely what you want."
        ) from e
    if not isinstance(sources, list):
        raise RuntimeError(
            f"{path} holds {type(sources).__name__}, not a JSON list of "
            f"filenames — restore it from a vault snapshot."
        )
    return sources


def mark_ingested(vault_path: str, filename: str) -> None:
    """Record a source as processed, atomically.

    Written to a sibling and renamed over the ledger rather than written in
    place, because write_text truncates before it writes: a crash in that gap
    leaves a file that parses as nothing. The gap is not hypothetical — the
    watchdog in agent/budget.py raises from a signal handler at whatever point
    the process happens to be, so every run that overruns its budget can land
    in it. os.replace is atomic, so the ledger is either the old list or the
    new one and never a half of either.
    """
    wiki_dir = _wiki_dir(vault_path)
    wiki_dir.mkdir(parents=True, exist_ok=True)
    path = wiki_dir / ".ingested.json"
    sources = get_ingested_sources(vault_path)
    if filename not in sources:
        sources.append(filename)
    tmp = path.parent / f"{path.name}.tmp"
    tmp.write_text(json.dumps(sources, indent=2), encoding="utf-8")
    os.replace(tmp, path)


READ_RAW_FILE_SCHEMA = {
    "type": "function",
    "function": {
        "name": "read_raw_file",
        "description": "Read the full content of a source document in the vault's raw/ folder.",
        "parameters": {
            "type": "object",
            "properties": {
                "filename": {"type": "string", "description": "Filename inside raw/, e.g. 'meeting-notes.txt'."},
            },
            "required": ["filename"],
        },
    },
}

LIST_WIKI_PAGES_SCHEMA = {
    "type": "function",
    "function": {
        "name": "list_wiki_pages",
        "description": "List all existing wiki page filenames (excluding index.md and log.md).",
        "parameters": {"type": "object", "properties": {}},
    },
}

READ_WIKI_PAGE_SCHEMA = {
    "type": "function",
    "function": {
        "name": "read_wiki_page",
        "description": "Read the current content of an existing wiki page, to update it rather than overwrite blindly.",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Page name, e.g. 'speakers-bureau' (with or without .md)."},
            },
            "required": ["name"],
        },
    },
}

WRITE_WIKI_PAGE_SCHEMA = {
    "type": "function",
    "function": {
        "name": "write_wiki_page",
        "description": "Create a new wiki page or overwrite an existing one with the given full content.",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Page name, e.g. 'speakers-bureau' (with or without .md)."},
                "content": {"type": "string", "description": "Full markdown content of the page, following the vault's page format."},
            },
            "required": ["name", "content"],
        },
    },
}

EDIT_WIKI_PAGE_SCHEMA = {
    "type": "function",
    "function": {
        "name": "edit_wiki_page",
        "description": "Add new material to ONE section of an existing wiki page. Send only what is new — the rest of the page is kept exactly as it is, so you must NOT repeat, restate, or re-send any text that is already on the page. The '**Sources**' and '**Last updated**' lines are updated for you; do not write them.",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Page name, e.g. 'ollama' (with or without .md)."},
                "section": {"type": "string", "description": "The '## ' heading to add this under — a few words, never a sentence and never the material itself, e.g. 'Context Management'. Use one the page already has where it fits; a new section is created if not. Use 'Related pages' to add a [[wiki-link]] to the page's outward links."},
                "content": {"type": "string", "description": "ONLY the new markdown to add — usually a bullet or a short paragraph. Never the whole page, and never a line that is already on it."},
                "summary": {"type": "string", "description": "Optional. A replacement for the page's one-line '**Summary**:' if this material changes what the page is about. Leave it out otherwise."},
            },
            "required": ["name", "section", "content"],
        },
    },
}

READ_INDEX_SCHEMA = {
    "type": "function",
    "function": {
        "name": "read_index",
        "description": "Read the current content of wiki/index.md, the table of contents for the whole wiki.",
        "parameters": {"type": "object", "properties": {}},
    },
}

LIST_INDEX_SECTIONS_SCHEMA = {
    "type": "function",
    "function": {
        "name": "list_index_sections",
        "description": "List the section headings in wiki/index.md. This is what you need to choose the 'section' argument for update_index — use it instead of reading the whole index, which is large and mostly entries you do not need.",
        "parameters": {"type": "object", "properties": {}},
    },
}

UPDATE_INDEX_SCHEMA = {
    "type": "function",
    "function": {
        "name": "update_index",
        "description": "File ONE page into wiki/index.md under a section heading. Call it once per page you created or changed. NEVER send the whole index — you are naming one page's home, not rewriting the file, and everything else in it is left untouched. The description is taken from the page's own Summary line, so do not write one. Re-filing a page that is already listed moves it to the new section.",
        "parameters": {
            "type": "object",
            "properties": {
                "page": {"type": "string", "description": "Page name, e.g. 'ollama' (no .md)."},
                "section": {"type": "string", "description": "Section heading it belongs under, e.g. 'AI & Agent Development'. Call list_index_sections to see the existing headings and use one where it fits; a new one is created if not."},
            },
            "required": ["page", "section"],
        },
    },
}

APPEND_LOG_SCHEMA = {
    "type": "function",
    "function": {
        "name": "append_log",
        "description": "Append one entry to wiki/log.md recording what changed and why (append-only, never rewrite prior entries).",
        "parameters": {
            "type": "object",
            "properties": {
                "entry": {"type": "string", "description": "One log entry, e.g. '- 2026-07-01: ingested meeting-notes.txt -> created speakers-bureau.md, updated volunteer-roster.md'."},
            },
            "required": ["entry"],
        },
    },
}

SUBMIT_PLAN_SCHEMA = {
    "type": "function",
    "function": {
        "name": "submit_plan",
        "description": "Submit the list of wiki pages this source should touch. Call this ONCE, at the end, after you have read the source and listed the existing pages. Do not write any page content here — a later step writes each page. Include a page for every idea worth recording, and also include any existing page that needs a link added back to a page you are creating.",
        "parameters": {
            "type": "object",
            "properties": {
                "pages": {
                    "type": "array",
                    "description": "One entry per page to create or update.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string", "description": "Page name, e.g. 'ollama' (with or without .md). For an update, use the exact name from list_wiki_pages."},
                            "action": {"type": "string", "enum": ["create", "update"], "description": "'update' if list_wiki_pages already shows this page, otherwise 'create'."},
                            "intent": {"type": "string", "description": "One sentence: what this source adds to this page. For a link-back, say which page to link to, e.g. 'add a link to [[gemma-4]]'."},
                            "section": {"type": "string", "description": "Index section heading this page belongs under. Use one from list_index_sections where it fits."},
                        },
                        "required": ["name", "action", "intent", "section"],
                    },
                },
                "skipped": {"type": "string", "description": "Anything in the source deliberately left out as out of scope, or an empty string. This goes in the log entry."},
            },
            "required": ["pages"],
        },
    },
}

# read_index is deliberately absent from every ingest stage: the two uses for
# the index are "what already exists" (list_wiki_pages, a tenth the size) and
# "which section" (list_index_sections, a hundredth), and offering the 57 KB
# version invites the model to spend a quarter of its context on it. It stays in
# the stage dispatches though — a vault's RULES.md is part of the system prompt
# and may well say "read wiki/index.md" in prose, and a call that arrives anyway
# should work rather than come back as an unknown-tool error.

# Stage 1. Deliberately has no read_wiki_page: deciding *which* pages to touch
# needs their names, not their bodies, and reading them here would pull the
# whole cost this split exists to remove back into the planning context.
PLAN_TOOL_SCHEMAS = [
    READ_RAW_FILE_SCHEMA,
    LIST_WIKI_PAGES_SCHEMA,
    LIST_INDEX_SECTIONS_SCHEMA,
    SUBMIT_PLAN_SCHEMA,
]

# Stage 2, one conversation per planned page. No list_wiki_pages — the plan
# already answered that, and re-listing 418 names per page would reintroduce the
# largest single item in the old single-pass context, once per page instead of
# once per source.
#
# Two sets, and which one a page gets is decided by whether it is already on
# disk (see wiki_ingest._execute_unit) rather than by what the plan called the
# action. The plan is a guess made before anything was read — _clean_plan_pages
# says as much, defaulting an unclear action to 'update' — while the file either
# exists or it does not.
#
# The split is the point, not a convenience. A page that exists is offered no
# way to be overwritten, so the whole-file rewrite that used to lose lines and
# bake in escaping cannot happen on an update at all; a page that does not exist
# is offered no way to be edited, which would only fail. Neither set can do the
# other's job, so the model cannot pick wrong.
CREATE_PAGE_TOOL_SCHEMAS = [
    READ_RAW_FILE_SCHEMA,
    READ_WIKI_PAGE_SCHEMA,
    WRITE_WIKI_PAGE_SCHEMA,
    UPDATE_INDEX_SCHEMA,
]

UPDATE_PAGE_TOOL_SCHEMAS = [
    READ_RAW_FILE_SCHEMA,
    READ_WIKI_PAGE_SCHEMA,
    EDIT_WIKI_PAGE_SCHEMA,
    UPDATE_INDEX_SCHEMA,
]

# Stage 3. One tool, because the only thing left to do is record what the other
# two stages did.
LOG_TOOL_SCHEMAS = [
    APPEND_LOG_SCHEMA,
]

# The read side keeps read_index: wiki_query.py and the lint judgment pass are
# answering questions about the wiki as a whole, which is what a table of
# contents is for, and neither runs in a loop that accumulates page after page.
QUERY_TOOL_SCHEMAS = [
    LIST_WIKI_PAGES_SCHEMA,
    READ_WIKI_PAGE_SCHEMA,
    READ_INDEX_SCHEMA,
]


def query_dispatch(vault_path: str) -> dict:
    """The {name: callable} map for QUERY_TOOL_SCHEMAS, bound to one vault.

    Lives beside the schema list so the two cannot drift — wiki_query.py and
    wiki_lint.py's judgment pass need exactly this set, and each used to build
    it by hand.
    """
    return {
        "list_wiki_pages": functools.partial(list_wiki_pages, vault_path),
        "read_wiki_page": functools.partial(read_wiki_page, vault_path),
        "read_index": functools.partial(read_index, vault_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cmd", choices=["list-raw", "list-wiki"])
    parser.add_argument("--vault", required=True, help="Path to the Obsidian vault.")
    args = parser.parse_args()

    if args.cmd == "list-raw":
        result = list_raw_files(args.vault)
    else:
        result = list_wiki_pages(args.vault)

    print(json.dumps(result, indent=2))
    return 1 if "error" in result else 0


if __name__ == "__main__":
    sys.exit(main())
