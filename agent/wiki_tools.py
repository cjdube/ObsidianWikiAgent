"""File I/O against an Obsidian vault, parameterized by vault_path so the
same functions work against any vault ([vault] today, others later).

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
    python -m agent.wiki_tools list-raw --vault ~/Documents/llm-wiki-learnings
    python -m agent.wiki_tools list-wiki --vault ~/Documents/llm-wiki-learnings
"""

import argparse
import glob
import json
import re
import sys
from pathlib import Path


def _raw_dir(vault_path: str) -> Path:
    return Path(vault_path) / "raw"


def _wiki_dir(vault_path: str) -> Path:
    return Path(vault_path) / "wiki"


def _safe_page_path(vault_path: str, name: str) -> Path:
    """Resolve a wiki page name to a path inside <vault>/wiki, rejecting any
    attempt to escape that directory (e.g. via '../')."""
    filename = name if name.endswith(".md") else f"{name}.md"
    wiki_dir = _wiki_dir(vault_path).resolve()
    candidate = (wiki_dir / filename).resolve()
    if wiki_dir not in candidate.parents:
        raise ValueError(f"page name '{name}' resolves outside the wiki directory")
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


def list_raw_files(vault_path: str) -> dict:
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
    Ties break on name so the order stays deterministic."""
    raw_dir = _raw_dir(vault_path)
    if not raw_dir.exists():
        return {"files": []}
    # Keyed by basename (the .ingested.json identity), so a name appearing in
    # two directories is one queue entry — dated by the older copy, which is how
    # long that source has actually been waiting.
    oldest: dict[str, float] = {}
    for p in raw_dir.rglob("*"):
        if not p.is_file() or p.name.startswith(".") or not _is_text_source(p):
            continue
        try:
            mtime = p.stat().st_mtime
        except OSError:
            continue
        if p.name not in oldest or mtime < oldest[p.name]:
            oldest[p.name] = mtime
    return {"files": [name for name, _ in sorted(oldest.items(), key=lambda kv: (kv[1], kv[0]))]}


def list_binary_raw_files(vault_path: str) -> dict:
    """The raw files list_raw_files refuses to hand the model. Surfaced so a
    run can say which sources it ignored — a screenshot or PDF put in raw/ was
    meant to be a source, and needs OCR or a vision model, not silence."""
    raw_dir = _raw_dir(vault_path)
    if not raw_dir.exists():
        return {"files": []}
    names = {
        p.name for p in raw_dir.rglob("*")
        if p.is_file() and not p.name.startswith(".") and not _is_text_source(p)
    }
    return {"files": sorted(names)}


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


def list_unsorted_raw_files(vault_path: str) -> dict:
    """Files sitting directly in raw/, i.e. dropped in but not yet sorted into
    one of the vault's subdirectories. These are the sort step's input.

    Binaries are excluded here too. The sorter classifies a file by reading the
    start of it, so handing it a PNG asks the model to pick a folder from
    binary noise — the same guess-from-the-filename that fabricated content
    downstream, just earlier in the run."""
    raw_dir = _raw_dir(vault_path)
    if not raw_dir.exists():
        return {"files": []}
    files = sorted(
        p.name for p in raw_dir.iterdir()
        if p.is_file() and not p.name.startswith(".") and _is_text_source(p)
    )
    return {"files": files}


def move_raw_file(vault_path: str, filename: str, folder: str) -> dict:
    """Move raw/<filename> into the raw/<folder>/ subdirectory, creating it if
    needed. Rejects any folder that isn't a plain subdirectory name inside
    raw/ (no path separators, no escaping via '..')."""
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
        and p.name not in ("index.md", "log.md")
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


_FRONTMATTER_RE = re.compile(r"^---\s*\n.*?\n---", re.DOTALL)


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
    """
    try:
        path = _safe_page_path(vault_path, name)
    except ValueError as e:
        return {"error": str(e)}
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file() and not content.startswith("---"):
        block = _FRONTMATTER_RE.match(path.read_text(encoding="utf-8"))
        if block:
            content = f"{block.group(0)}\n\n{content.lstrip()}"
    path.write_text(content, encoding="utf-8")
    return {"written": path.name}


def read_index(vault_path: str) -> dict:
    path = _wiki_dir(vault_path) / "index.md"
    if not path.is_file():
        return {"content": ""}
    return {"content": path.read_text(encoding="utf-8")}


UNFILED_HEADING = "## Unfiled"


def linked_page_names(content: str) -> set[str]:
    """Page names already linked from index content, as bare stems —
    [[foo]], [[foo.md]], [[foo|alias]] and [[foo#section]] all mean foo."""
    names = set()
    for target in re.findall(r"\[\[([^\]]+)\]\]", content):
        name = target.split("|", 1)[0].split("#", 1)[0].strip()
        if name.endswith(".md"):
            name = name[:-3]
        if name:
            names.add(name)
    return names


def delink_broken(content: str, valid: set[str]) -> tuple[str, int]:
    """Replace any [[link]] whose target is not a real page with its plain
    display text, returning the cleaned content and the count de-linked.

    The local model authors index links as free text and mistypes a few each
    run (a dropped letter, a doubled hyphen, a since-deleted page), and nothing
    else vets them before they reach disk. A [[target]] that resolves to no
    page is a dead link — clicking it in Obsidian only offers to create an
    empty page — so it is flattened to text (the alias if one was given, else
    the target) rather than left to rot in the table of contents. Typos are
    dropped, never guess-corrected: repointing [[cla-...]] at claude-... risks
    linking the wrong page."""
    count = 0

    def repl(m: re.Match) -> str:
        nonlocal count
        target = m.group(1).split("|", 1)[0].split("#", 1)[0].strip()
        if target.endswith(".md"):
            target = target[:-3]
        if target in valid:
            return m.group(0)
        count += 1
        return m.group(1).split("|", 1)[-1].strip()

    return re.sub(r"\[\[([^\]]+)\]\]", repl, content), count


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
            return re.sub(
                r"\[\[([^\]]+)\]\]",
                lambda m: m.group(1).split("|", 1)[-1].strip(),
                summary,
            )
    return ""


_ENTRY_RE = re.compile(r"^- \[\[([^\]]+)\]\]\s*(.*)$")


def _entry_name(line: str) -> str | None:
    """The page a `- [[name]] description` line points at, or None."""
    m = _ENTRY_RE.match(line)
    if not m:
        return None
    return m.group(1).split("|", 1)[0].split("#", 1)[0].strip().removesuffix(".md")


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
        name = _entry_name(line)
        if name and not _ENTRY_RE.match(line).group(2).strip():
            summary = _page_summary(vault_path, name)
            if summary:
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


def _section_bounds(lines: list[str], section: str) -> tuple[int, int] | None:
    """(heading index, end index) of `## section`, or None if it isn't there."""
    want = section.strip().lstrip("#").strip().casefold()
    for i, line in enumerate(lines):
        if line.startswith("## ") and line[3:].strip().casefold() == want:
            for j in range(i + 1, len(lines)):
                if lines[j].startswith("## "):
                    return i, j
            return i, len(lines)
    return None


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
    path = _wiki_dir(vault_path) / ".ingested.json"
    if not path.is_file():
        return []
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []


def mark_ingested(vault_path: str, filename: str) -> None:
    wiki_dir = _wiki_dir(vault_path)
    wiki_dir.mkdir(parents=True, exist_ok=True)
    path = wiki_dir / ".ingested.json"
    sources = get_ingested_sources(vault_path)
    if filename not in sources:
        sources.append(filename)
    path.write_text(json.dumps(sources, indent=2), encoding="utf-8")


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

READ_INDEX_SCHEMA = {
    "type": "function",
    "function": {
        "name": "read_index",
        "description": "Read the current content of wiki/index.md, the table of contents for the whole wiki.",
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
                "section": {"type": "string", "description": "Section heading it belongs under, e.g. 'AI & Agent Development'. Use an existing heading from the index where one fits; a new one is created if not."},
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

INGEST_TOOL_SCHEMAS = [
    READ_RAW_FILE_SCHEMA,
    LIST_WIKI_PAGES_SCHEMA,
    READ_WIKI_PAGE_SCHEMA,
    WRITE_WIKI_PAGE_SCHEMA,
    READ_INDEX_SCHEMA,
    UPDATE_INDEX_SCHEMA,
    APPEND_LOG_SCHEMA,
]

QUERY_TOOL_SCHEMAS = [
    LIST_WIKI_PAGES_SCHEMA,
    READ_WIKI_PAGE_SCHEMA,
    READ_INDEX_SCHEMA,
]


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
