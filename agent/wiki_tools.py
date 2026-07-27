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
    if wiki_dir not in candidate.parents and candidate != wiki_dir:
        raise ValueError(f"page name '{name}' resolves outside the wiki directory")
    return candidate


def _safe_raw_path(vault_path: str, name: str) -> Path:
    """Resolve a raw source name to a path inside <vault>/raw, rejecting any
    attempt to escape that directory (e.g. via '../'). The raw-side mirror of
    _safe_page_path — filenames here come from the model too, so they get the
    same containment guard."""
    raw_dir = _raw_dir(vault_path).resolve()
    candidate = (raw_dir / name).resolve()
    if raw_dir not in candidate.parents and candidate != raw_dir:
        raise ValueError(f"raw file name '{name}' resolves outside the raw directory")
    return candidate


def list_raw_files(vault_path: str) -> dict:
    """Every raw source, by basename, wherever it sits under raw/.

    Recurses into the sort subdirectories (daily-*, misc, ...) so files moved
    there by the pre-ingest sort step are still seen. Returns bare basenames —
    not relative paths — so the identity a file is tracked by in
    .ingested.json is unchanged by sorting and nothing gets re-ingested."""
    raw_dir = _raw_dir(vault_path)
    if not raw_dir.exists():
        return {"files": []}
    names = {
        p.name for p in raw_dir.rglob("*")
        if p.is_file() and not p.name.startswith(".")
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
        # direct path.
        matches = [p for p in raw_dir.rglob(filename) if p.is_file()]
        if not matches:
            return {"error": f"raw file '{filename}' not found"}
        path = matches[0]
    try:
        return {"content": path.read_text(encoding="utf-8", errors="replace")}
    except Exception as e:
        return {"error": str(e)}


def list_unsorted_raw_files(vault_path: str) -> dict:
    """Files sitting directly in raw/, i.e. dropped in but not yet sorted into
    one of the vault's subdirectories. These are the sort step's input."""
    raw_dir = _raw_dir(vault_path)
    if not raw_dir.exists():
        return {"files": []}
    files = sorted(
        p.name for p in raw_dir.iterdir()
        if p.is_file() and not p.name.startswith(".")
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


def write_wiki_page(vault_path: str, name: str, content: str) -> dict:
    try:
        path = _safe_page_path(vault_path, name)
    except ValueError as e:
        return {"error": str(e)}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return {"written": path.name}


def read_index(vault_path: str) -> dict:
    path = _wiki_dir(vault_path) / "index.md"
    if not path.is_file():
        return {"content": ""}
    return {"content": path.read_text(encoding="utf-8")}


UNFILED_HEADING = "## Unfiled"


def _linked_page_names(content: str) -> set[str]:
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


def _delink_broken(content: str, valid: set[str]) -> tuple[str, int]:
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


def update_index(vault_path: str, content: str) -> dict:
    """Overwrite wiki/index.md, guaranteeing every wiki page stays linked.

    The model authors the curated sections, but it rewrites the index from
    memory and silently drops pages — including ones it created moments
    earlier in the same run. A dropped page is never re-linked on its own,
    because its source is already marked ingested and won't be reprocessed, so
    the loss is permanent (observed 2026-07-14: 73 of 160 pages orphaned).
    Completeness is therefore enforced here rather than asked for in the
    prompt: any page missing from `content` is appended under Unfiled.
    """
    wiki_dir = _wiki_dir(vault_path)
    wiki_dir.mkdir(parents=True, exist_ok=True)

    pages = list_wiki_pages(vault_path)["pages"]
    valid = {page[: -len(".md")] for page in pages}

    body = _strip_unfiled(content).rstrip("\n")
    body, delinked = _delink_broken(body, valid)
    linked = _linked_page_names(body)
    unfiled = [
        name for name in (page[: -len(".md")] for page in pages)
        if name not in linked
    ]
    if unfiled:
        entries = "\n".join(
            f"- [[{name}]] {_page_summary(vault_path, name)}".rstrip()
            for name in unfiled
        )
        body += f"\n\n{UNFILED_HEADING}\n\n{entries}"

    path = wiki_dir / "index.md"
    path.write_text(body + "\n", encoding="utf-8")
    return {"written": "index.md", "unfiled": len(unfiled), "delinked": delinked}


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
        "description": "Overwrite wiki/index.md with the given full content (include every page, not just new ones). Any page you leave out is appended under an '## Unfiled' heading rather than dropped — move those into the section they belong in.",
        "parameters": {
            "type": "object",
            "properties": {
                "content": {"type": "string", "description": "Full new content of index.md."},
            },
            "required": ["content"],
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
