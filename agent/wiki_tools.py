"""File I/O against an Obsidian vault, parameterized by vault_path so the
same functions work against any vault ([vault] today, others later).

Every function takes vault_path explicitly rather than reading it from env —
that's what lets one script serve N vaults with zero per-vault code.

Vault layout expected:
    <vault>/RULES.md         -- the wiki's own rules (folder structure, page
                                 format, citation rules) — read as-is by the
                                 caller, not touched here.
    <vault>/raw/              -- source documents, never modified
    <vault>/wiki/             -- maintained pages
    <vault>/wiki/index.md     -- table of contents
    <vault>/wiki/log.md       -- append-only operation log
    <vault>/wiki/.ingested.json -- list of raw filenames already processed

Usage:
    python -m agent.wiki_tools list-raw --vault ~/Documents/llm-wiki-[vault]
    python -m agent.wiki_tools list-wiki --vault ~/Documents/llm-wiki-[vault]
"""

import argparse
import json
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


def list_raw_files(vault_path: str) -> dict:
    raw_dir = _raw_dir(vault_path)
    if not raw_dir.exists():
        return {"files": []}
    files = sorted(
        p.name for p in raw_dir.iterdir()
        if p.is_file() and not p.name.startswith(".")
    )
    return {"files": files}


def read_raw_file(vault_path: str, filename: str) -> dict:
    path = _raw_dir(vault_path) / filename
    if not path.is_file():
        return {"error": f"raw file '{filename}' not found"}
    try:
        return {"content": path.read_text(encoding="utf-8", errors="replace")}
    except Exception as e:
        return {"error": str(e)}


def list_wiki_pages(vault_path: str) -> dict:
    wiki_dir = _wiki_dir(vault_path)
    if not wiki_dir.exists():
        return {"pages": []}
    pages = sorted(
        p.name for p in wiki_dir.iterdir()
        if p.is_file() and p.suffix == ".md" and p.name not in ("index.md", "log.md")
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


def update_index(vault_path: str, content: str) -> dict:
    wiki_dir = _wiki_dir(vault_path)
    wiki_dir.mkdir(parents=True, exist_ok=True)
    path = wiki_dir / "index.md"
    path.write_text(content, encoding="utf-8")
    return {"written": "index.md"}


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
        "description": "Overwrite wiki/index.md with the given full content (include every page, not just new ones).",
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
