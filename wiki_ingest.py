"""Ingest new raw/ sources into a vault's wiki, unattended.

Reads the vault's own RULES.md as the system prompt (folder structure, page
format, citation rules — vault-specific, not hardcoded here) and processes
one new raw source at a time via a tool-calling loop against the local
Ollama model. Skips sources already recorded in wiki/.ingested.json.

Vault-agnostic: this script has no idea what subject any given vault covers.
Adding a new vault means a new folder with its own RULES.md, not new code.

Usage:
    python wiki_ingest.py --vault ~/Documents/llm-wiki-[vault]
"""

import argparse
import functools
import sys
from pathlib import Path

from agent.common import setup_logger
from agent.loop import run_agent
from agent.wiki_tools import (
    INGEST_TOOL_SCHEMAS,
    append_log,
    get_ingested_sources,
    list_raw_files,
    list_wiki_pages,
    mark_ingested,
    read_index,
    read_raw_file,
    read_wiki_page,
    update_index,
    write_wiki_page,
)

# A single source can touch 10-15 pages per RULES.md's own note on that,
# well beyond agent.loop's default cap (sized for the much shorter
# LocalLLMAgent tasks this loop was copied from).
MAX_INGEST_ITERATIONS = 30

UNATTENDED_WRAPPER = """

You are running unattended — there is no human available to discuss takeaways \
with or ask clarifying questions. Wherever the rules above would have you ask \
the user something, instead make your best judgment call, proceed, and note \
what you decided and why in the log.md entry you append at the end. Never \
leave a source partially processed and waiting for input.

Do not ask any questions. Do not wait for confirmation."""


def _build_dispatch(vault_path: str, write_counter: list) -> dict:
    """write_counter is a single-element list used as a mutable int — appended
    to whenever a tool that actually changes the wiki gets called, so the
    caller can tell a real ingest from a run that silently did nothing."""

    def _tracked_write_wiki_page(**kwargs):
        write_counter.append(1)
        return write_wiki_page(vault_path, **kwargs)

    def _tracked_append_log(**kwargs):
        write_counter.append(1)
        return append_log(vault_path, **kwargs)

    return {
        "read_raw_file": functools.partial(read_raw_file, vault_path),
        "list_wiki_pages": functools.partial(list_wiki_pages, vault_path),
        "read_wiki_page": functools.partial(read_wiki_page, vault_path),
        "write_wiki_page": _tracked_write_wiki_page,
        "read_index": functools.partial(read_index, vault_path),
        "update_index": functools.partial(update_index, vault_path),
        "append_log": _tracked_append_log,
    }


def _load_rules(vault_path: str) -> str:
    rules_path = Path(vault_path) / "RULES.md"
    if not rules_path.is_file():
        raise FileNotFoundError(
            f"{rules_path} not found — every vault must have its own RULES.md "
            "defining folder structure, page format, and citation rules."
        )
    return rules_path.read_text(encoding="utf-8")


def ingest_vault(vault_path: str, logger) -> int:
    rules = _load_rules(vault_path)
    system_prompt = rules + UNATTENDED_WRAPPER

    raw_files = list_raw_files(vault_path).get("files", [])
    already_ingested = set(get_ingested_sources(vault_path))
    pending = [f for f in raw_files if f not in already_ingested]

    if not pending:
        logger.info("Nothing to ingest — all raw sources already processed.")
        return 0

    failures = 0

    for filename in pending:
        logger.info(f"Ingesting '{filename}'")
        write_counter: list = []
        dispatch = _build_dispatch(vault_path, write_counter)
        try:
            result = run_agent(
                system_prompt=system_prompt,
                user_prompt=(
                    f"Ingest the source file '{filename}' from raw/ per the ingest "
                    "workflow above: read it, create or update the relevant wiki "
                    "pages with wiki-links between related concepts, update "
                    "wiki/index.md, and append a wiki/log.md entry describing what "
                    "changed."
                ),
                tools=INGEST_TOOL_SCHEMAS,
                dispatch=dispatch,
                logger=logger,
                max_iterations=MAX_INGEST_ITERATIONS,
            )
            logger.info(f"Agent final response for '{filename}': {result}")
            if write_counter:
                mark_ingested(vault_path, filename)
            else:
                failures += 1
                logger.warning(
                    f"'{filename}' produced no wiki writes (model returned without "
                    "calling write_wiki_page or append_log) — leaving it unmarked "
                    "so the next run retries it."
                )
        except Exception as e:
            failures += 1
            logger.exception(f"Failed to ingest '{filename}': {e}")

    return 1 if failures else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vault", required=True, help="Path to the Obsidian vault.")
    args = parser.parse_args()

    vault_name = Path(args.vault).name
    logger = setup_logger(f"wiki_ingest.{vault_name}")
    logger.info(f"Starting wiki ingest run for vault: {args.vault}")

    try:
        rc = ingest_vault(args.vault, logger)
        logger.info("Wiki ingest run complete")
        return rc
    except Exception as e:
        logger.exception(f"Wiki ingest run failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
