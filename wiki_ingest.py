"""Ingest new raw/ sources into a vault's wiki, unattended.

Reads the vault's own RULES.md as the system prompt (folder structure, page
format, citation rules — vault-specific, not hardcoded here) and processes
one new raw source at a time via a tool-calling loop against the local
Ollama model. Skips sources already recorded in wiki/.ingested.json.

Vault-agnostic: this script has no idea what subject any given vault covers.
Adding a new vault means a new folder with its own RULES.md, not new code.

Usage:
    python wiki_ingest.py --vault ~/Documents/llm-wiki-learnings
"""

import argparse
import functools
import os
import sys
import time
from pathlib import Path

from agent import budget
from agent.common import setup_logger, trim_launchd_log
from agent.loop import complete_text, run_agent
from agent.notify import notify_failure
from agent.wiki_tools import (
    INGEST_TOOL_SCHEMAS,
    append_log,
    get_ingested_sources,
    list_binary_raw_files,
    list_raw_files,
    list_unsorted_raw_files,
    list_wiki_pages,
    mark_ingested,
    move_raw_file,
    parse_raw_folders,
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

# The local model intermittently reads a source and then returns a final
# answer without ever calling a write tool — a transient no-op, not a
# capacity problem (observed on 2026-05-11, which finally wrote its 14 pages
# on the 4th identical attempt). Re-attempt a few times in-run before
# deferring to the next scheduled run.
MAX_INGEST_ATTEMPTS = 3

# Ceiling on transport retries across everything one source costs — all of its
# attempts, all of their loop iterations. MAX_INGEST_ATTEMPTS bounds attempts
# and _MAX_HTTP_ATTEMPTS bounds one call, but nothing bounded the product
# (30 iterations x 5 http attempts x 3 attempts = up to 450 retries per
# source). 8 is far more than a healthy run has ever needed and stops a wedged
# server in well under a minute.
MAX_RETRIES_PER_SOURCE = 8

# Hard stop for the whole run. The daily job shares one Ollama with the
# LocalLLMAgent chat server and its tasks, and that Ollama runs with
# OLLAMA_NUM_PARALLEL=1 — so an ingest that overruns is not just late, it is
# holding the only slot every other consumer is queued behind. Better to drop
# the remaining sources (they stay unmarked and are retried tomorrow) than to
# keep the queue shut.
DEFAULT_RUN_BUDGET_MINUTES = 45

# A file the sort step can't confidently place goes here rather than blocking
# ingestion — but only if the vault actually declares a "misc" folder.
SORT_FALLBACK_FOLDER = "misc"

SORT_SYSTEM_PROMPT = """\
You sort one dropped file into exactly one destination folder.

You are given the folder names, each with a short description of what belongs \
in it, plus the file's name and the start of its content. Reply with ONLY the \
destination folder name, exactly as it appears in the list, and nothing else — \
no punctuation, no explanation."""

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


def _classify_raw_file(filename: str, content: str, folders: list[dict]) -> str | None:
    """Ask the local model which declared folder a file belongs in, returning
    the matched folder name (as declared) or None if the reply matches none."""
    folder_lines = "\n".join(f"- {f['name']}: {f['description']}" for f in folders)
    system = f"{SORT_SYSTEM_PROMPT}\n\nFolders:\n{folder_lines}"
    user = (
        f"Filename: {filename}\n\n"
        f"Content (may be truncated):\n{content[:2000]}"
    )
    reply = complete_text(system_prompt=system, user_prompt=user).strip().lower()
    by_lower = {f["name"].lower(): f["name"] for f in folders}

    if reply in by_lower:
        return by_lower[reply]
    tokens = reply.split()
    first = tokens[0].strip(".,:;—-") if tokens else ""
    if first in by_lower:
        return by_lower[first]
    for lower, declared in by_lower.items():
        if lower in reply:
            return declared
    return None


def sort_raw_files(vault_path: str, logger) -> None:
    """Organize files dropped into raw/ into the subdirectories the vault
    declares in its RULES.md '## Raw folders' section, using the local model
    to classify each. A no-op for vaults that declare no such section.

    Runs before ingestion so the rest of the run only ever sees sorted files."""
    folders = parse_raw_folders(vault_path)
    if not folders:
        logger.info(
            "No '## Raw folders' section in RULES.md — skipping raw sort step."
        )
        return

    unsorted = list_unsorted_raw_files(vault_path).get("files", [])
    if not unsorted:
        logger.info("No unsorted files in raw/ — nothing to sort.")
        return

    folder_names = {f["name"] for f in folders}
    fallback = SORT_FALLBACK_FOLDER if SORT_FALLBACK_FOLDER in folder_names else None

    for filename in unsorted:
        budget.check(f"sorting '{filename}'")
        budget.start_source(f"sort of {filename}", MAX_RETRIES_PER_SOURCE)
        content = read_raw_file(vault_path, filename).get("content", "")
        try:
            choice = _classify_raw_file(filename, content, folders)
        except budget.BudgetExceeded:
            # Never demoted to "couldn't classify this one" — the run is over.
            raise
        except Exception as e:
            logger.exception(f"Sorting '{filename}' raised: {e}")
            choice = None

        if choice is None:
            if fallback is None:
                logger.warning(
                    f"Could not classify '{filename}' and no '{SORT_FALLBACK_FOLDER}' "
                    "folder is declared — leaving it in place."
                )
                continue
            logger.warning(
                f"Could not classify '{filename}' — defaulting to '{fallback}'."
            )
            choice = fallback

        result = move_raw_file(vault_path, filename, choice)
        if "error" in result:
            logger.warning(
                f"Failed to sort '{filename}' into '{choice}': {result['error']}"
            )
        else:
            logger.info(f"Sorted '{filename}' -> {result['moved']}")


def _ingest_source(vault_path: str, filename: str, system_prompt: str, logger) -> bool:
    """Run the ingest loop for one source, retrying the no-op failure below.
    Returns whether any attempt actually wrote to the wiki."""
    # One retry ceiling for the whole source, spanning every attempt.
    budget.start_source(filename, MAX_RETRIES_PER_SOURCE)

    for attempt in range(1, MAX_INGEST_ATTEMPTS + 1):
        # Fresh dispatch + write_counter per attempt — a retried source is
        # re-processed from scratch. That is safe for the no-op failure
        # this was written for (the model answers without calling a tool,
        # so nothing was written), and for page writes generally, since
        # write_wiki_page and update_index overwrite by name.
        #
        # It is NOT fully safe once the model call can fail *mid-loop*,
        # which a remote provider makes possible (a Gemini 503 that
        # outlasts the backoff in agent/loop.py). An attempt that wrote
        # pages and appended to log.md before dying will, on retry,
        # re-append: append_log is the one non-idempotent tool. The cost
        # is a duplicate ledger entry, not lost or corrupted pages.
        write_counter: list = []
        dispatch = _build_dispatch(vault_path, write_counter)
        try:
            result = run_agent(
                system_prompt=system_prompt,
                user_prompt=(
                    f"Ingest the source file '{filename}' from raw/ per the ingest "
                    "workflow above: read it, create or update the relevant wiki "
                    "pages with wiki-links between related concepts, call "
                    "update_index once per page you wrote to file it under a "
                    "section, and append a wiki/log.md entry describing what "
                    "changed."
                ),
                tools=INGEST_TOOL_SCHEMAS,
                dispatch=dispatch,
                logger=logger,
                max_iterations=MAX_INGEST_ITERATIONS,
            )
            logger.info(
                f"Agent final response for '{filename}' "
                f"(attempt {attempt}/{MAX_INGEST_ATTEMPTS}): {result}"
            )
        except budget.BudgetExceeded:
            # The whole point of the budget is that it outranks the retry
            # policy — another attempt is exactly what it exists to prevent.
            raise
        except Exception as e:
            logger.exception(
                f"Attempt {attempt}/{MAX_INGEST_ATTEMPTS} for '{filename}' "
                f"raised: {e}"
            )
            continue

        if write_counter:
            return True
        logger.warning(
            f"Attempt {attempt}/{MAX_INGEST_ATTEMPTS} for '{filename}' produced "
            "no wiki writes (model returned without calling write_wiki_page or "
            "append_log)."
        )

    return False


def ingest_vault(vault_path: str, logger) -> int:
    rules = _load_rules(vault_path)
    system_prompt = rules + UNATTENDED_WRAPPER

    # Organize freshly dropped files into their subdirectories first, so the
    # ingest loop below only ever encounters sorted sources.
    sort_raw_files(vault_path, logger)

    # Binaries never reach the model (see _is_text_source), but a file put in
    # raw/ was put there to be read, so say so rather than ignoring it in
    # silence — the fix is OCR or a vision model, and only a human can decide
    # that's worth doing.
    binaries = list_binary_raw_files(vault_path).get("files", [])
    if binaries:
        logger.warning(
            f"Skipping {len(binaries)} binary source(s) in raw/ — these need OCR "
            f"or a vision model, not a text read: {', '.join(binaries)}"
        )

    raw_files = list_raw_files(vault_path).get("files", [])
    already_ingested = set(get_ingested_sources(vault_path))
    pending = [f for f in raw_files if f not in already_ingested]

    if not pending:
        logger.info("Nothing to ingest — all raw sources already processed.")
        return 0

    failures = 0
    processed = 0

    for filename in pending:
        logger.info(f"Ingesting '{filename}'")
        try:
            budget.check(f"'{filename}'")
            wrote = _ingest_source(vault_path, filename, system_prompt, logger)
        except budget.BudgetExceeded as e:
            # Either limit abandons the whole run, not just this source. A
            # spent retry ceiling means the *server* is unwell — transport
            # failures are a property of the box, not of the file being read —
            # so the next source would only burn its own ceiling behind it.
            logger.error(
                f"Run stopped after {processed}/{len(pending)} source(s): {e}. "
                "The remaining sources stay unmarked and are retried on the "
                "next scheduled run."
            )
            raise
        processed += 1

        if wrote:
            mark_ingested(vault_path, filename)
        else:
            failures += 1
            logger.warning(
                f"'{filename}' produced no wiki writes after {MAX_INGEST_ATTEMPTS} "
                "attempts — leaving it unmarked so the next run retries it."
            )

    return 1 if failures else 0


def _budget_minutes() -> float:
    raw = os.getenv("WIKI_RUN_BUDGET_MINUTES")
    if not raw:
        return DEFAULT_RUN_BUDGET_MINUTES
    try:
        return float(raw)
    except ValueError:
        return DEFAULT_RUN_BUDGET_MINUTES


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vault", required=True, help="Path to the Obsidian vault.")
    parser.add_argument(
        "--budget-minutes",
        type=float,
        default=None,
        help="Wall-clock budget for the whole run (default: "
             f"$WIKI_RUN_BUDGET_MINUTES or {DEFAULT_RUN_BUDGET_MINUTES}).",
    )
    args = parser.parse_args()

    vault_name = Path(args.vault).name
    logger = setup_logger(f"wiki_ingest.{vault_name}")
    trim_launchd_log(logger)

    minutes = args.budget_minutes if args.budget_minutes is not None else _budget_minutes()
    budget.start_run(minutes * 60)
    logger.info(
        f"Starting wiki ingest run for vault: {args.vault} "
        f"(budget {minutes:g} min)"
    )

    job = f"wiki_ingest[{vault_name}]"
    started = time.monotonic()
    try:
        rc = ingest_vault(args.vault, logger)
        logger.info("Wiki ingest run complete")
    except budget.BudgetExceeded as e:
        mins = (time.monotonic() - started) / 60
        logger.error(f"Wiki ingest run abandoned after {mins:.1f} min: {e}")
        notify_failure(job, f"abandoned after {mins:.1f} min — {e}", logger)
        return 1
    except Exception as e:
        logger.exception(f"Wiki ingest run failed: {e}")
        notify_failure(job, e, logger)
        return 1

    if rc:
        notify_failure(job, "one or more sources produced no wiki writes", logger)
    return rc


if __name__ == "__main__":
    sys.exit(main())
