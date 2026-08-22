"""Ingest new raw/ sources into a vault's wiki, unattended.

Each source goes through three stages, and each stage is its own conversation
with the model:

    1. plan     which pages this source should create or update, and nothing else
    2. execute  one conversation per planned page, which writes that page
    3. log      one wiki/log.md entry for the whole source

They are separate because the model keeps nothing between calls, so a stage's
whole transcript is re-sent every turn and every tool result stays in it until
the stage ends. As one conversation, a source carried the full page list, every
page it read and every page it wrote, all the way to the final log entry — 27,528
tokens on 2026-08-20, 84% of the window, of which 74% was material already on
disk. Split, planning costs ~15k and each page ~10k no matter how many pages the
source touches.

The vault's RULES.md supplies policy — scope, page format, citation rules, which
pages should exist. The procedure is here, in the three stage wrappers below: it
is a property of how this script drives the model, not of any vault, and when it
lived in RULES.md every vault kept its own copy and they drifted apart.

Vault-agnostic: this script has no idea what subject any given vault covers.
Adding a new vault means a new folder with its own RULES.md, not new code.

Usage:
    python wiki_ingest.py --vault ~/Vaults/llm-wiki-learnings
    python wiki_ingest.py --vault ~/Vaults/llm-wiki-learnings --plan-only
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
    EXECUTE_TOOL_SCHEMAS,
    LOG_TOOL_SCHEMAS,
    PLAN_TOOL_SCHEMAS,
    RESERVED,
    append_log,
    get_ingested_sources,
    list_binary_raw_files,
    list_index_sections,
    list_raw_files,
    list_unsorted_raw_files,
    list_wiki_pages,
    mark_ingested,
    move_raw_file,
    parse_raw_folders,
    read_index,
    read_raw_file,
    read_wiki_page,
    scan_raw,
    update_index,
    write_wiki_page,
)

# Iteration caps, per stage. These are small now because each stage is a short
# conversation with one job, where the old single-pass ingest needed 30 to carry
# a whole source from read to log in one context.
#
# Stage 1 reads the source, lists the pages and the index sections, then submits
# the plan: 4 calls, plus room to re-read or retry one.
MAX_PLAN_ITERATIONS = 8
# Stage 2 reads the source, optionally reads the page it is updating, writes it,
# and files it in the index: 4 calls. The extra slack is for the cut-off nudge in
# agent/loop.py, which costs an iteration each time it fires — one page in the
# 2026-08-20 staged e2e run burned three that way before its write landed, and at
# 8 a fourth would have exhausted the conversation with the page unwritten.
MAX_EXECUTE_ITERATIONS = 12
# Stage 3 appends one entry.
MAX_LOG_ITERATIONS = 4

# A plan is a list of page names and one-line intents, not page bodies. 15 pages
# (RULES.md's own upper figure for one source) is roughly 450 tokens, so a plan
# that runs far past this is a model looping rather than a big source.
MAX_PLANNED_PAGES = 25

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
what you decided and why. Never leave a source partially processed and waiting \
for input.

Do not ask any questions. Do not wait for confirmation."""

# The three stage prompts below are the ingest *procedure*. It used to live in
# each vault's RULES.md as a numbered workflow, which broke once an ingest
# stopped being one conversation: no single ordering is true for all three
# stages, and a vault repeating the procedure per copy meant it drifted (the
# aarp vault's copy still asked for hand-written index descriptions months after
# update_index took that over). RULES.md now keeps policy — which pages should
# exist, what goes on them — and this file keeps procedure.
#
# Splitting the stages also split the question of whether the model should reason
# before answering, and the three stages want different answers. Stage 1 judges:
# it decides which pages a source touches, and whether a thing the wiki already
# covers is the same thing under another name. Stages 2 and 3 transcribe — the
# page to write and what happened are both settled by the time they start, and
# handed to them in the prompt.
#
# Measured, and the two kinds part company sharply. Stage 2 with reasoning off
# ran 3.4x faster and *kept more of the page* (94/95 lines against 23/95), so
# stages 2 and 3 pass think=False; agent/loop.py's _run_ollama carries the
# numbers and why the accuracy moves with the speed rather than against it.
# Stage 1 with reasoning off failed to call submit_plan in 3 of 6 trials, so it
# is left alone — a stage that decides nothing is a source not ingested.

PLAN_WRAPPER = """

## Your task right now

You are step 1 of 3. Your only job is to decide WHICH pages this source should \
touch. You are not writing any page content in this step — a later step writes \
each page, one at a time.

1. Read the source document in full.
2. List the existing wiki pages. This is how you tell a new page from an \
existing one: if a page already covers the entity or concept, the action is \
'update', not 'create'. Never propose a second page for something the wiki \
already covers under a different name.
3. List the index sections, so you can name the section each page belongs under.
4. Call submit_plan ONCE with the full list, then stop.

Include in the plan:

- every page that should be created or updated from this source's in-scope \
material
- for each page you are creating, at least one EXISTING page that should gain a \
link back to it, as its own entry with action 'update' and an intent that says \
which link to add. Links out of a new page are only half the connection.

Do not call submit_plan more than once. Do not write page content in the \
'intent' field — one sentence saying what changes is all that is needed."""

EXECUTE_WRAPPER = """

## Your task right now

You are step 2 of 3. You are handling ONE page, named in the message below. \
Every other page in the plan is handled by its own separate step — do not \
write, read, or edit any page except the one you were given.

1. Read the source document for the material this page needs.
2. If the action is 'update', read the page first. write_wiki_page replaces the \
whole file, so anything already on the page that is still true must be carried \
forward into what you write. Everything you leave out is deleted.
3. Call write_wiki_page once with the complete page, following the vault's page \
format above.
4. Call update_index once, naming this page and its section. Do not write a \
description — it is taken from the page's own Summary line. Never send the whole \
index; you are naming one page's home, not rewriting the file.

Then stop. Do not append to the log — a later step does that for the whole \
source at once.

The source filename is NOT a wiki page name. Cite it as plain text in the \
`**Sources**` line and in inline citations, exactly as given. Never write it as \
a wiki-link. Wiki-links point only at page slugs from the list below — lowercase \
with hyphens — so `[[ai-chat-learnings-2026-08-19]]`, never \
`[[AI-Chat-Learnings-2026-08-19.md]]`.

Keep the page focused. If it is getting long, the material probably belongs on \
one of the other pages in the plan, and that page's own step will handle it."""

LOG_WRAPPER = """

## Your task right now

You are step 3 of 3. Every page for this source has already been written. Your \
only job is to record what happened.

Call append_log ONCE with a single entry giving the date, the source filename, \
which pages were created and which were updated, and anything that was skipped \
as out of scope. The message below tells you exactly what happened — report \
that, do not guess or add pages that are not listed.

Then stop. Do not write or read any wiki page."""


class _WriteCounter:
    """Counts the tool calls that actually change the wiki, so the caller can
    tell a real ingest from a run that read a source and silently did nothing."""

    def __init__(self):
        self.count = 0

    def __bool__(self) -> bool:
        return self.count > 0


class _Plan:
    """What stage 1 decided this source should touch.

    Holds validated entries only — see _clean_plan_pages for what is rejected.
    Falsy when nothing survived, which the caller treats the same way it treats
    a stage that wrote nothing: retry, then leave the source for tomorrow.
    """

    def __init__(self):
        self.pages: list[dict] = []
        self.skipped: str = ""

    def __bool__(self) -> bool:
        return bool(self.pages)

    def names(self) -> list[str]:
        return [p["name"] for p in self.pages]


def _clean_plan_pages(pages) -> list[dict]:
    """Keep the entries that are safe to act on, drop the rest silently.

    Stage 2 turns each entry straight into a write, so a malformed entry is not
    a cosmetic problem — an entry naming 'index' would send a page body at the
    file update_index owns. The tool layer refuses that too (write_wiki_page's
    RESERVED check), but a plan that reaches stage 2 carrying it costs a whole
    conversation to find out.

    Duplicates are dropped rather than merged: two entries for one page mean two
    stage-2 conversations racing to replace the same file, and the second would
    overwrite the first with a version written without knowledge of it.
    """
    if not isinstance(pages, list):
        return []

    cleaned, seen = [], set()
    for entry in pages:
        if not isinstance(entry, dict):
            continue
        name = str(entry.get("name") or "").strip()
        if not name:
            continue
        # Normalize the way the tools do, so 'ollama' and 'ollama.md' cannot
        # both survive as separate entries.
        stem = name[:-3] if name.endswith(".md") else name
        if not stem or "/" in stem or "\\" in stem or stem.startswith("."):
            continue
        if f"{stem}.md".lower() in RESERVED:
            continue
        if stem.lower() in seen:
            continue
        seen.add(stem.lower())

        action = str(entry.get("action") or "").strip().lower()
        cleaned.append({
            "name": stem,
            # Anything that is not clearly 'create' is treated as an update.
            # Guessing wrong toward 'update' is the safe direction: stage 2
            # reads the page first, and read_wiki_page on a page that does not
            # exist returns a plain not-found the model can act on. Guessing
            # wrong toward 'create' skips the read and overwrites a real page.
            "action": "create" if action == "create" else "update",
            "intent": str(entry.get("intent") or "").strip(),
            "section": str(entry.get("section") or "").strip(),
        })
    return cleaned[:MAX_PLANNED_PAGES]


def _plan_dispatch(vault_path: str, plan: _Plan) -> dict:
    """Stage 1's tools. submit_plan captures rather than writes — this stage
    touches nothing, which is what makes --plan-only safe by construction."""
    def _submit_plan(pages=None, skipped="", **_ignored):
        cleaned = _clean_plan_pages(pages)
        if not cleaned:
            return {"error": "no usable pages in the plan — every entry needs a "
                             "name, and index/log cannot be planned"}
        plan.pages = cleaned
        plan.skipped = str(skipped or "").strip()
        return {"accepted": len(cleaned), "pages": [p["name"] for p in cleaned]}

    return {
        "read_raw_file": functools.partial(read_raw_file, vault_path),
        "list_wiki_pages": functools.partial(list_wiki_pages, vault_path),
        "list_index_sections": functools.partial(list_index_sections, vault_path),
        "submit_plan": _submit_plan,
        # Unadvertised but dispatchable, in every stage — a vault's RULES.md is
        # part of the system prompt and may name wiki/index.md in prose, and a
        # call that arrives anyway should work rather than come back as an
        # unknown-tool error.
        "read_index": functools.partial(read_index, vault_path),
    }


def _execute_dispatch(vault_path: str, writes: _WriteCounter) -> dict:
    """Stage 2's tools, for one planned page."""
    def _tracked_write_wiki_page(**kwargs):
        result = write_wiki_page(vault_path, **kwargs)
        # Only a write that landed counts. A refused reserved name comes back
        # as an error result, and treating that as progress would mark a page
        # done on the strength of a call that wrote nothing.
        if "error" not in result:
            writes.count += 1
        return result

    return {
        "read_raw_file": functools.partial(read_raw_file, vault_path),
        "read_wiki_page": functools.partial(read_wiki_page, vault_path),
        "write_wiki_page": _tracked_write_wiki_page,
        "update_index": functools.partial(update_index, vault_path),
        "read_index": functools.partial(read_index, vault_path),
    }


def _log_dispatch(vault_path: str, writes: _WriteCounter) -> dict:
    """Stage 3's one tool."""
    def _tracked_append_log(**kwargs):
        writes.count += 1
        return append_log(vault_path, **kwargs)

    return {"append_log": _tracked_append_log}


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


def _attempt(what: str, logger, run) -> bool:
    """Run one stage up to MAX_INGEST_ATTEMPTS times, returning whether it
    succeeded. `run` returns True when the stage did its job.

    The local model intermittently reads its input and then answers without
    calling a tool — a transient no-op rather than a capacity problem (observed
    2026-05-11, which finally wrote its 14 pages on the 4th identical attempt).
    That is what the retry is for, and it applies to all three stages.
    """
    for attempt in range(1, MAX_INGEST_ATTEMPTS + 1):
        try:
            if run():
                return True
        except budget.BudgetExceeded:
            # The whole point of the budget is that it outranks the retry
            # policy — another attempt is exactly what it exists to prevent.
            raise
        except Exception as e:
            logger.exception(
                f"Attempt {attempt}/{MAX_INGEST_ATTEMPTS} of {what} raised: {e}"
            )
            continue
        logger.warning(
            f"Attempt {attempt}/{MAX_INGEST_ATTEMPTS} of {what} did nothing "
            "(model returned without calling the tool it was asked for)."
        )
    return False


def _plan_source(vault_path: str, filename: str, rules: str, logger) -> _Plan:
    """Stage 1: decide which pages this source should touch. Writes nothing."""
    plan = _Plan()

    def run() -> bool:
        plan.pages = []
        result = run_agent(
            system_prompt=rules + UNATTENDED_WRAPPER + PLAN_WRAPPER,
            user_prompt=(
                f"Plan the ingest of the source file '{filename}' from raw/. "
                "Read it, list the existing wiki pages and the index sections, "
                "then call submit_plan once with every page this source should "
                "create or update."
            ),
            tools=PLAN_TOOL_SCHEMAS,
            dispatch=_plan_dispatch(vault_path, plan),
            logger=logger,
            max_iterations=MAX_PLAN_ITERATIONS,
        )
        logger.info(f"Plan response for '{filename}': {result}")
        return bool(plan)

    _attempt(f"planning '{filename}'", logger, run)
    return plan


def _execute_unit(
    vault_path: str, filename: str, unit: dict, plan: _Plan, rules: str, logger
) -> bool:
    """Stage 2: write one planned page, in its own conversation.

    The whole batch's page names go in the prompt. That is what keeps the two
    RULES.md rules that need cross-page awareness working once the pages are no
    longer written in one conversation — links must point at pages that will
    exist, and the same idea must not be written twice under two names. It costs
    about 150 tokens, against the ~7,500 that reading those pages cost when they
    all shared one context.
    """
    writes = _WriteCounter()
    siblings = [n for n in plan.names() if n != unit["name"]]

    def run() -> bool:
        writes.count = 0
        result = run_agent(
            system_prompt=rules + UNATTENDED_WRAPPER + EXECUTE_WRAPPER,
            user_prompt=(
                f"Source file to cite (plain text, never a wiki-link): "
                f"'{filename}'\n"
                f"Page to write: '{unit['name']}'\n"
                f"Action: {unit['action']}\n"
                f"What this page needs from the source: {unit['intent']}\n"
                f"Index section: {unit['section']}\n\n"
                "Other pages being written from this same source, by their own "
                "separate steps. These are the only wiki-links you may add for "
                "this source — link to them where the text calls for it, and do "
                "not duplicate what they cover: "
                + (", ".join(f"[[{n}]]" for n in siblings) if siblings else "none")
            ),
            tools=EXECUTE_TOOL_SCHEMAS,
            dispatch=_execute_dispatch(vault_path, writes),
            logger=logger,
            max_iterations=MAX_EXECUTE_ITERATIONS,
            think=False,
        )
        # Say which of the two happened. This used to read "Wrote 'x'"
        # unconditionally, so a no-op attempt logged a write that never landed —
        # and the log is the record outside the model's reach.
        verb = "Wrote" if writes else "No write from"
        logger.info(f"{verb} '{unit['name']}' for '{filename}': {result}")
        return bool(writes)

    return _attempt(f"writing '{unit['name']}' for '{filename}'", logger, run)


def _write_log_entry(
    vault_path: str, filename: str, plan: _Plan, done: list[dict], rules: str, logger
) -> bool:
    """Stage 3: one log.md entry for the whole source.

    Kept apart from the page writes for two reasons. append_log is the one
    non-idempotent tool in the set, so it belongs where it runs exactly once;
    and this way the entry is written from a record of what actually landed
    rather than from the model's recollection of a long conversation.
    """
    writes = _WriteCounter()
    created = [u["name"] for u in done if u["action"] == "create"]
    updated = [u["name"] for u in done if u["action"] != "create"]

    def run() -> bool:
        writes.count = 0
        result = run_agent(
            system_prompt=rules + UNATTENDED_WRAPPER + LOG_WRAPPER,
            user_prompt=(
                f"Source file ingested: '{filename}'\n"
                f"Pages created: {', '.join(created) if created else 'none'}\n"
                f"Pages updated: {', '.join(updated) if updated else 'none'}\n"
                f"Skipped as out of scope: {plan.skipped or 'nothing'}\n\n"
                "Append one log.md entry recording this."
            ),
            tools=LOG_TOOL_SCHEMAS,
            dispatch=_log_dispatch(vault_path, writes),
            logger=logger,
            max_iterations=MAX_LOG_ITERATIONS,
            think=False,
        )
        logger.info(f"Log entry for '{filename}': {result}")
        return bool(writes)

    return _attempt(f"logging '{filename}'", logger, run)


def _ingest_source(vault_path: str, filename: str, rules: str, logger) -> bool:
    """Plan the source, write each planned page in its own conversation, then
    record what happened. Returns whether the source is fully ingested.

    Fully: every planned page landed. A partial result deliberately returns
    False and leaves the source unmarked, so tomorrow's run redoes it. That is
    safe because write_wiki_page and update_index overwrite by name, and because
    stage 3 — the one non-idempotent step — only runs on a complete pass.

    This is stricter than the old single-pass ingest, which marked a source done
    as soon as *any* write landed. wiki_lint.check_source_coverage exists partly
    to find the half-ingested sources that produced.
    """
    # One retry ceiling for the whole source, spanning all three stages. A spent
    # ceiling means the server is unwell, which is a property of the box and not
    # of any one page.
    budget.start_source(filename, MAX_RETRIES_PER_SOURCE)

    plan = _plan_source(vault_path, filename, rules, logger)
    if not plan:
        logger.warning(
            f"No usable plan for '{filename}' after {MAX_INGEST_ATTEMPTS} "
            "attempts — nothing was written."
        )
        return False

    logger.info(
        f"Plan for '{filename}': {len(plan.pages)} page(s) — "
        + ", ".join(f"{p['name']} ({p['action']})" for p in plan.pages)
    )

    done, failed = [], []
    for unit in plan.pages:
        budget.check(f"page '{unit['name']}' of '{filename}'")
        if _execute_unit(vault_path, filename, unit, plan, rules, logger):
            done.append(unit)
        else:
            failed.append(unit["name"])

    if failed:
        logger.warning(
            f"'{filename}': {len(done)}/{len(plan.pages)} planned page(s) "
            f"written; failed on {', '.join(failed)}. Leaving the source "
            "unmarked so the next run redoes it."
        )
        return False

    # Only now, with every page on disk, is there something true to record.
    return _write_log_entry(vault_path, filename, plan, done, rules, logger)


def ingest_vault(vault_path: str, logger, plan_only: bool = False) -> int:
    rules = _load_rules(vault_path)

    # Organize freshly dropped files into their subdirectories first, so the
    # ingest loop below only ever encounters sorted sources.
    sort_raw_files(vault_path, logger)

    # Binaries never reach the model (see _is_text_source), but a file put in
    # raw/ was put there to be read, so say so rather than ignoring it in
    # silence — the fix is OCR or a vision model, and only a human can decide
    # that's worth doing.
    # INFO, not WARNING. A binary in raw/ is a standing condition — nothing
    # here removes it — so a warning would fire on every scheduled run forever,
    # and LocalLLMAgent's log_inspector reports every WARNING it finds. That is
    # the same reasoning wiki_lint applies to its own findings: something you
    # read, not something that should page you.
    # One walk of raw/, taken after the sort step has finished moving things,
    # and shared by both listings below. They used to walk it once each.
    scan = scan_raw(vault_path)

    binaries = list_binary_raw_files(vault_path, scan).get("files", [])
    if binaries:
        logger.info(
            f"Skipping {len(binaries)} binary source(s) in raw/ — these need OCR "
            f"or a vision model, not a text read: {', '.join(binaries)}"
        )

    raw_files = list_raw_files(vault_path, scan).get("files", [])
    already_ingested = set(get_ingested_sources(vault_path))
    pending = [f for f in raw_files if f not in already_ingested]

    if not pending:
        logger.info("Nothing to ingest — all raw sources already processed.")
        return 0

    if plan_only:
        return _plan_only(vault_path, pending, rules, logger)

    failures = 0
    processed = 0

    for filename in pending:
        logger.info(f"Ingesting '{filename}'")
        try:
            budget.check(f"'{filename}'")
            wrote = _ingest_source(vault_path, filename, rules, logger)
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
                f"'{filename}' was not fully ingested — leaving it unmarked so "
                "the next run retries it."
            )

    return 1 if failures else 0


def _plan_only(vault_path: str, pending: list[str], rules: str, logger) -> int:
    """Show what each pending source would do, and change nothing.

    Stage 1 has no tool that writes (see PLAN_TOOL_SCHEMAS), so this cannot
    touch the vault even if the model tries — the point is to be able to read
    real plans before trusting the stage that acts on them.
    """
    for filename in pending:
        budget.check(f"planning '{filename}'")
        plan = _plan_source(vault_path, filename, rules, logger)
        print(f"\n=== {filename} ===")
        if not plan:
            print("  (no usable plan)")
            continue
        for unit in plan.pages:
            print(f"  [{unit['action']:6}] {unit['name']}")
            print(f"           {unit['intent']}")
            print(f"           index section: {unit['section']}")
        if plan.skipped:
            print(f"  skipped: {plan.skipped}")
    print(f"\n{len(pending)} source(s) planned. Nothing was written.")
    return 0


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
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="Print what each pending source would create or update, then stop. "
             "Writes nothing and marks nothing.",
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
        rc = ingest_vault(args.vault, logger, plan_only=args.plan_only)
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
