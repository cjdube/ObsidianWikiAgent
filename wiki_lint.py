"""Audit a vault's wiki and report problems, on demand.

Report-only by default: without --fix it never writes to the vault. RULES.md
asks for "findings as a numbered list with suggested fixes", and the fixes
that matter need a human — deciding which of two overlapping pages survives is
judgment, not mechanics. The opt-in --fix flag applies only the provably-safe,
mechanical subset (stripping self-links, de-linking dead index entries) and
leaves every judgment call untouched.

Two passes, split by what each is actually good at:

  Structural checks run in Python. Link integrity, orphans, index
  completeness and page format are exact, enumerable facts over every page.
  Asking a model to enumerate every page correctly is what produced the
  orphaned index in the first place (see update_index in agent/wiki_tools.py),
  so these run as code: instant, free, and they cannot miss one.

  --deep adds a model pass for the checks code cannot do: contradictions
  between pages, two pages covering one concept under different names, pages
  whose subject is out of scope, and claims a newer source has overtaken. The
  model receives the structural findings as context so it does not re-derive
  them.

The prose report goes to stdout, for a human. Alongside it the run is logged
through setup_logger like the ingest and snapshot jobs are — run boundaries,
per-section counts, and the judgment pass's tool calls. That is what makes a
scheduled lint show up as a run in LocalLLMAgent's dashboard, which reports on
this repo's launchd jobs (see its docs/external-tasks.md). Findings log at INFO,
never WARNING: they're a weekly read, not an alert.

--json swaps the prose report for a machine-readable one, for a caller that
renders its own UI (LocalLLMAgent's /wiki/lint view shells out to exactly this).
It is the structural pass only, and it is deliberately silent — see _lint_json.

Usage:
    python wiki_lint.py --vault ~/Vaults/llm-wiki-learnings
    python wiki_lint.py --vault ~/Vaults/llm-wiki-learnings --deep
    python wiki_lint.py --vault ~/Vaults/llm-wiki-learnings --fix
    python wiki_lint.py --vault ~/Vaults/llm-wiki-learnings --json
"""

import argparse
import datetime
import json
import re
import sys
from pathlib import Path

from agent import budget
from agent.common import setup_logger, trim_launchd_log
from agent.loop import run_agent
from agent.notify import notify_failure
from agent.wiki_tools import (
    QUERY_TOOL_SCHEMAS,
    RawScan,
    get_ingested_sources,
    list_wiki_pages,
    query_dispatch,
    read_index,
    scan_raw,
)
from agent.wikilinks import (
    LINK_RE,
    delink_broken,
    linked_page_names,
    strip_links_to,
)

# Bounds for the --deep pass only. This is a scheduled unattended job against a
# provider that can be slow or down, and 60 iterations x 5 HTTP attempts is the
# same unbounded product that had wiki_ingest retrying for three hours. Smaller
# than the ingest's 45 minutes because this is one conversation, not a queue of
# sources: the pass normally costs a couple of minutes.
DEEP_RUN_BUDGET_MINUTES = 30
MAX_DEEP_RETRIES = 8

LINT_WRAPPER = """

You are auditing this wiki. A structural pass has already run in code and
found every broken link, orphan page, index gap and page-format violation —
those are listed below, and you must NOT repeat them or re-check them.

Your job is only the checks that need judgment:

1. Contradictions — two pages asserting incompatible things.
2. Duplicate concepts — two pages covering the same subject under different
   names (they will not share a title, or the structural pass would have
   caught them).
3. Out-of-scope pages — subjects the Scope section above excludes.
4. Outdated claims — a claim a later source has superseded.

Read wiki/index.md first, then read the pages you need. Report findings as a
numbered list, each naming the specific pages and a suggested fix. Report only
what you actually verified by reading the pages — if you find nothing in a
category, say so rather than inventing something. Do not write to the wiki."""


def _pages(vault_path: str) -> dict[str, str]:
    """{slug: content} for every real wiki page."""
    wiki_dir = Path(vault_path) / "wiki"
    return {
        name[: -len(".md")]: (wiki_dir / name).read_text(encoding="utf-8", errors="replace")
        for name in list_wiki_pages(vault_path)["pages"]
    }


def _field(content: str, label: str) -> str | None:
    for line in content.splitlines():
        if line.startswith(f"**{label}**:"):
            return line.split(":", 1)[1].strip()
    return None


def _title(content: str) -> str | None:
    """The page's '# Title' heading, or None if it has none."""
    return next(
        (l[2:].strip() for l in content.splitlines() if l.startswith("# ")), None
    )


def check_links(pages: dict[str, str]) -> list[str]:
    """Wiki-links pointing at pages that do not exist, and self-links."""
    findings = []
    for slug, content in sorted(pages.items()):
        for target in sorted(linked_page_names(content)):
            if target == slug:
                findings.append(f"{slug}.md links to itself — remove the self-link.")
            elif target not in pages:
                findings.append(
                    f"{slug}.md links to [[{target}]], which has no page — "
                    f"create it, repoint the link, or make it plain text."
                )
    return findings


# A slug ending in an ISO date is a dated chronological log (daily-chrome-…,
# ai-chat-learnings-…, strategic-weekly-review-…), not a concept. Nothing has
# reason to link *to* "July 22's browsing log," so being reachable only from
# the index is the right bar for these — flagging them as orphans reports a
# non-problem every ingest run creates.
_DATED_LOG = re.compile(r"-\d{4}-\d{2}-\d{2}$")


def check_orphans(pages: dict[str, str]) -> list[str]:
    """Concept pages nothing links to. The index does not count: being listed
    in the table of contents is not the same as being reachable from related
    work. Dated chronological logs are exempt (see _DATED_LOG) — they are leaf
    notes by nature, and still count as linkers to the pages they reference."""
    inbound = {slug: set() for slug in pages}
    for slug, content in pages.items():
        for target in linked_page_names(content):
            if target in inbound and target != slug:
                inbound[target].add(slug)
    return [
        f"{slug}.md is an orphan — no other page links to it. Link it from a "
        f"related page, or reconsider whether it earns its own page."
        for slug in sorted(pages)
        if not inbound[slug] and not _DATED_LOG.search(slug)
    ]


def check_index(vault_path: str, pages: dict[str, str]) -> list[str]:
    """Index completeness in both directions. update_index guarantees every
    page is listed, but it never prunes links to pages that were deleted."""
    linked = linked_page_names(read_index(vault_path)["content"])
    findings = [
        f"{slug}.md is not linked from index.md — update_index should have "
        f"caught this; check it ran."
        for slug in sorted(set(pages) - linked)
    ]
    findings += [
        f"index.md links to [[{name}]], which has no page — the page was "
        f"deleted and the index entry left behind; remove the entry."
        for name in sorted(linked - set(pages))
    ]
    return findings


def _cited_sources(sources: str, raw: set[str]) -> list[str]:
    """Split a Sources line into the filenames it cites.

    Entries are comma-separated, but a filename may itself contain a comma —
    clippings are named after the article title, and titles have commas in
    them. Splitting blind turns one real citation into two fragments that
    match nothing, and the page gets reported for inventing sources it cited
    correctly. Observed 2026-08-02: six pages citing one clipping produced
    twelve findings, every one of them false.

    There is no delimiter that tells a separating comma from a filename's own,
    so the raw/ listing arbitrates: consecutive fragments are rejoined when
    they spell a file that actually exists, longest run first. A citation that
    matches nothing is left split on its commas — it is already being reported
    as unresolvable, and guessing at its boundaries would not change that."""
    parts = sources.split(",")
    cited, i = [], 0
    while i < len(parts):
        for j in range(len(parts), i, -1):
            candidate = ",".join(parts[i:j]).strip().strip("[]`")
            if candidate in raw:
                cited.append(candidate)
                i = j
                break
        else:
            cited.append(parts[i].strip().strip("[]`"))
            i += 1
    return cited


def _letters(s: str) -> str:
    """A name reduced to its letters and digits, for comparing a slug against a
    title without tripping over how each spells the gaps between words."""
    return re.sub(r"[^a-z0-9]", "", s.lower().replace("&", "and"))


def _edit_distance(a: str, b: str) -> int:
    """Levenshtein distance. Only ever called on two spellings of one page
    name, so the quadratic cost is over a few dozen characters."""
    if a == b:
        return 0
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def check_slug_typos(pages: dict[str, str]) -> list[str]:
    """Pages whose filename looks like a misspelling of their own title.

    write_wiki_page takes the slug as an argument rather than deriving it, so
    the model can title a page 'Ollama Thread Wedges' and file it as
    olloma-thread-wedges.md. Nothing downstream notices: the page is written,
    the index entry is built from the same wrong slug, and the run exits clean
    (observed 2026-08-05). The result is a page whose filename contradicts its
    subject and which no search for the real spelling will find.

    Slug and title legitimately diverge all the time, so the comparison ignores
    everything that is not a letter or digit — 'Innovator's Dilemma' filing as
    innovators-dilemma, 'LocalLLMAgent' as local-llm-agent and 'Context Routing
    Trade-offs' as context-routing-tradeoffs are all differences in separators
    only, and all reduce to a distance of 0. What is left is a difference
    *inside* a word, which is what a typo is.

    The threshold is one character, which is narrow on purpose. Over the 312
    pages this was tuned against, distance 1 flagged exactly the real typo and
    nothing else; distance 2 additionally flagged hybrid-agent-architecture,
    titled 'Hybrid AI Agent Architecture', where the shorter slug is a
    deliberate choice and not a mistake. A two-character typo therefore slips
    through — the trade is deliberate, since a check that cries wolf on
    correct pages is one nobody reads.
    """
    findings = []
    for slug, content in sorted(pages.items()):
        title = _title(content)
        if not title:
            continue
        if _edit_distance(_letters(title), _letters(slug)) == 1:
            findings.append(
                f"{slug}.md is titled '# {title}', which spells the name "
                f"differently — the filename looks like a typo. Rename the "
                f"page and update the [[{slug}]] link in index.md."
            )
    return findings


def check_format(
    vault_path: str,
    pages: dict[str, str],
    today: datetime.date,
    scan: RawScan = None,
) -> list[str]:
    """The page format RULES.md requires — the defect classes a weaker model
    reliably produces: slug-as-title, placeholder and future dates, and
    citations to sources that do not exist."""
    # Binaries too: the queue deliberately hides them from the *model*, but a
    # citation's job is to point at a file that exists, and a PDF or screenshot
    # in raw/ does. Checking against the readable set alone reported every page
    # citing one as having invented the source — the same class of false
    # positive _cited_sources was written to kill, through another door.
    scan = scan or scan_raw(vault_path)
    raw = set(scan.text) | set(scan.binary)
    findings = []
    for slug, content in sorted(pages.items()):
        title = _title(content)
        if not title:
            findings.append(f"{slug}.md has no '# Title' heading.")
        elif title == slug:
            findings.append(
                f"{slug}.md uses its slug as the title ('# {title}') — "
                f"give it a real human-readable title."
            )

        for label in ("Summary", "Sources", "Last updated"):
            if _field(content, label) is None:
                findings.append(f"{slug}.md is missing its '**{label}**:' line.")

        updated = _field(content, "Last updated")
        if updated is not None:
            # The regex only proves the shape. '2026-13-45' matches it and then
            # makes fromisoformat raise — which aborted the entire lint run,
            # report and all, on a page this very check exists to report. An
            # invented month is exactly what a model that invents dates emits,
            # so it is reported like any other unusable date rather than raised.
            parsed = None
            if re.fullmatch(r"\d{4}-\d{2}-\d{2}", updated):
                try:
                    parsed = datetime.date.fromisoformat(updated)
                except ValueError:
                    pass
            if parsed is None:
                findings.append(
                    f"{slug}.md has a non-ISO 'Last updated' ({updated!r}) — "
                    f"use a plain date like {today.isoformat()}."
                )
            elif parsed > today:
                findings.append(
                    f"{slug}.md is dated {updated}, in the future — "
                    f"the model invented a date."
                )

        sources = _field(content, "Sources")
        if sources:
            for src in _cited_sources(sources, raw):
                if not src:
                    continue
                if "/" in src:
                    findings.append(
                        f"{slug}.md cites '{src}' with a directory prefix — "
                        f"cite the bare filename '{Path(src).name}'."
                    )
                elif src not in raw:
                    findings.append(
                        f"{slug}.md cites '{src}', which is not a file in raw/ — "
                        f"the citation is invented or the source was removed."
                    )
    return findings


# A lens page carries YAML frontmatter — `lens: true` plus its own check
# config — that a consuming agent reads to discover and configure it. Only the
# body reliably survives an ingest: write_wiki_page replaces the whole file and
# the model rebuilds the page from what it read, dropping frontmatter it has no
# reason to reproduce. Observed 2026-07-27: ai-slop.md lost `lens: true`,
# max_em_dashes_per_sentence and banned_phrases in a single run, silently fell
# out of the consumer's lens list, and took its mechanical prose checks with it
# — no error anywhere, because nothing reads frontmatter to know it's missing.
# The body always names the consuming tool, so that prose is the tell that the
# marker should still be there.
#
# This is the one check keyed to a convention outside the vault; every other
# check here is pure wiki structure. It earns the coupling by catching silent
# deletion of hand-authored config that nothing else can detect.
_LENS_PROSE = re.compile(r"evaluate_against")
_LENS_MARKER = re.compile(r"^lens:\s*true\s*$", re.MULTILINE | re.IGNORECASE)
_FRONTMATTER = re.compile(r"^---\s*\n(.*?)\n---", re.DOTALL)


def check_lens_frontmatter(pages: dict[str, str]) -> list[str]:
    """Pages that describe themselves as an evaluation lens but no longer carry
    the `lens: true` marker that makes them one.

    Dated logs are exempt (see _DATED_LOG). A lens is a concept page a human
    authored; a dated capture is a record of one day, and RULES.md step 4 names
    it after its source rather than after any concept. So a dated page can only
    ever *mention* the tool — ai-chat-learnings-2026-08-03.md was reported for
    the line "Updated `docs/tool-loading.md` to include `evaluate_against` in
    the `web` group", which is a note about a day's work, not a self-description.
    Exempting them cannot hide a real lens, and the alternative (reading around
    inline code) would still flag the next page that discusses the tool in prose.
    """
    findings = []
    for slug, content in sorted(pages.items()):
        if _DATED_LOG.search(slug) or not _LENS_PROSE.search(content):
            continue
        block = _FRONTMATTER.match(content)
        if block and _LENS_MARKER.search(block.group(1)):
            continue
        findings.append(
            f"{slug}.md describes itself as an evaluate_against lens but has no "
            f"'lens: true' frontmatter — an ingest rewrite drops the whole YAML "
            f"block; restore it from a vault snapshot or the lens is invisible."
        )
    return findings


def check_duplicate_titles(pages: dict[str, str]) -> list[str]:
    """Two pages with the same title are the same page. Concepts duplicated
    under *different* titles need judgment and are left to the --deep pass."""
    by_title: dict[str, list[str]] = {}
    for slug, content in pages.items():
        title = _title(content)
        if title:
            by_title.setdefault(title.lower(), []).append(slug)
    return [
        f"{' and '.join(f'{s}.md' for s in sorted(slugs))} share the title "
        f"'{title}' — merge them and redirect inbound links."
        for title, slugs in sorted(by_title.items())
        if len(slugs) > 1
    ]


# Two pages whose text is identical once each one's own name is blanked out were
# written from a single template with the subject swapped. Observed 2026-08-02:
# fable.md and sol.md, two genuinely different products that one source named in
# a single clause and distinguished in no way, so the model asserted the same
# four facts about both. Nothing structural caught it — the titles differ so
# check_duplicate_titles was blind, and each links to the other so neither was an
# orphan — and it took a model pass over the whole vault to notice something a
# string comparison settles for free.
#
# Names are blanked instead of comparing pages pairwise so this stays one pass
# over the vault. Blanking every link target is what absorbs the asymmetry these
# pairs always have: each twin links to the other, and to nothing else different.
_UPDATED_LINE = re.compile(r"^\*\*Last updated\*\*:.*$", re.MULTILINE)

# Below this, a skeleton is too short to mean anything — two pages that are
# little more than a heading would otherwise match each other on structure alone.
_MIN_SKELETON = 80


def _skeleton(slug: str, content: str) -> str:
    """The page with everything naming *this particular page* removed: its slug
    and title wherever they appear, every link target, and the Last updated date
    (metadata, and the one field twins routinely differ on). What survives is the
    template the page was filled in from."""
    body = _UPDATED_LINE.sub("", content)
    body = LINK_RE.sub("[[·]]", body)
    title = _title(content) or ""
    names = {n for n in (slug, slug.replace("-", " "), title) if n}
    # Longest first, so "model fusion" is blanked before a bare "model" can
    # carve it up into something that no longer matches its twin.
    for name in sorted(names, key=len, reverse=True):
        body = re.sub(rf"\b{re.escape(name)}\b", "·", body, flags=re.IGNORECASE)
    return " ".join(body.split())


def check_template_twins(pages: dict[str, str]) -> list[str]:
    """Pages that are one template filled in twice. Unlike check_duplicate_titles
    this catches twins with different names, and unlike the --deep pass it costs
    nothing and cannot hallucinate: the pages really are character-identical
    apart from what names them."""
    by_skeleton: dict[str, list[str]] = {}
    for slug, content in sorted(pages.items()):
        skeleton = _skeleton(slug, content)
        if len(skeleton) >= _MIN_SKELETON:
            by_skeleton.setdefault(skeleton, []).append(slug)
    return [
        f"{' and '.join(f'{s}.md' for s in slugs)} are one template filled in "
        f"twice — identical once each page's own name is removed. Merge them, or "
        f"give each page content specific to its own subject."
        for slugs in sorted(by_skeleton.values())
        if len(slugs) > 1
    ]


def check_source_coverage(
    vault_path: str, pages: dict[str, str], scan: RawScan = None
) -> list[str]:
    """Dated sources the ingest recorded as done that produced no page of their
    own — the one thing RULES.md step 4 makes mandatory for a dated capture.

    _ingest_source marks a source ingested as soon as *any* write lands, so a
    run that wrote one topic page and stopped is indistinguishable from one that
    did the whole job, and .ingested.json then guarantees it is never retried.
    Observed 2026-08-03: AI-Chat-Learnings-2026-08-03.md was marked done having
    written 3 of its 8 sessions into topic pages, with no dated page and no
    log.md entry at all. Two sessions' material is simply absent from the vault,
    and nothing reported it for two weeks.

    The dated page is the check because it is the only per-source artefact whose
    name is knowable in advance. Topic pages vary by content and cannot be
    predicted; the log entry is free text. A missing dated page does not prove
    material was lost, but every loss of this shape shows up as one.

    Scoped to sources still present in raw/ and already marked ingested: a
    source still in the queue has not failed at anything, and one deleted from
    raw/ is a decision, not a gap."""
    ingested = set(get_ingested_sources(vault_path))
    named = {_letters(slug) for slug in pages}
    findings = []
    for name in sorted((scan or scan_raw(vault_path)).text):
        stem = name.rsplit(".", 1)[0]
        if name not in ingested or not _DATED_LOG.search(stem):
            continue
        if _letters(stem) not in named:
            findings.append(
                f"{name} is recorded as ingested but has no page named after "
                f"it — the ingest stopped early and marked it done anyway. "
                f"Remove it from wiki/.ingested.json to re-run it."
            )
    return findings


def structural_findings(
    vault_path: str,
    today: datetime.date = None,
    pages: dict[str, str] = None,
    scan: RawScan = None,
) -> dict[str, list[str]]:
    """Every structural check, over one read of the pages and one walk of raw/.

    `pages` and `scan` exist for the same reason: two checks each need the raw/
    listing and every check needs the pages, and each used to fetch its own. A
    lint therefore walked raw/ three times, probing every file for a NUL byte
    each time.
    """
    today = today or datetime.date.today()
    pages = _pages(vault_path) if pages is None else pages
    scan = scan_raw(vault_path) if scan is None else scan
    return {
        "Broken and self links": check_links(pages),
        "Orphan pages": check_orphans(pages),
        "Index integrity": check_index(vault_path, pages),
        "Source coverage": check_source_coverage(vault_path, pages, scan),
        "Page format": check_format(vault_path, pages, today, scan),
        "Misspelled slugs": check_slug_typos(pages),
        "Duplicate titles": check_duplicate_titles(pages),
        "Template twins": check_template_twins(pages),
        "Lens integrity": check_lens_frontmatter(pages),
    }


def apply_safe_fixes(vault_path: str, pages: dict[str, str]) -> list[str]:
    """Apply only the provably-safe, mechanical fixes and return a log of what
    changed. This is the one place wiki_lint writes to the vault, gated behind
    --fix. Every judgment call — orphans, duplicate concepts, broken body links
    (create-vs-delink), bad dates, invented citations — is deliberately left
    for a human. `pages` is updated in place to reflect the writes.

    Two fixes qualify as safe:
      1. Self-links — a page linking to itself is never meaningful.
      2. Dead index links — a table-of-contents entry pointing at no real page
         (reusing the same de-linking the ingest guard applies).

    Both go through agent/wikilinks.py, which is also what check_links reads
    links with. That shared pattern is the point: this function writes to the
    vault on the strength of what a check reported, so a fix that recognised
    links the check did not could edit a page nobody was told about."""
    wiki_dir = Path(vault_path) / "wiki"
    changes: list[str] = []

    for slug in sorted(pages):
        cleaned, n = strip_links_to(pages[slug], slug)
        if n:
            (wiki_dir / f"{slug}.md").write_text(cleaned, encoding="utf-8")
            pages[slug] = cleaned
            changes.append(f"{slug}.md: removed {n} self-link{'s' if n > 1 else ''}")

    index_path = wiki_dir / "index.md"
    if index_path.is_file():
        cleaned, n = delink_broken(index_path.read_text(encoding="utf-8"), set(pages))
        if n:
            index_path.write_text(cleaned, encoding="utf-8")
            changes.append(f"index.md: de-linked {n} dead link{'s' if n > 1 else ''}")

    return changes


def _render(findings: dict[str, list[str]]) -> tuple[str, int]:
    lines, n = [], 0
    for section, items in findings.items():
        if not items:
            continue
        lines.append(f"\n## {section}\n")
        for item in items:
            n += 1
            lines.append(f"{n}. {item}")
    return "\n".join(lines), n


def _lint(args, rules_path: Path, logger) -> int:
    """Run the passes and print the report. Returns the structural finding
    count. Split out of main() so main() owns the run markers and the one
    try/except that turns a crash into a phone alert."""
    pages = _pages(args.vault)
    today = datetime.date.today()

    # Dated so each run stands apart in the appended launchd log.
    print(f"# Wiki lint — {Path(args.vault).name} — {today.isoformat()}\n")
    if args.fix:
        changes = apply_safe_fixes(args.vault, pages)
        print("## Auto-fixes applied\n")
        for c in changes:
            print(f"- {c}")
        if not changes:
            print("No mechanical fixes needed.")
        print()
        pages = _pages(args.vault)  # reload from disk to report the fixed state

    findings = structural_findings(args.vault, today=today, pages=pages)
    report, count = _render(findings)

    print(f"{len(pages)} pages checked{' (after fixes)' if args.fix else ''}.")
    if count:
        print(report)
    else:
        print("\nNo structural problems found.")

    # Counts, not the findings themselves: the prose report above is the thing a
    # human reads, and duplicating every line into the log would double a report
    # that is already long. INFO, never WARNING — LocalLLMAgent's log_inspector
    # reports every WARNING it finds, and a weekly lint result is a dashboard
    # read, not a 7am phone push.
    for section, items in findings.items():
        if items:
            logger.info(f"{section}: {len(items)}")

    if args.deep:
        print("\n---\n\n## Judgment pass\n")
        context = report if count else "The structural pass found no problems."
        dispatch = query_dispatch(args.vault)
        # The judgment pass is one unit of work for retry-ceiling purposes: 60
        # iterations x 5 HTTP attempts is up to 300 retries against a provider
        # that may simply be down, and nothing else here bounds that.
        budget.start_source("judgment pass", MAX_DEEP_RETRIES)
        # logger= gives the run a tool-call timeline (`tool_call name(args) ->
        # result`), which is the shape LocalLLMAgent's dashboard renders.
        print(run_agent(
            system_prompt=rules_path.read_text(encoding="utf-8") + LINT_WRAPPER
            + "\n\nStructural findings already reported (do not repeat these):\n"
            + context,
            user_prompt="Audit the wiki and report your findings.",
            tools=QUERY_TOOL_SCHEMAS,
            dispatch=dispatch,
            max_iterations=60,
            logger=logger,
        ))

    logger.info(
        f"Wiki lint run complete — {len(pages)} pages checked, "
        f"{count} structural finding{'' if count == 1 else 's'}"
    )
    return count


def _lint_json(args) -> int:
    """The structural pass as one JSON object on stdout, for a UI to render.

    Nothing here logs. That is the point, not an oversight: this path is driven
    by a button, and setup_logger writes logs/wiki_lint.<vault>.log — the file
    LocalLLMAgent's dashboard parses for this job's run history (see its
    docs/external-tasks.md). A click that logged "Starting wiki lint run" would
    invent a scheduled run that never happened, and a click that crashed
    mid-write would leave a run its log_inspector reports as started-and-never-
    finished. So no logger, no launchd-log trim, and no failure push either —
    the caller gets the exception through a non-zero exit and reports it itself.

    --deep is refused rather than ignored: it is a multi-minute model
    conversation, which is not something a page load can wait on.

    --fix, by contrast, is honoured. Nothing above should be read as making
    this path read-only: --json --fix writes to the vault exactly as the prose
    path does, so a button wired to it is a button that edits pages.

    Exit code matches the prose path — 1 means findings exist, not failure.
    """
    if args.deep:
        print("error: --json cannot be combined with --deep (the judgment pass "
              "is a multi-minute model run)", file=sys.stderr)
        return 2

    vault = Path(args.vault)
    if not (vault / "wiki").is_dir():
        print(json.dumps({"error": f"no wiki/ directory in {vault}"}))
        return 2

    pages = _pages(args.vault)
    fixes = []
    if args.fix:
        fixes = apply_safe_fixes(args.vault, pages)
        pages = _pages(args.vault)  # reload so findings describe the fixed state

    findings = structural_findings(args.vault, pages=pages)
    print(json.dumps({
        "vault": str(vault),
        "pages": len(pages),
        "sections": findings,
        "fixes": fixes,
    }))
    return 1 if any(findings.values()) else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vault", required=True, help="Path to the Obsidian vault.")
    parser.add_argument(
        "--deep",
        action="store_true",
        help="Also run the model pass for contradictions, duplicate concepts, "
             "out-of-scope pages and outdated claims.",
    )
    parser.add_argument(
        "--fix",
        action="store_true",
        help="Apply the safe, mechanical fixes (strip self-links, de-link dead "
             "index entries) before reporting. Judgment calls are left alone.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the structural findings as JSON instead of prose, and log "
             "nothing. For a UI that renders its own report. Not valid with --deep.",
    )
    args = parser.parse_args()

    # Before setup_logger, deliberately — see _lint_json.
    if args.json:
        return _lint_json(args)

    vault_name = Path(args.vault).name
    logger = setup_logger(f"wiki_lint.{vault_name}")
    trim_launchd_log(logger)
    # setup_logger flushes each record to stdout, while print() is block-buffered
    # when launchd points stdout at a file — without this the prose report and the
    # log lines interleave out of order in the launchd log.
    sys.stdout.reconfigure(line_buffering=True)

    rules_path = Path(args.vault) / "RULES.md"
    if not rules_path.is_file():
        print(f"error: {rules_path} not found", file=sys.stderr)
        return 1

    logger.info(f"Starting wiki lint run for vault: {args.vault}")

    # Only --deep talks to a model, and only a model call can hang. The
    # structural pass is pure Python over local files and consults no budget, so
    # starting one for it would bound nothing.
    if args.deep:
        budget.start_run(DEEP_RUN_BUDGET_MINUTES * 60)

    job = f"wiki_lint[{vault_name}]"
    try:
        count = _lint(args, rules_path, logger)
    except budget.BudgetExceeded as e:
        logger.error(f"Wiki lint run abandoned: {e}")
        notify_failure(job, f"abandoned — {e}", logger)
        return 1
    except Exception as e:
        # A weekly unattended audit that dies leaves no trace anyone reads: the
        # report simply doesn't appear, which looks identical to a clean week.
        logger.exception(f"Wiki lint run failed: {e}")
        notify_failure(job, e, logger)
        return 1

    # A run that finds problems still RAN — the nonzero exit is for a human or a
    # CI check, and the completion marker is what the dashboard reads. Findings
    # are deliberately not an alert: a weekly audit result is something you read,
    # not something that should page you.
    return 1 if count else 0


if __name__ == "__main__":
    sys.exit(main())
