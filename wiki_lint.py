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
  Asking a model to enumerate 81 pages correctly is what produced the orphaned
  index in the first place (see update_index in agent/wiki_tools.py), so these
  run as code: instant, free, and they cannot miss one.

  --deep adds a model pass for the checks code cannot do: contradictions
  between pages, two pages covering one concept under different names, pages
  whose subject is out of scope, and claims a newer source has overtaken. The
  model receives the structural findings as context so it does not re-derive
  them.

Usage:
    python wiki_lint.py --vault ~/Documents/llm-wiki-learnings
    python wiki_lint.py --vault ~/Documents/llm-wiki-learnings --deep
    python wiki_lint.py --vault ~/Documents/llm-wiki-learnings --fix
"""

import argparse
import datetime
import functools
import re
import sys
from pathlib import Path

from agent.loop import run_agent
from agent.wiki_tools import (
    QUERY_TOOL_SCHEMAS,
    _delink_broken,
    _linked_page_names,
    list_raw_files,
    list_wiki_pages,
    read_index,
    read_wiki_page,
)

RESERVED = ("index.md", "log.md")

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


def check_links(pages: dict[str, str]) -> list[str]:
    """Wiki-links pointing at pages that do not exist, and self-links."""
    findings = []
    for slug, content in sorted(pages.items()):
        for target in sorted(_linked_page_names(content)):
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
        for target in _linked_page_names(content):
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
    linked = _linked_page_names(read_index(vault_path)["content"])
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


def check_format(vault_path: str, pages: dict[str, str], today: datetime.date) -> list[str]:
    """The page format RULES.md requires — the defect classes a weaker model
    reliably produces: slug-as-title, placeholder and future dates, and
    citations to sources that do not exist."""
    raw = set(list_raw_files(vault_path)["files"])
    findings = []
    for slug, content in sorted(pages.items()):
        title = next((l[2:].strip() for l in content.splitlines() if l.startswith("# ")), None)
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
            if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", updated):
                findings.append(
                    f"{slug}.md has a non-ISO 'Last updated' ({updated!r}) — "
                    f"use a plain date like {today.isoformat()}."
                )
            elif datetime.date.fromisoformat(updated) > today:
                findings.append(
                    f"{slug}.md is dated {updated}, in the future — "
                    f"the model invented a date."
                )

        sources = _field(content, "Sources")
        if sources:
            for src in (s.strip().strip("[]`") for s in sources.split(",")):
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


def check_duplicate_titles(pages: dict[str, str]) -> list[str]:
    """Two pages with the same title are the same page. Concepts duplicated
    under *different* titles need judgment and are left to the --deep pass."""
    by_title: dict[str, list[str]] = {}
    for slug, content in pages.items():
        title = next((l[2:].strip() for l in content.splitlines() if l.startswith("# ")), None)
        if title:
            by_title.setdefault(title.lower(), []).append(slug)
    return [
        f"{' and '.join(f'{s}.md' for s in sorted(slugs))} share the title "
        f"'{title}' — merge them and redirect inbound links."
        for title, slugs in sorted(by_title.items())
        if len(slugs) > 1
    ]


def structural_findings(
    vault_path: str,
    today: datetime.date = None,
    pages: dict[str, str] = None,
) -> dict[str, list[str]]:
    today = today or datetime.date.today()
    pages = _pages(vault_path) if pages is None else pages
    return {
        "Broken and self links": check_links(pages),
        "Orphan pages": check_orphans(pages),
        "Index integrity": check_index(vault_path, pages),
        "Page format": check_format(vault_path, pages, today),
        "Duplicate titles": check_duplicate_titles(pages),
    }


def _strip_self_links(content: str, slug: str) -> tuple[str, int]:
    """De-link any link a page makes to itself, returning the cleaned content
    and the count removed. A page linking to itself is never meaningful, so the
    link is flattened to its plain display text (the alias if one was given)."""
    count = 0

    def repl(m: re.Match) -> str:
        nonlocal count
        target = m.group(1).split("|", 1)[0].split("#", 1)[0].strip()
        if target.endswith(".md"):
            target = target[:-3]
        if target != slug:
            return m.group(0)
        count += 1
        return m.group(1).split("|", 1)[-1].strip()

    return re.sub(r"\[\[([^\]]+)\]\]", repl, content), count


def apply_safe_fixes(vault_path: str, pages: dict[str, str]) -> list[str]:
    """Apply only the provably-safe, mechanical fixes and return a log of what
    changed. This is the one place wiki_lint writes to the vault, gated behind
    --fix. Every judgment call — orphans, duplicate concepts, broken body links
    (create-vs-delink), bad dates, invented citations — is deliberately left
    for a human. `pages` is updated in place to reflect the writes.

    Two fixes qualify as safe:
      1. Self-links — a page linking to itself is never meaningful.
      2. Dead index links — a table-of-contents entry pointing at no real page
         (reusing the same de-linking the ingest guard applies)."""
    wiki_dir = Path(vault_path) / "wiki"
    changes: list[str] = []

    for slug in sorted(pages):
        cleaned, n = _strip_self_links(pages[slug], slug)
        if n:
            (wiki_dir / f"{slug}.md").write_text(cleaned, encoding="utf-8")
            pages[slug] = cleaned
            changes.append(f"{slug}.md: removed {n} self-link{'s' if n > 1 else ''}")

    index_path = wiki_dir / "index.md"
    if index_path.is_file():
        cleaned, n = _delink_broken(index_path.read_text(encoding="utf-8"), set(pages))
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
    args = parser.parse_args()

    rules_path = Path(args.vault) / "RULES.md"
    if not rules_path.is_file():
        print(f"error: {rules_path} not found", file=sys.stderr)
        return 1

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

    if args.deep:
        print("\n---\n\n## Judgment pass\n")
        context = report if count else "The structural pass found no problems."
        dispatch = {
            "list_wiki_pages": functools.partial(list_wiki_pages, args.vault),
            "read_wiki_page": functools.partial(read_wiki_page, args.vault),
            "read_index": functools.partial(read_index, args.vault),
        }
        print(run_agent(
            system_prompt=rules_path.read_text(encoding="utf-8") + LINT_WRAPPER
            + "\n\nStructural findings already reported (do not repeat these):\n"
            + context,
            user_prompt="Audit the wiki and report your findings.",
            tools=QUERY_TOOL_SCHEMAS,
            dispatch=dispatch,
            max_iterations=60,
        ))

    return 1 if count else 0


if __name__ == "__main__":
    sys.exit(main())
