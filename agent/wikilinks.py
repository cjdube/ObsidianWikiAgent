"""One owner for Obsidian's [[wiki-link]] syntax.

Four regexes across two files used to parse this, and only one of them carried
the hardening below — which is how `wiki_lint --fix` came to write links back to
disk using the pattern the detector had already been fixed to stop using. The
two disagreed about what a link is, so `--fix` could rewrite a link inside a
code span that the linter never reported, and could silently fail to strip one
that it did.

Every caller now shares this pattern and this idea of a target. A check and the
fix that repairs it cannot drift apart again without changing this file.
"""

import re
from typing import Callable

# A link target never contains a newline or a backtick. Both exclusions come
# from one observed failure: a page whose body reads
# "Unclosed `[[` brackets can cause the linter to swallow subsequent lines"
# opened a match at the `[[` inside the backticks, ran past the real
# [[obsidian-wiki-agent]] link that followed, and closed on its ]] — reporting a
# broken link named "` brackets can cause the [[obsidian-wiki-agent" while the
# genuine link went uncounted, which also makes its target look like an orphan.
# Excluding backticks stops a code span's [[ from opening a match; excluding
# newlines keeps one unclosed [[ from consuming the rest of the file.
LINK_RE = re.compile(r"\[\[([^\]\n`]+)\]\]")

# LINK_RE alone still matches a whole link inside a code span, which Obsidian
# treats as text. Obsidian's own link list disagreed with this module on exactly
# that on 2026-11-04, on `[[claude-code]]`. Matching the span first consumes it,
# so a link inside it is never seen: group 1 is the span's backticks, group 2 a
# real link's body.
# shortcut: a span must close on its own line and fenced blocks are not skipped;
# upgrade if Obsidian's link list disagrees again (no wiki page needs it today).
_SPAN_OR_LINK_RE = re.compile(rf"(?<!`)(`+)(?!`)[^\n]*?(?<!`)\1(?!`)|{LINK_RE.pattern}")


def _sub_links(repl: Callable[[str], str], content: str) -> str:
    """Replace each link outside a code span with repl(link body)."""
    return _SPAN_OR_LINK_RE.sub(lambda m: m[0] if m[1] else repl(m[2]), content)


def link_target(inner: str) -> str:
    """The bare page stem a link body names — [[foo]], [[foo.md]], [[foo|alias]]
    and [[foo#section]] all mean foo."""
    name = inner.split("|", 1)[0].split("#", 1)[0].strip()
    return name.removesuffix(".md")


def display_text(inner: str) -> str:
    """What a link body reads as with the brackets taken off: the alias if one
    was given, otherwise the target as written."""
    return inner.split("|", 1)[-1].strip()


def linked_page_names(content: str) -> set[str]:
    """Page names linked from `content`, as bare stems."""
    return {
        name
        for m in _SPAN_OR_LINK_RE.finditer(content)
        if m[2] and (name := link_target(m[2]))
    }


def flatten_links(
    content: str, should_flatten: Callable[[str], bool]
) -> tuple[str, int]:
    """Replace every link whose target `should_flatten` accepts with its plain
    display text, returning the new content and the count replaced. Links it
    rejects are left exactly as written.

    The predicate is the only thing that varies between callers — a linter
    flattening a page's links to itself and an index dropping links to pages
    that no longer exist are the same rewrite over the same syntax, asking a
    different question about the target.
    """
    count = 0

    def repl(inner: str) -> str:
        nonlocal count
        if not should_flatten(link_target(inner)):
            return f"[[{inner}]]"
        count += 1
        return display_text(inner)

    return _sub_links(repl, content), count


def delink_broken(content: str, valid: set[str]) -> tuple[str, int]:
    """Flatten any [[link]] whose target is not a real page.

    The local model authors index links as free text and mistypes a few each
    run (a dropped letter, a doubled hyphen, a since-deleted page), and nothing
    else vets them before they reach disk. A [[target]] that resolves to no
    page is a dead link — clicking it in Obsidian only offers to create an
    empty page — so it is flattened to text rather than left to rot in the
    table of contents. Typos are dropped, never guess-corrected: repointing
    [[cla-...]] at claude-... risks linking the wrong page."""
    return flatten_links(content, lambda target: target not in valid)


def strip_links_to(content: str, slug: str) -> tuple[str, int]:
    """Flatten any link `content` makes to `slug` itself. A page linking to
    itself is never meaningful, so it becomes its plain display text."""
    return flatten_links(content, lambda target: target == slug)


def lowercase_targets(content: str) -> tuple[str, int]:
    """Lower-case the page stem of every [[link]], returning the new content
    and the count rewritten. The alias and the '#heading' anchor are left
    exactly as written — only the part that names a page is touched.

    RULES.md has always asked for lowercase-hyphenated slugs, in both
    directions ("Wiki-links point at real page slugs, lowercase with hyphens"
    and "Keep page names lowercase with hyphens"). The model reads a source
    called raw/daily-chrome/Daily-Chrome-2026-08-22.md, and writes
    [[Daily-Chrome-2026-08-22]] into the page's Related pages list anyway: it
    copies the filename it was handed rather than deriving the slug. Thirteen
    such links were on the vault on 2026-08-24, across two source families.

    They are worse than untidy. On a case-insensitive filesystem Obsidian will
    open one, so nothing looks wrong — but the name it matches is ambiguous
    between wiki/daily-chrome-2026-08-22.md and the raw source of nearly the
    same name, and wiki_lint's check_links compares exactly, so all thirteen
    reported as broken links to pages that in fact existed. A link that
    resolves for a human and not for the graph is the failure this vault can
    least afford.

    Asking the prompt again was not an option: the rule is already there twice
    and was already ignored, which is the same ground update_index and
    write_wiki_page stand on — a guarantee belongs in code. Unconditional
    rather than only-when-the-target-exists, because during an ingest the page
    a link points at is frequently written later in the same run; a check
    against what is on disk now would pass over exactly the new links this
    exists to fix."""
    count = 0

    def repl(inner: str) -> str:
        nonlocal count
        head, bar, alias = inner.partition("|")
        stem, hash_, anchor = head.partition("#")
        if stem == stem.lower():
            return f"[[{inner}]]"
        count += 1
        return f"[[{stem.lower()}{hash_}{anchor}{bar}{alias}]]"

    return _sub_links(repl, content), count
