"""Guards the prose that makes a promise the code has to keep.

Documentation drifts silently: nothing at runtime reads SECURITY.md, so nothing
ever contradicts it. That is how the privacy paragraph came to say no plist
template sets LLM_PROVIDER while launchd/template-lint.plist.txt shipped with
LLM_PROVIDER=gemini in it — a tracked file, installable with one argument, that
sends a vault's pages to a third party every Sunday.

Only claims worth a test live here: the ones a reader would act on before they
could check them.
"""

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
LAUNCHD = ROOT / "launchd"

_PROVIDER = re.compile(
    r"<key>LLM_PROVIDER</key>\s*<string>([^<]+)</string>", re.MULTILINE
)


def _templates() -> dict[str, str]:
    return {p.name: p.read_text(encoding="utf-8") for p in LAUNCHD.glob("template*.txt")}


def _provider_setters() -> dict[str, str]:
    """Template filename -> the LLM_PROVIDER it ships, for the ones that set one."""
    return {
        name: m.group(1)
        for name, text in _templates().items()
        if (m := _PROVIDER.search(text))
    }


@pytest.mark.parametrize("doc", ("SECURITY.md", "README.md"))
def test_privacy_docs_name_every_template_that_sets_a_cloud_provider(doc):
    """A template that picks a provider is a privacy default shipped in the
    repo. Both files that make the privacy promise must name that file, so the
    reader knows which job to look at rather than trusting a blanket 'nothing
    here turns it on'.

    README is checked alongside SECURITY.md because it is the file a reader
    meets first, and it carried the stale claim for longer: SECURITY.md was
    corrected while README's opening paragraph still said no template sets it,
    four hundred lines above its own paragraph saying one does."""
    text = (ROOT / doc).read_text(encoding="utf-8")

    setters = _provider_setters()
    assert setters, "expected at least the lint template to set a provider"

    for name, provider in setters.items():
        assert name in text, (
            f"{name} sets LLM_PROVIDER={provider} but {doc} never names it"
        )


@pytest.mark.parametrize("doc", ("SECURITY.md", "README.md"))
def test_privacy_docs_do_not_claim_the_templates_are_provider_free(doc):
    """The exact wordings that went stale. Kept as its own check because the
    claim above can be satisfied while a leftover sentence still denies it —
    which is exactly what README did."""
    text = " ".join((ROOT / doc).read_text(encoding="utf-8").split())
    for stale in (
        "neither plist template sets it",
        "Nothing in this repo turns it on",
        "no plist template in `launchd/` sets it",
    ):
        assert stale not in text, f"{doc} still says: {stale!r}"


_NUMBER_WORDS = {
    6: "Six", 7: "Seven", 8: "Eight", 9: "Nine", 10: "Ten", 11: "Eleven",
}

_INGEST_SCHEMA_LISTS = (
    "PLAN_TOOL_SCHEMAS",
    "CREATE_PAGE_TOOL_SCHEMAS",
    "UPDATE_PAGE_TOOL_SCHEMAS",
    "LOG_TOOL_SCHEMAS",
)


def _advertised_ingest_tools() -> set[str]:
    from agent import wiki_tools

    names: set[str] = set()
    for list_name in _INGEST_SCHEMA_LISTS:
        names |= {
            schema["function"]["name"] for schema in getattr(wiki_tools, list_name)
        }
    return names


@pytest.mark.parametrize("doc", ("SECURITY.md", "README.md"))
def test_privacy_docs_count_the_advertised_ingest_tools_correctly(doc):
    """Both files tell the reader how many tools the model can reach, and that
    number is the only part of the paragraph a reader can check. It was
    'Nine' against eight advertised tools, which is how you find out the list
    was never re-counted against the code.

    Spelled out rather than matched loosely on purpose: a tool added to any of
    the four schema lists must fail this until the prose is re-counted."""
    count = len(_advertised_ingest_tools())
    assert count in _NUMBER_WORDS, f"add {count} to _NUMBER_WORDS"
    word = _NUMBER_WORDS[count]

    # Both files hard-wrap their prose, so the claim routinely straddles a line
    # break ("Eight distinct\ntools"). Match on the unwrapped text.
    text = " ".join((ROOT / doc).read_text(encoding="utf-8").split())
    assert f"{word} distinct tools" in text or f"{word.lower()} distinct tools" in text, (
        f"{count} tools are advertised across {', '.join(_INGEST_SCHEMA_LISTS)}, "
        f"but {doc} does not say '{word} distinct tools'"
    )


def test_read_index_is_dispatchable_but_never_advertised():
    """The docs call read_index the one callable-but-unadvertised tool, so it
    is the tool after the advertised ones — a ninth, not a tenth. That only
    holds while it stays out of every schema list."""
    assert "read_index" not in _advertised_ingest_tools()

    from agent import wiki_tools

    assert "read_index" not in {
        schema["function"]["name"] for schema in wiki_tools.QUERY_TOOL_SCHEMAS
    }


def test_readme_agrees_with_the_lint_template_existing():
    """README used to send a reader to copy launchd/template.plist.txt for the
    audit job, in a repo that has a template-lint.plist.txt and an install.sh
    that picks it."""
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert (LAUNCHD / "template-lint.plist.txt").is_file()
    assert "there is no lint template" not in readme
