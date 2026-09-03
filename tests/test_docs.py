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

ROOT = Path(__file__).resolve().parent.parent
LAUNCHD = ROOT / "launchd"

_PROVIDER = re.compile(
    r"<key>LLM_PROVIDER</key>\s*<string>([^<]+)</string>", re.MULTILINE
)


def _templates() -> dict[str, str]:
    return {p.name: p.read_text(encoding="utf-8") for p in LAUNCHD.glob("template*.txt")}


def test_security_md_names_every_template_that_sets_a_cloud_provider():
    """A template that picks a provider is a privacy default shipped in the
    repo. SECURITY.md must name that file, so the reader knows which job to
    look at rather than trusting a blanket 'nothing here turns it on'."""
    security = (ROOT / "SECURITY.md").read_text(encoding="utf-8")

    setters = {
        name: m.group(1)
        for name, text in _templates().items()
        if (m := _PROVIDER.search(text))
    }
    assert setters, "expected at least the lint template to set a provider"

    for name, provider in setters.items():
        assert name in security, (
            f"{name} sets LLM_PROVIDER={provider} but SECURITY.md never names it"
        )


def test_security_md_does_not_claim_the_templates_are_provider_free():
    """The exact wording that went stale. Kept as its own check because the
    claim above can be satisfied while a leftover sentence still denies it."""
    security = (ROOT / "SECURITY.md").read_text(encoding="utf-8")
    for stale in ("neither plist template sets it", "Nothing in this repo turns it on"):
        assert stale not in security
