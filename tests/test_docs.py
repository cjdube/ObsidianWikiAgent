"""Guards the prose that makes a promise the code has to keep.

Documentation drifts silently: nothing at runtime reads SECURITY.md, so nothing
ever contradicts it. That is how the privacy paragraph came to say no plist
template sets LLM_PROVIDER while launchd/template-lint.plist.txt shipped with
LLM_PROVIDER=gemini in it — a tracked file, installable with one argument, that
sends a vault's pages to a third party every Sunday.

That template stopped setting a provider on 2026-09-10, so the claim and the
tree agree again. The drift can now run either way, and both directions are
tested: a template that picks a cloud provider must be named in the docs, and
the docs must not keep saying one does after it stopped.

Only claims worth a test live here: the ones a reader would act on before they
could check them.
"""

import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
LAUNCHD = ROOT / "launchd"

_PROVIDER = re.compile(
    r"<key>LLM_PROVIDER</key>\s*<string>([^<]+)</string>", re.MULTILINE
)
_XML_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)


def _templates() -> dict[str, str]:
    return {p.name: p.read_text(encoding="utf-8") for p in LAUNCHD.glob("template*.txt")}


def _provider_setters() -> dict[str, str]:
    """Template filename -> the LLM_PROVIDER it ships, for the ones that set one.

    Comments are stripped first, for the same reason install.sh reads a plist
    with plutil instead of grep: a commented-out setting is not a setting. The
    lint template documents the two lines to add inside its own comment, and
    reading that as a live gemini default would make every check below fire on
    a job that runs locally. A template is not a valid plist until install.sh
    substitutes its placeholders, so plutil is not available here — stripping
    comments is the same rule enforced with the tool that does work on a .txt.
    """
    return {
        name: m.group(1)
        for name, text in _templates().items()
        if (m := _PROVIDER.search(_XML_COMMENT.sub("", text)))
    }


def test_no_plist_template_ships_a_cloud_provider():
    """The live guard, and the reason the paragraphs below can say what they
    say. A template is a tracked file that install.sh turns into a scheduled
    job with one argument, so a cloud provider set in one routes somebody
    else's vault off their machine before they have read anything. Turning it
    on is a decision for the person installing the job; the template shows them
    which two lines to add.

    template-lint.plist.txt set gemini from the day it was written until
    2026-09-10. Nothing stopped it but a person noticing."""
    setters = _provider_setters()
    cloud = {name: p for name, p in setters.items() if p != "ollama"}
    assert not cloud, (
        f"{sorted(cloud)} set a cloud LLM_PROVIDER. A template must leave the "
        f"key out (inheriting the local default) or set 'ollama'."
    )


@pytest.mark.parametrize("doc", ("SECURITY.md", "README.md"))
def test_privacy_docs_name_every_template_that_sets_a_cloud_provider(doc):
    """A template that picks a provider is a privacy default shipped in the
    repo. Both files that make the privacy promise must name that file, so the
    reader knows which job to look at rather than trusting a blanket 'nothing
    here turns it on'.

    README is checked alongside SECURITY.md because it is the file a reader
    meets first, and it carried the stale claim for longer: SECURITY.md was
    corrected while README's opening paragraph still said no template sets it,
    four hundred lines above its own paragraph saying one does.

    No template sets one today, so this passes over an empty set. It is kept
    armed rather than deleted: the day somebody adds a provider back, this is
    what makes them write the paragraph too. The check that keeps the tree
    honest meanwhile is test_no_plist_template_ships_a_cloud_provider."""
    text = (ROOT / doc).read_text(encoding="utf-8")

    for name, provider in _provider_setters().items():
        assert name in text, (
            f"{name} sets LLM_PROVIDER={provider} but {doc} never names it"
        )


@pytest.mark.parametrize("doc", ("SECURITY.md", "README.md"))
def test_privacy_docs_do_not_still_say_a_template_picks_gemini(doc):
    """The mirror of the check above, and the one with teeth right now. Both
    files spent months claiming the templates were provider-free while the lint
    template set gemini; from 2026-09-10 the same sentences are true, and the
    stale claim runs the other way — prose telling a reader that installing
    `lint` ships their vault to Google, about a job that runs locally.

    A reader who believes that does not install the audit at all. Listed as
    literal strings because that is what actually went stale last time: a
    paragraph can be rewritten correctly and still leave one sentence behind."""
    text = " ".join((ROOT / doc).read_text(encoding="utf-8").split())
    for stale in (
        "ships `LLM_PROVIDER=gemini`",
        "puts `LLM_PROVIDER=gemini` in the weekly",
        "the one template that ships",
        "it is the one job whose template",
        "the one place vault content leaves the machine",
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


def _fake_repo(tmp_path):
    """A throwaway tree install.sh will accept as its own repo.

    install.sh derives ROOT from its own location, so copying it and the
    templates somewhere else is enough to drive it against edited templates
    without writing to the real launchd/ directory. It only checks that the
    interpreter is executable, never runs it, so a stub is enough."""
    root = tmp_path / "repo"
    (root / "launchd").mkdir(parents=True)
    (root / ".venv" / "bin").mkdir(parents=True)
    python = root / ".venv" / "bin" / "python"
    python.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    python.chmod(0o755)
    shutil.copy2(LAUNCHD / "install.sh", root / "launchd" / "install.sh")
    for name in _templates():
        shutil.copy2(LAUNCHD / name, root / "launchd" / name)
    return root


def test_install_sh_warns_out_loud_when_a_template_sets_a_cloud_provider(tmp_path):
    """The privacy opt-in has a second door. On 2026-09-03 this machine's lint
    plist was hand-edited back to the local model, in ~/Library/LaunchAgents
    only — but install.sh regenerates that file from template-lint.plist.txt,
    which set LLM_PROVIDER=gemini. So a later install would have quietly
    restored the cloud provider. Nothing in the output said so.

    The warning has to name the provider, because 'check your plist' is the
    instruction that already failed once.

    No shipped template sets one now, so the cloud default is planted here
    rather than borrowed from the tree. That is the case worth testing anyway:
    the warning exists for whoever edits a template to opt in, and a guard that
    only fired while the repo itself shipped gemini would have gone quiet on
    2026-09-10 without anyone noticing."""
    if shutil.which("plutil") is None:
        pytest.skip("plutil is macOS-only, and so is launchd")

    root = _fake_repo(tmp_path)
    template = root / "launchd" / "template-lint.plist.txt"
    text = template.read_text(encoding="utf-8")
    marker = "        <key>WIKI_LAUNCHD_LOG</key>"
    assert marker in text, "template-lint.plist.txt no longer has the anchor line"
    template.write_text(
        text.replace(
            marker,
            "        <key>LLM_PROVIDER</key>\n"
            "        <string>gemini</string>\n" + marker,
        ),
        encoding="utf-8",
    )

    vault = tmp_path / "some-vault"
    vault.mkdir()
    run = subprocess.run(
        [str(root / "launchd" / "install.sh"), "--dry-run", str(vault),
         "ingest", "lint", "snapshot"],
        capture_output=True, text=True, check=True,
    )
    warnings = [ln for ln in run.stdout.splitlines() if "PRIVACY" in ln]

    assert any("'gemini'" in ln for ln in warnings), (
        "a lint template setting LLM_PROVIDER=gemini drew no warning naming it. "
        f"Output was:\n{run.stdout}"
    )
    assert not any("-ingest" in ln or "-snapshot" in ln for ln in warnings), (
        f"only the lint job set a provider, but another job was warned about:\n"
        f"{run.stdout}"
    )


def test_install_sh_is_quiet_about_the_templates_as_they_ship(tmp_path):
    """The other half: the warning must not cry wolf. Every template now
    inherits the local default, so a plain install has nothing to announce —
    and the lint template's own comment shows the two gemini lines, which a
    grep would read as a setting."""
    if shutil.which("plutil") is None:
        pytest.skip("plutil is macOS-only, and so is launchd")

    root = _fake_repo(tmp_path)
    vault = tmp_path / "some-vault"
    vault.mkdir()
    run = subprocess.run(
        [str(root / "launchd" / "install.sh"), "--dry-run", str(vault),
         "ingest", "lint", "snapshot"],
        capture_output=True, text=True, check=True,
    )
    assert "PRIVACY" not in run.stdout, (
        f"no shipped template sets a provider, but install.sh warned:\n{run.stdout}"
    )


def test_install_sh_reads_the_provider_with_a_parser_not_a_grep(tmp_path):
    """A commented-out setting is not a setting. This machine's local lint plist
    kept the two LLM_PROVIDER lines inside an XML comment for a week, so a grep
    would have reported a cloud provider on a job that ran locally — and a
    privacy warning that cries wolf gets ignored the one time it is real."""
    if shutil.which("plutil") is None:
        pytest.skip("plutil is macOS-only, and so is launchd")

    plist = tmp_path / "commented.plist"
    plist.write_text(
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" '
        '"http://www.apple.com/DTDs/PropertyList-1.0.dtd">\n'
        '<plist version="1.0"><dict>\n'
        "  <key>EnvironmentVariables</key>\n"
        "  <dict>\n"
        "    <!-- <key>LLM_PROVIDER</key><string>gemini</string> -->\n"
        "    <key>WIKI_LAUNCHD_LOG</key><string>/tmp/x.log</string>\n"
        "  </dict>\n"
        "</dict></plist>\n",
        encoding="utf-8",
    )

    assert "gemini" in plist.read_text(encoding="utf-8")  # a grep would fire
    extracted = subprocess.run(
        ["plutil", "-extract", "EnvironmentVariables.LLM_PROVIDER", "raw", "-o", "-",
         str(plist)],
        capture_output=True, text=True,
    )
    assert extracted.returncode != 0, "plutil must not see a commented-out key"
