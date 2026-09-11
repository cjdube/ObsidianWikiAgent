#!/usr/bin/env python3
"""Score local models against each other on the `wiki_lint.py --deep` audit.

Run it:

    .venv/bin/python tools/compare_lint_models.py --check
    .venv/bin/python tools/compare_lint_models.py \\
        --models gemma4:26b-mlx,gemma4:31b-mlx,qwen3.8:27b-mlx --trials 3

Not a job. Nothing schedules this, and it never writes to a real vault: every
run builds a throwaway copy under a temp directory and prints where it put it.

WHY THIS EXISTS

The deep pass is the one job where the model decides whether the findings are
worth reading. The 2026-09-03 comparison in docs/agent-context.md scored two
providers on a seeded vault and then vanished — it was run by hand, the seeded
vault was never saved, and nothing in the repo could rebuild it. So the next
question about model choice started from zero.

The seed pack in tools/lint_defects/ is that vault, written down. This script
builds it, runs the audit, and scores the report against the answer key.

WHAT IT MEASURES, AND WHAT IT REFUSES TO

Recall is scored automatically: a planted defect counts as found when one
numbered finding names every page the defect spans. Requiring them in a single
item is the whole trick — a report that happens to mention two pages in two
unrelated findings has not found the defect that connects them.

Precision is NOT scored automatically, and the table says so in words. A
keyword rule can check that a report names the right page. It cannot tell a
true claim from a confident false one, and the local model's one measured
false finding on 2026-09-03 is exactly the shape that would slip through. Read
the saved reports. The count of false findings is a number a person produces.

TWO TIERS

Default (Tier A) builds the seed pack as its own ~35-page vault. Small enough
that a thorough auditor can plausibly reach every page, so the score measures
judgment rather than sampling luck. This is the discriminator.

--inject <vault> (Tier B) plants the same defects in a git archive copy of a
real vault. Recall there is dominated by whether search happens to surface a
seeded page, so it is an acceptance test for a model already chosen, not a way
to choose one.

SAMPLE SIZE

Three trials is the floor, not a target. These runs are not deterministic. The
verdict line says how many trials are behind each number.
"""

import argparse
import datetime
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SEED_DIR = REPO / "tools" / "lint_defects"

# wiki_lint.py prints this before the model's report. Everything above it is
# the structural pass, whose findings are also a numbered list — scoring those
# would credit a model for work Python did.
JUDGMENT_MARKER = "## Judgment pass"

# agent/loop.py marks a conversation that ran out of iterations with this.
INCOMPLETE = "[incomplete"

# setup_logger mirrors every record to stdout, so one subprocess capture holds
# the report, the tool-call timeline and the context-fill warnings together.
_TOOL_CALL = re.compile(r"\[INFO\] tool_call (\w+)\(")
_CONTEXT_FILL = re.compile(r"\((\d+)% of num_ctx=(\d+)\)")
# A finding is numbered, but the number is not always the first character.
# qwen3.8:27b-mlx writes every finding as "**7. Out-of-scope page — ...**",
# and an earlier version of this pattern scored two of its three trials at
# zero while they were in fact its best reports. Leading bullets, heading
# hashes and emphasis are formatting, so they are skipped rather than
# treated as the absence of a list.
_NUMBERED = re.compile(
    r"^\s{0,3}(?:[-*+]\s+)?(?:#{1,6}\s*)?(?:\*\*|__|\*|_)?(\d{1,3})[.)]\s",
    re.MULTILINE,
)

# The vault directory's basename becomes the structured log's name, so this
# also decides what lands in logs/. Fixed rather than timestamped: one
# identifiable file that rotates, instead of a new one per invocation.
VAULT_NAME = "lintcmp"


# --- building the seeded vault ---------------------------------------------


def load_seed(seed_dir: Path = SEED_DIR) -> dict:
    seed = json.loads((seed_dir / "seed.json").read_text(encoding="utf-8"))
    seed["dir"] = seed_dir
    return seed


def _summary_of(content: str) -> str:
    """The page's one-line Summary, which becomes its index description.

    Read off the page rather than repeated in seed.json: two copies of the same
    sentence drift, and the index is the copy nobody looks at.
    """
    for line in content.splitlines():
        if line.startswith("**Summary**:"):
            return line.split(":", 1)[1].strip()
    return ""


def _seed_pages(seed: dict) -> dict[str, str]:
    """{slug: content} for every page in the pack, defect and filler alike."""
    return {
        p.stem: p.read_text(encoding="utf-8")
        for p in sorted((seed["dir"] / "pages").glob("*.md"))
    }


def _index_block(seed: dict, pages: dict[str, str]) -> str:
    lines = [f"## {seed['index_section']}", ""]
    for slug in sorted(pages):
        lines.append(f"- [[{slug}]] {_summary_of(pages[slug])}")
    return "\n".join(lines) + "\n"


def _git_archive(src: Path, dest: Path) -> None:
    """A copy of the vault at HEAD, per AGENTS.md.

    git archive rather than cp: it takes the committed tree, so a copy cannot
    inherit whatever half-finished edit happens to be sitting in the working
    directory when the comparison is started.
    """
    dest.mkdir(parents=True, exist_ok=True)
    archive = subprocess.run(
        ["git", "-C", str(src), "archive", "HEAD"],
        check=True, stdout=subprocess.PIPE,
    )
    subprocess.run(
        ["tar", "-x", "-C", str(dest)], check=True, input=archive.stdout
    )


def build_vault(dest: Path, seed: dict, inject_from: Path = None) -> Path:
    """Write the seeded vault to `dest` and return it.

    Standalone, the pack is the whole vault and brings its own RULES.md. With
    --inject the host vault's RULES.md is kept, because scope is that vault's
    policy and overwriting it would audit a wiki against somebody else's rules.
    The caller is warned that the out-of-scope defects then depend on the host
    excluding the same subjects; see main().
    """
    pages = _seed_pages(seed)
    if inject_from:
        _git_archive(inject_from, dest)
        existing = {p.stem for p in (dest / "wiki").glob("*.md")}
        if clash := sorted(existing & set(pages)):
            raise SystemExit(
                f"the seed pack would overwrite {len(clash)} existing page(s) "
                f"in {inject_from}: {', '.join(clash)}. Rename them in "
                f"{seed['dir']}/pages/ and update seed.json."
            )
    else:
        (dest / "wiki").mkdir(parents=True, exist_ok=True)
        shutil.copyfile(seed["dir"] / "RULES.md", dest / "RULES.md")

    for slug, content in pages.items():
        (dest / "wiki" / f"{slug}.md").write_text(content, encoding="utf-8")

    raw_dir = dest / "raw" / seed["raw_subdir"]
    raw_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(seed["dir"] / seed["raw_source"], raw_dir / seed["raw_source"])

    index = dest / "wiki" / "index.md"
    head = index.read_text(encoding="utf-8") if index.exists() else "# Index\n"
    if f"## {seed['index_section']}" in head:
        raise SystemExit(
            f"the host index already has a '## {seed['index_section']}' "
            f"heading — two headings for one section is itself a lint finding. "
            f"Change index_section in {seed['dir']}/seed.json."
        )
    index.write_text(
        head.rstrip("\n") + "\n\n" + _index_block(seed, pages), encoding="utf-8"
    )
    return dest


# --- running one audit ------------------------------------------------------


def run_once(vault: Path, model: str, timeout: int) -> dict:
    """One `wiki_lint.py --vault <copy> --deep` run under one model.

    The model is passed as a real environment variable, which is all it takes:
    agent/__init__.py calls load_dotenv without override, so config/.env fills
    in what the environment has not already set. That is the same mechanism a
    plist EnvironmentVariables entry would use to ship the winner, so this
    measures the arrangement it is deciding about.
    """
    env = {**os.environ, "OLLAMA_MODEL": model}
    env.pop("LLM_PROVIDER", None)  # a stray export would silently score Gemini
    started = time.monotonic()
    proc = subprocess.run(
        [sys.executable, str(REPO / "wiki_lint.py"),
         "--vault", str(vault), "--deep"],
        cwd=REPO, env=env, timeout=timeout,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    return {
        "model": model,
        "seconds": time.monotonic() - started,
        "exit_code": proc.returncode,
        "stdout": proc.stdout,
    }


def structural_only(vault: Path) -> tuple[int, str]:
    """The structural pass alone, as (finding count, stdout). Exit code 1 means
    findings exist, not failure — see wiki_lint.main()."""
    proc = subprocess.run(
        [sys.executable, str(REPO / "wiki_lint.py"),
         "--vault", str(vault), "--json"],
        cwd=REPO, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    try:
        data = json.loads(proc.stdout)
    except json.JSONDecodeError:
        # --json prints one JSON object and nothing else, so anything else is a
        # traceback. Showing it beats a decode error pointing at column 1.
        raise SystemExit(
            f"the structural pass did not return JSON (exit {proc.returncode}):"
            f"\n{proc.stdout}"
        )
    if "error" in data:
        raise SystemExit(f"structural pass failed: {data['error']}")
    return sum(len(v) for v in data["sections"].values()), proc.stdout


def findings_in(raw: str) -> set[str]:
    """Every structural finding as '<section>: <text>'.

    A set, not a count, because --inject has to answer a different question:
    not "is this vault clean" but "did the pack make it worse". A host vault
    carries its own findings and it is not this tool's job to fix them.
    """
    data = json.loads(raw)
    return {f"{section}: {text}"
            for section, items in data["sections"].items() for text in items}


# --- scoring ----------------------------------------------------------------


# setup_logger mirrors the run's own records into the same stream as the
# report, so the tool-call timeline sits between the marker and the report and
# the "run complete" line lands *after* it. The timeline is harmless (it is
# preamble, dropped before the first finding) but the trailing line is not: it
# falls inside the last finding and is scored as if the model wrote it.
_LOG_LINE = re.compile(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3} \[", re.MULTILINE)


def judgment_text(stdout: str) -> str:
    """The model's report alone, with structural findings and log lines cut away."""
    _, sep, tail = stdout.partition(JUDGMENT_MARKER)
    if not sep:
        return ""
    return "\n".join(l for l in tail.splitlines() if not _LOG_LINE.match(l))


def numbered_items(report: str) -> list[str]:
    """The report split into its numbered findings.

    Preamble before the first number is dropped. A model that answers in prose
    with no numbered list therefore scores zero recall, which is the honest
    answer: RULES.md and the wrapper both ask for a numbered list naming
    specific pages, and a wall of prose has not delivered one.

    A bold or bulleted number is still a number. See _NUMBERED: reading
    "**7." as prose cost qwen3.8:27b-mlx two whole trials.
    """
    starts = [m.start() for m in _NUMBERED.finditer(report)]
    if not starts:
        return []
    bounds = starts + [len(report)]
    return [report[bounds[i]:bounds[i + 1]] for i in range(len(starts))]


def _letters(text: str) -> str:
    """Text reduced to its letters and digits.

    The same normalisation check_slug_typos uses, for the same reason: a model
    names a page in whatever spelling suits its sentence. 'vector-index-refresh',
    '[[vector-index-refresh]]', 'vector-index-refresh.md' and 'Vector Index
    Refresh' are one page, and only this comparison sees all four as one. A
    token-by-token match misses the last of them, which is the spelling a model
    writing prose actually reaches for.
    """
    return re.sub(r"[^a-z0-9]", "", text.lower().replace("&", "and"))


def score_report(report: str, defects: list[dict]) -> dict:
    """Which planted defects one report found.

    A defect scores only when a *single* numbered finding names every page it
    spans. Two pages named in two unrelated findings is not a report that
    connected them, and crediting it would flatter every model that talks a
    lot. For a single-page defect this reduces to naming that page.

    KNOWN OVER-CREDIT, and why it is not fixed here. A finding runs from its
    number to the next number, so anything the model writes after its *last*
    finding belongs to that finding. Models end with a wrap-up naming the pages
    they checked and cleared. On 2026-09-10 that handed qwen3.8:27b-mlx credit
    for a defect on `quantization-builds` from a sentence saying it found
    nothing there. Only the last finding, and only a single-page defect, can be
    inflated this way. It is left alone because every rule that cuts the last
    finding short — stop at a blank line, stop at a wrap-up phrase — also cuts
    real multi-paragraph findings, and losing a true finding is the worse
    error. Read the last finding of each report when the score is close.

    Known limit: models group. The 2026-09-10 control run answered with one
    numbered item per *category*, listing both duplicate pairs under item 2. So
    a numbered item is not always one finding, and a defect whose pages were
    named for two different reasons inside one grouped item would score. The
    alternative — splitting on the bullets inside an item — loses a genuine
    finding written as one bullet per page, which is the worse error. Left as
    it is, and the reports are saved so a reader can see which happened.
    """
    items = [_letters(i) for i in numbered_items(report)]
    found = [d["id"] for d in defects
             if any(all(_letters(p) in item for p in d["pages"])
                    for item in items)]
    return {
        "found": found,
        "recall": len(found),
        "total": len(defects),
        "items": len(items),
    }


def run_metrics(stdout: str) -> dict:
    """Tool calls, peak context fill and completion, read off the captured
    stdout. setup_logger mirrors the structured log there, so no log file has
    to be located or de-interleaved from a concurrent run."""
    fills = [(int(p), int(n)) for p, n in _CONTEXT_FILL.findall(stdout)]
    report = judgment_text(stdout)
    return {
        "tool_calls": len(_TOOL_CALL.findall(stdout)),
        "peak_pct": max((p for p, _ in fills), default=0),
        "num_ctx": fills[0][1] if fills else 0,
        "completed": bool(report.strip()) and INCOMPLETE not in report,
    }


# --- reporting --------------------------------------------------------------

_HEADER = (f"{'model':<20}{'trials':>7}{'recall':>18}{'done':>7}"
           f"{'calls':>8}{'ctx':>6}{'secs':>8}")


def _mean(xs):
    return sum(xs) / len(xs) if xs else 0


def report_table(rows: list[dict], out=print) -> None:
    by_model: dict[str, list[dict]] = {}
    for r in rows:
        by_model.setdefault(r["model"], []).append(r)

    out("")
    out(_HEADER)
    out("-" * len(_HEADER))
    for model, trials in by_model.items():
        total = trials[0]["score"]["total"]
        recalls = [t["score"]["recall"] for t in trials]
        out(
            f"{model:<20}{len(trials):>7}"
            f"{f'{_mean(recalls):.1f}/{total}':>18}"
            f"{sum(1 for t in trials if t['metrics']['completed']):>7}"
            f"{_mean([t['metrics']['tool_calls'] for t in trials]):>8.0f}"
            f"{max(t['metrics']['peak_pct'] for t in trials):>5.0f}%"
            f"{_mean([t['seconds'] for t in trials]):>8.0f}"
        )

    out("")
    for model, trials in by_model.items():
        per_trial = ", ".join(
            f"{sorted(t['score']['found'])}" for t in trials
        )
        out(f"{model}: {per_trial}")

    out("")
    out("recall  mean planted defects named, of the total. Auto-scored.")
    out("done    trials that returned a report rather than [incomplete].")
    out("calls   mean tool calls. The 2026-09-06 production run used 19 of 120.")
    out("ctx     highest context fill seen, across trials.")
    out("")
    out("FALSE FINDINGS ARE NOT SCORED HERE. Read the saved reports and count")
    out("them by hand. A model that names every planted defect and invents two")
    out("more has not won.")
    out("")
    out("Read the LAST finding of each report too. Text after it is scored as")
    out("part of it, so a wrap-up naming a clean page can add a false point.")
    n = min(len(t) for t in by_model.values()) if by_model else 0
    if n < 3:
        out("")
        out(f"Only {n} trial(s) per model — these runs are not deterministic. "
            f"Treat this as a signal, not a result.")


# --- entry point ------------------------------------------------------------


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--models", default="",
                    help="comma-separated Ollama tags, in `ollama list` "
                         "spelling (e.g. gemma4:26b-mlx,gemma4:31b-mlx)")
    ap.add_argument("--trials", type=int, default=3,
                    help="runs per model (default 3)")
    ap.add_argument("--inject", metavar="VAULT",
                    help="plant the defects in a git archive copy of this "
                         "vault (Tier B) instead of building the pack alone")
    ap.add_argument("--workdir", help="where to build and save reports "
                                      "(default: a new temp directory)")
    ap.add_argument("--seed-dir", default=str(SEED_DIR),
                    help=f"the seed pack to use (default {SEED_DIR})")
    ap.add_argument("--timeout", type=int, default=2400,
                    help="seconds to allow one run (default 2400; "
                         "wiki_lint's own budget is 30 minutes)")
    ap.add_argument("--check", action="store_true",
                    help="build the vault, assert the structural pass is "
                         "clean, and stop. No model is called.")
    args = ap.parse_args(argv)

    seed = load_seed(Path(args.seed_dir))
    work = Path(args.workdir) if args.workdir else Path(
        tempfile.mkdtemp(prefix="lintcmp-")
    )
    vault = work / VAULT_NAME
    reports = work / "reports"
    reports.mkdir(parents=True, exist_ok=True)

    # A --workdir reused between invocations would otherwise append the seed's
    # index section to an index that already has one, and fail the build for a
    # reason that has nothing to do with the pack. Only ever a directory this
    # script built: same name, and it must already look like one of its vaults.
    if vault.exists():
        if not (vault / "wiki").is_dir():
            raise SystemExit(
                f"{vault} exists but is not a vault this script built. "
                f"Point --workdir somewhere else."
            )
        shutil.rmtree(vault)

    inject = Path(args.inject).expanduser() if args.inject else None
    build_vault(vault, seed, inject)
    print(f"vault:   {vault}")
    print(f"reports: {reports}")

    # A real vault has findings of its own. On 2026-09-10 an --inject run into
    # the learnings vault stopped on n8n.md using its slug as a title — a true
    # finding, present before the pack and nothing to do with it. So the gate
    # is what the pack ADDED, which is also the comparison AGENTS.md asks for:
    # newly introduced findings, not an absolute count.
    baseline: set[str] = set()
    if inject:
        base = vault.parent / f"{VAULT_NAME}-baseline"
        if base.exists():
            shutil.rmtree(base)
        _git_archive(inject, base)
        _, base_raw = structural_only(base)
        baseline = findings_in(base_raw)
        if baseline:
            print(f"host vault carries {len(baseline)} structural finding(s) of "
                  f"its own; those are ignored, not fixed.")

    _, raw = structural_only(vault)
    added = sorted(findings_in(raw) - baseline)
    if added:
        for line in added:
            print(f"  {line}")
        raise SystemExit(
            f"\nthe pack ADDED {len(added)} STRUCTURAL finding(s). The seed is "
            f"leaking into the pass Python already does, so any judgment score "
            f"from it would be measuring the wrong thing. Fix the pack first — "
            f"{seed['dir']}/README.md lists what each check requires."
        )
    print(f"structural pass adds nothing over {len(_seed_pages(seed))} seeded pages.")

    if inject:
        print(
            "\nNOTE: --inject keeps the host vault's RULES.md. The three "
            "out-of-scope defects only count if that file also excludes social "
            "events, volunteer days and benefits administration. Read its "
            "Scope section before trusting those three."
        )
    if args.check:
        return 0

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    if not models:
        raise SystemExit("--models is required unless you passed --check")

    rows = []
    for model in models:
        for trial in range(1, args.trials + 1):
            print(f"\n=== {model} trial {trial}/{args.trials} ===", flush=True)
            result = run_once(vault, model, args.timeout)
            report = judgment_text(result["stdout"])
            result["score"] = score_report(report, seed["defects"])
            result["metrics"] = run_metrics(result["stdout"])

            name = f"{model.replace(':', '_').replace('/', '_')}.{trial}.txt"
            (reports / name).write_text(result["stdout"], encoding="utf-8")
            print(f"recall {result['score']['recall']}/"
                  f"{result['score']['total']} "
                  f"{sorted(result['score']['found'])} — "
                  f"{result['metrics']['tool_calls']} calls, "
                  f"{result['seconds']:.0f}s -> {name}")
            rows.append(result)

    report_table(rows)
    (work / "results.json").write_text(
        json.dumps(
            {"when": datetime.datetime.now().isoformat(timespec="seconds"),
             "vault": str(vault),
             "injected_into": str(inject) if inject else None,
             "trials": [{k: v for k, v in r.items() if k != "stdout"}
                        for r in rows]},
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nresults: {work / 'results.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
