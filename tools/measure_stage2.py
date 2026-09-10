#!/usr/bin/env python3
"""Report how often stage 2 gives up on a page, from the structured ingest log.

Run it:

    .venv/bin/python tools/measure_stage2.py

Not a job. Nothing schedules this and it writes nothing — it reads the logs the
ingest already produces and prints a table. Kept in the repo because the answer
it gives has to be recomputed against the *current* log, and rebuilding the
parser from scratch each time is how a measurement quietly changes definition
between two runs that are supposed to be comparable.

WHAT IT COUNTS, AND WHY THAT ONE THING

Stage 2 opens one conversation per page and gives it MAX_EXECUTE_ITERATIONS
turns. When a conversation spends all of them without finishing, the page is
saved with an "[incomplete: hit max_iterations=12 ...]" marker. That marker is
the metric. It means the agent stopped mid-thought, and it is the only outcome
here a reader of the vault can actually feel.

The cause was a wording bug, fixed in 26046d3 (2026-09-07). edit_wiki_page
answers "unchanged" when the line is already on the page. The model read that
as a failure and re-sent the byte-identical call — on 2026-08-29 the 'tailscale'
page took five identical calls and then ran out of turns, six seconds after the
write had already succeeded. The fix rewrote that reply to say the page is
finished and not to send the call again.

So the run of numbers below is deliberately arranged around one question: are
pages still being abandoned? The supporting columns (unchanged calls, immediate
repeats) are there to show *why* if the answer goes bad again.

DO NOT read the wasted-seconds figure as the point. It was measured at ~12
seconds per run before the fix and ~2 after. That is real but trivial; the
repeat calls are cheap. Quoting a time saving here oversells the change and
sends the next person optimizing the wrong thing.

SAMPLE SIZE

Two runs is not evidence. The verdict line says how many runs are on each side
of the split, so a thin sample is visible rather than implied.
"""

import argparse
import glob
import re
from datetime import datetime

# The commit that rewrote the "unchanged" reply. Runs are split here so a
# before/after comparison does not depend on remembering the date.
FIX_COMMIT = "26046d3"
FIX_TIME = datetime(2026, 9, 7, 12, 10, 23)

# Every disposable `git archive` test vault leaves a log here beside the real
# one (see AGENTS.md on testing ingest changes against a copy). Those runs are
# not comparable with scheduled runs — different vault size, hand-picked
# sources, often a deliberately broken build — so the tool refuses to average
# them together and makes the caller name one vault.
LOG_GLOB = "logs/wiki_ingest.*.log*"
_VAULT_FROM_LOG = re.compile(r"wiki_ingest\.(.+?)\.log")

_TS = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),(\d{3})")
_RUN = re.compile(r"Starting wiki ingest run for vault")
_SOURCE = re.compile(r"\[INFO\] Ingesting '")
_EDIT = re.compile(r"tool_call edit_wiki_page\((\{.*?\})\) -> (\{.*)$")
# max_iterations is per stage: 14 plan, 12 execute, 4 log. The number is what
# makes this line unambiguously stage 2, so it is matched rather than inferred.
_INCOMPLETE = re.compile(r"incomplete: hit max_iterations=12")


def _when(line: str):
    m = _TS.match(line)
    if not m:
        return None
    return datetime.strptime(
        f"{m.group(1)}.{m.group(2)}000", "%Y-%m-%d %H:%M:%S.%f"
    )


def scan(paths):
    """Walk the logs once and return one dict per ingest run, oldest first."""
    runs, cur = [], None
    prev_args = None

    for path in paths:
        with open(path, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                ts = _when(line)
                if ts is None:
                    continue

                if _RUN.search(line):
                    cur = {"start": ts, "sources": 0, "edits": 0,
                           "unchanged": 0, "repeats": 0, "incomplete": 0}
                    runs.append(cur)
                    prev_args = None
                    continue

                if cur is None:
                    # Lines before the first run header belong to a run whose
                    # start was rotated away. Counting them against the next run
                    # would inflate it, so they are dropped.
                    continue

                if _SOURCE.search(line):
                    cur["sources"] += 1
                    continue

                if _INCOMPLETE.search(line):
                    cur["incomplete"] += 1
                    continue

                m = _EDIT.search(line)
                if m:
                    args, result = m.group(1), m.group(2)
                    cur["edits"] += 1
                    if '"unchanged"' in result:
                        cur["unchanged"] += 1
                        # Only an immediately-repeated identical call is the
                        # defect. The same content sent again much later is the
                        # model revisiting a page, which is not this bug.
                        if args == prev_args:
                            cur["repeats"] += 1
                    prev_args = args

    runs.sort(key=lambda r: r["start"])
    return runs


def _totals(runs):
    keys = ("sources", "edits", "unchanged", "repeats", "incomplete")
    tot = {k: sum(r[k] for r in runs) for k in keys}
    tot["runs"] = len(runs)
    return tot


_HEADER = (f"{'run start':<18}{'src':>5}{'edits':>7}{'unchg':>7}"
           f"{'repeat':>8}{'INCOMPLETE':>12}")


def _row(label, r):
    return (f"{label:<18}{r['sources']:>5}{r['edits']:>7}{r['unchanged']:>7}"
            f"{r['repeats']:>8}{r['incomplete']:>12}")


def report(runs, split, out=print):
    before = [r for r in runs if r["start"] < split]
    after = [r for r in runs if r["start"] >= split]

    for label, group in ((f"BEFORE {FIX_COMMIT}", before),
                         (f"AFTER  {FIX_COMMIT}", after)):
        tot = _totals(group)
        out(f"\n=== {label} — {tot['runs']} runs ===")
        out(_HEADER)
        for r in group:
            out(_row(f"{r['start']:%Y-%m-%d %H:%M}", r))
        out("-" * len(_HEADER))
        out(_row("TOTAL", tot))

    b, a = _totals(before), _totals(after)
    out("")
    if not a["runs"]:
        out("No runs after the fix yet — nothing to compare.")
        return
    per_b = b["incomplete"] / b["sources"] if b["sources"] else 0
    out(f"Pages abandoned mid-thought: {b['incomplete']} before "
        f"({per_b:.2f} per source), {a['incomplete']} after "
        f"({a['incomplete'] / a['sources']:.2f} per source)."
        if a["sources"] else "No sources ingested after the fix.")
    if a["runs"] < 5:
        out(f"Only {a['runs']} run(s) since the fix — treat this as a signal, "
            f"not a result. Re-run once there are five or more.")


def vaults_in(paths) -> dict[str, list[str]]:
    """Log paths grouped by the vault name embedded in each filename."""
    found: dict[str, list[str]] = {}
    for p in paths:
        m = _VAULT_FROM_LOG.search(p)
        if m:
            found.setdefault(m.group(1), []).append(p)
    return found


def resolve(explicit_logs, vault):
    """Pick the log files to read, or explain why the choice is ambiguous."""
    if explicit_logs:
        return sorted(explicit_logs)

    by_vault = vaults_in(sorted(glob.glob(LOG_GLOB)))
    if not by_vault:
        raise SystemExit(f"no logs matched {LOG_GLOB} — run from the repo root")

    if vault:
        if vault not in by_vault:
            raise SystemExit(
                f"no logs for vault '{vault}'. Found: {', '.join(sorted(by_vault))}"
            )
        return sorted(by_vault[vault])

    if len(by_vault) > 1:
        names = "\n  ".join(
            f"--vault {n}   ({len(f)} file{'s' if len(f) > 1 else ''})"
            for n, f in sorted(by_vault.items())
        )
        raise SystemExit(
            "logs from more than one vault are present, and scheduled runs are "
            "not comparable with disposable test-copy runs.\nPick one:\n  "
            + names
        )
    return sorted(next(iter(by_vault.values())))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("logs", nargs="*",
                    help=f"log files to read (default: discovered from {LOG_GLOB})")
    ap.add_argument("--vault", help="vault name to select, when logs/ holds more "
                                    "than one vault's runs")
    ap.add_argument("--split", default=FIX_TIME.isoformat(),
                    help=f"ISO timestamp dividing before/after (default: the "
                         f"{FIX_COMMIT} commit time)")
    args = ap.parse_args(argv)

    paths = resolve(args.logs, args.vault)
    print("reading: " + ", ".join(paths))

    runs = scan(paths)
    if not runs:
        raise SystemExit("no ingest runs found in those logs")
    report(runs, datetime.fromisoformat(args.split))


if __name__ == "__main__":
    main()
