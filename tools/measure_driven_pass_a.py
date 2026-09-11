#!/usr/bin/env python3
"""Score a judgment sweep whose page reads are driven by Python, not the model.

Run it:

    .venv/bin/python tools/measure_driven_pass_a.py \\
        --vault <copy> --score tools/lint_defects/seed.json --trials 3

Not a job. Nothing schedules this and it writes nothing to the vault. It exists
to price and score one design before that design is built into wiki_lint.py.

WHY THIS EXISTS

`wiki_lint.py --deep` asks a model to audit a wiki and lets the model choose
what to open. Measured 2026-09-11 against a 70-page slice of the live vault, it
opened 24 of 70 and reported the wiki clean, with 86 tool calls and 20 minutes
of its budget still unspent. Nothing stopped it; it stopped because it believed
it was done. Slicing the vault does not fix that, because slicing changes what
the model *can* read and the defect is what it *chooses* to read.

So the loop moves into Python. One call per page, the page's text supplied by
the caller, no read tool offered. "Every page was read" stops being a claim
about the model's diligence and becomes a property of a for-loop.

WHAT PASS A CAN ACTUALLY JUDGE — ONE CATEGORY, NOT TWO

Of the four judgment categories, only **out-of-scope** survives being shown a
single page. The seed pack settles this: every `outdated` defect in
tools/lint_defects/seed.json carries a `superseded_by` naming a DIFFERENT page,
so an outdated claim is as pairwise as a contradiction or a duplicate. Nine of
the twelve planted defects need two pages. Scoring here therefore counts the
`out_of_scope` defects and ignores the rest — crediting this pass for a
category it cannot see would flatter it.

THREE VERDICTS, NOT TWO

A reply that is neither a clean verdict nor a finding is counted as UNPARSED and
reported as its own number. It is never folded into "clean". A sweep that
quietly treats an unreadable answer as a clean page is the exact failure this
whole effort exists to kill, and on the first measured run three of ten flagged
replies were the model reasoning in prose and concluding nothing.

Precision is NOT scored. A flagged page that is not in the answer key may be a
true finding the pack never planted — on the 35-page fixture the three
`daily-notes` pages were flagged, and all three were right. Read them.
"""

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from agent.loop import complete_text  # noqa: E402
from agent.wiki_tools import list_wiki_pages, read_wiki_page  # noqa: E402

CLEAN, FINDING, UNPARSED = "CLEAN", "FINDING", "UNPARSED"
VERDICT = "VERDICT:"

# The verdict goes LAST, and that ordering is the whole trick. Forcing it onto
# the first line made the model commit before it reasoned: with think=False its
# reasoning has nowhere to live but the reply, so a verdict-first prompt is a
# guess-first prompt. Measured on the 35-page fixture, verdict-first flagged 15
# of 35 pages and then argued itself back to CLEAN inside nine of them.
PASS_A_WRAPPER = f"""

You are auditing ONE page of this wiki. That page is below, in full. It is the
only page you can see and the only page you may judge.

Judge one thing only: does the Scope section above exclude this page's subject?

Do NOT report contradictions, duplicate concepts, or outdated claims. Each of
those needs a second page to compare against and you do not have one. Another
pass does that work.

Work it out in a few sentences if you need to. Then finish.

The LAST line of your reply must be exactly one of these two lines:

{VERDICT} {CLEAN}
{VERDICT} {FINDING}

Use {VERDICT} {FINDING} only when the Scope section excludes this page's
subject. When it does, put one numbered item directly above that line naming
this page, quoting the Scope line that excludes it, and giving a one-sentence
fix.

Write nothing after the verdict line."""


def verdict_of(reply: str) -> str:
    """CLEAN, FINDING or UNPARSED, read from the last verdict line.

    Scanned from the end because the model reasons on its way there and may
    use both words while doing so. Anything without a verdict line is UNPARSED
    on purpose: guessing a verdict out of prose is how a sweep starts
    reporting pages it never really judged.
    """
    for line in reversed(reply.strip().splitlines()):
        line = line.strip().lstrip("*# ").rstrip("*.")
        if line.upper().startswith(VERDICT):
            answer = line[len(VERDICT):].strip().upper()
            if answer.startswith(FINDING):
                return FINDING
            if answer.startswith(CLEAN):
                return CLEAN
    return UNPARSED


def slice_of(vault: Path, limit: int) -> list[str]:
    """The first `limit` page names, alphabetically.

    Alphabetical because it is deterministic and reviewable. Which pages land
    in which slice does not matter to Pass A: its one check needs one page, so
    the ordering cannot change the answer, only the reading order.
    """
    pages = list_wiki_pages(str(vault))["pages"]
    return pages[:limit] if limit else pages


def judge(vault: Path, name: str, rules: str, model: str) -> tuple[str, float]:
    """One page, one call. Returns the model's reply and the seconds it took.

    think=False is not a tuning choice. Measured 2026-09-11 on
    qwen3.8:27b-mlx, reasoning cost 46s a page against 0.4s with it off — 7.3
    hours against 72 minutes for a 569-page vault. The stage-1 warning in
    _run_ollama does not transfer: that failure was a model not calling a tool,
    and this call offers none.
    """
    page = read_wiki_page(str(vault), name)["content"]
    started = time.monotonic()
    reply = complete_text(
        system_prompt=rules + PASS_A_WRAPPER,
        user_prompt=f"Page: {name}\n\n{page}",
        model=model,
        think=False,
    )
    return reply.strip(), time.monotonic() - started


def sweep(vault: Path, names: list[str], rules: str, model: str,
          out=print) -> list[dict]:
    """Every page in `names`, in order. Returns one row per page."""
    rows = []
    for i, name in enumerate(names, 1):
        reply, seconds = judge(vault, name, rules, model)
        v = verdict_of(reply)
        rows.append({"name": name, "reply": reply,
                     "seconds": seconds, "verdict": v})
        mark = {CLEAN: ".", FINDING: "!", UNPARSED: "?"}[v]
        out(f"{i:>4}/{len(names)} {mark} {seconds:>5.1f}s  {name}")
    return rows


def answer_key(path: Path) -> dict[str, str]:
    """{page slug: defect id} for the defects a single-page pass can reach.

    `kind` is the pack's own field, so this does not depend on the id letters.
    """
    seed = json.loads(path.read_text(encoding="utf-8"))
    return {d["pages"][0]: d["id"]
            for d in seed["defects"] if d["kind"] == "out_of_scope"}


def report(rows: list[dict], total: float, key: dict[str, str] | None,
           out=print) -> dict:
    """One trial's numbers. Returns them so trials can be compared."""
    flagged = [r for r in rows if r["verdict"] == FINDING]
    unparsed = [r for r in rows if r["verdict"] == UNPARSED]
    times = sorted(r["seconds"] for r in rows)

    for r in flagged + unparsed:
        out(f"\n--- {r['name']} [{r['verdict']}] ---\n{r['reply']}")

    out("")
    out(f"Swept {len(rows)} of {len(rows)} pages in {total:.0f}s.")
    out(f"mean {total / len(rows):.1f}s/page, "
        f"median {times[len(times) // 2]:.1f}s, slowest {times[-1]:.1f}s")
    out(f"{len(flagged)} flagged, "
        f"{len(rows) - len(flagged) - len(unparsed)} clean, "
        f"{len(unparsed)} UNPARSED.")

    caught = 0
    if key:
        named = {r["name"].removesuffix(".md") for r in flagged}
        hit = sorted(key[p] for p in key if p in named)
        missed = sorted(key[p] for p in key if p not in named)
        caught = len(hit)
        out(f"out_of_scope recall {caught} of {len(key)}"
            f"  found {hit}  missed {missed}")
        extra = sorted(n for n in named if n not in key)
        out(f"{len(extra)} flagged page(s) not in the key — read them, they "
            f"are not automatically wrong: {extra}")

    return {"seconds": total, "flagged": len(flagged),
            "unparsed": len(unparsed), "caught": caught}


def summarise(trials: list[dict], key_size: int, out=print) -> None:
    """Across trials. One trial is not a measurement; see the handoff brief."""
    out("")
    out(f"=== {len(trials)} trials ===")
    for field, label in (("caught", f"recall of {key_size}"),
                         ("flagged", "flagged"),
                         ("unparsed", "UNPARSED"),
                         ("seconds", "seconds")):
        vals = [t[field] for t in trials]
        out(f"{label:<16} mean {statistics.mean(vals):>7.1f}   {vals}")
    out("")
    out("Precision is NOT scored here. Read the findings.")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--vault", required=True, type=Path,
                    help="a COPY of a vault — never the live one")
    ap.add_argument("--pages", type=int, default=0,
                    help="how many pages to sweep (default: all of them)")
    ap.add_argument("--model", default=None,
                    help="Ollama tag; defaults to OLLAMA_MODEL")
    ap.add_argument("--score", type=Path, default=None,
                    help="seed.json to score the out_of_scope defects against")
    ap.add_argument("--trials", type=int, default=1,
                    help="sweeps to run (default 1; the brief asks for 3)")
    args = ap.parse_args(argv)

    rules_path = args.vault / "RULES.md"
    if not rules_path.exists():
        raise SystemExit(f"no RULES.md in {args.vault}")

    rules = rules_path.read_text(encoding="utf-8")
    names = slice_of(args.vault, args.pages)
    if not names:
        raise SystemExit(f"no pages in {args.vault}")

    key = answer_key(args.score) if args.score else None
    if key:
        missing = sorted(p for p in key if f"{p}.md" not in names)
        if missing:
            raise SystemExit(
                f"the answer key names {len(missing)} page(s) this sweep will "
                f"not visit: {missing}. Scoring would report a miss the sweep "
                f"never had a chance at. Drop --pages, or drop --score."
            )

    print(f"sweeping {len(names)} pages from {args.vault}", flush=True)

    trials = []
    for t in range(1, args.trials + 1):
        if args.trials > 1:
            print(f"\n=== trial {t}/{args.trials} ===", flush=True)
        started = time.monotonic()
        rows = sweep(args.vault, names, rules, args.model,
                     out=lambda s: print(s, flush=True))
        trials.append(report(rows, time.monotonic() - started, key))

    if args.trials > 1:
        summarise(trials, len(key) if key else 0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
