# ObsidianWikiAgent

A local-first tool that maintains one or more Obsidian "LLM wikis" — Andrej
Karpathy's raw-notes-in, LLM-organized-wiki-out pattern. The daily ingests run
against a **local LLM served by Ollama** and send nothing off the box.

There is one opt-in exception, and it is worth knowing about before you put
anything private in a vault: setting `LLM_PROVIDER=gemini` routes that run to
Google's API instead, which means the vault pages and raw sources it reads
leave your machine. Nothing here turns it on: the default is Ollama, and no
plist template in `launchd/` sets it. It is a per-run or per-job opt-in you
have to make yourself — worth knowing up front because the weekly lint job is
where it is tempting (see step 6 below), and a scheduled job you configure that
way sends vault content to a third party every week.

Vault-agnostic by design: the script has no idea what subject any given
vault covers. Every vault supplies its own `RULES.md` (folder structure,
page format, citation rules); the Python is the same for all of them.

Sibling project to `LocalLLMAgent` (same Ollama-calling pattern, same
launchd-scheduling convention) but with no dependency on it — this repo only
knows about vaults, not calendars or email. Nor is there a code dependency the
other way: `LocalLLMAgent`'s dashboard can *report* on these jobs — schedules,
run history, failures — by reading this repo's `launchd/` and `logs/` directly,
which needs nothing from here beyond the log format the scripts already use.
Optional on both sides; see the `WREN_RUN_LOG` key in
`launchd/template.plist.txt`.

`LocalLLMAgent` is a separate repo; nothing here needs it. The whole of the
integration is the log format described under [Scheduling](#scheduling--launchd),
and this repo runs standalone without it.

## Architecture

```
launchd (per-vault .plist, timed)
   -> python wiki_ingest.py --vault <path>
       -> reads <vault>/RULES.md as the system prompt
       -> sorts freshly dropped raw/ files into the subdirectories the
          vault declares under "## Raw folders" (local model picks the
          folder per file; no-op for vaults that declare none)
       -> finds raw/ sources not yet in wiki/.ingested.json, OLDEST FIRST
       -> for each: tool-calling loop against Ollama's /api/chat
          (read_raw_file, list_wiki_pages, read/write_wiki_page,
           list_index_sections, update_index, append_log) files it into
           wiki/, unattended
       -> marks it ingested, logs everything to logs/wiki_ingest.<vault>.log
```

### Two rules the ingest queue and index depend on

**The queue is oldest-first, not alphabetical.** The order decides who starves
when a run hits its budget, and alphabetical order looks neutral while being
anything but: feeds drop files under stable prefixes, so the same prefix sits
at the tail every single day. `Daily-YouTube-*` went unfiled from 2026-07-31
onward behind `Daily-Chrome-*` and `AI-Chat-Learnings-*` — each day added two
sources that sorted ahead of it while the run only ever reached two. FIFO
cannot starve anything indefinitely: waiting longest is what earns a turn.

**`update_index` files one page, and never takes the document.** It used to
accept the whole of `index.md` as a string, which made it the most expensive
thing in an ingest: regenerating a 45 KB table of contents is ~12k output
tokens against a 32k context already holding the source and several pages. On
2026-08-04 the model spent 30 of one run's 45 budgeted minutes on five attempts
(45426, 45460, 45450, 12224, 3108 chars), timed out three times, and each retry
came back shorter until a truncated 3 KB stub was written over the real index,
stripping 91 descriptions. That was not the model failing at its job — it was
being asked to do Python's. Naming a page and a section is ~15 tokens and
cannot be truncated into a valid-looking wrong answer. Descriptions come from
each page's own `**Summary**:` line, so they cannot be fabricated either.

The read side follows: the ingest is offered `list_index_sections`, not
`read_index`. Filing a page only requires knowing which headings exist — a few
dozen names — while the index itself is 57 KB on a 336-page vault, about a
quarter of the default context window, and grows with every page added. Both
`write_wiki_page` calls on `index.md` and `log.md` are refused outright, so
neither file can be rewritten by a route that skips these guarantees.

`wiki_query.py --vault <path> "question"` is the read side — manual and
on-demand only, since a question needs a live human to ask it.

### `agent/loop.py`

Same `run_agent()` / `complete_text()` pair as `LocalLLMAgent` — a
tool-calling loop that dispatches to Ollama's `/api/chat` by default, or to
Gemini's `generateContent` when `LLM_PROVIDER=gemini`. Only the wire format
differs; each provider owns its own request/history translation and they share
the tool-dispatch step.

The 6-iteration cap is only the default for callers that don't say otherwise —
`wiki_ingest.py` raises it to 30 (one source can touch 10–15 pages),
`wiki_lint.py --deep` to 60, and `wiki_query.py` to 15 (the index plus several
pages). A loop that runs out of turns returns a `[incomplete: …]` marker rather
than an answer; `wiki_query.py` exits non-zero when it sees one, so a truncated
run can't pass for a real reply.

`OLLAMA_MODEL` has no fallback: an unset one raises with the fix rather than
sending a bare family name Ollama can't resolve and retrying the 404 five
times. Context overflow is silent (Ollama drops the oldest messages), so a run
whose opening prompt already fills half of `OLLAMA_NUM_CTX` logs a warning
before it starts accumulating tool results. That estimate only covers the part
of a run that can't grow, so every Ollama turn also logs the prompt size Ollama
itself reports (`prompt_eval_count`) at INFO, and warns past 70% of the window.
Output length is capped by
`OLLAMA_NUM_PREDICT` (2000): Ollama's own default is unlimited, so a repetition
loop generates until the client times out — one such call cost a whole 45-minute
run. A reply that stops at the cap logs a warning, so truncation is never
silent either.

### `agent/wiki_tools.py`

Vault-path-parameterized file I/O against `raw/` and `wiki/` — every
function takes `vault_path` explicitly, so the same functions serve any
vault. `INGEST_TOOL_SCHEMAS` / `QUERY_TOOL_SCHEMAS` are the OpenAI-style
tool schemas passed to `run_agent`.

It also has a small CLI for inspecting what the agent would see, which is the
quickest way to check the ingest queue's order or spot a page the model filed
under a name you didn't expect:

```bash
.venv/bin/python -m agent.wiki_tools list-raw --vault ~/Vaults/llm-wiki-learnings
.venv/bin/python -m agent.wiki_tools list-wiki --vault ~/Vaults/llm-wiki-learnings
```

## Setup (from scratch)

1. Install [Ollama](https://ollama.com) and pull a **tool-calling-capable**
   model — the ingest is nothing but tool calls, so a model without that
   support fails every run. Verify with `ollama list`, and set `OLLAMA_MODEL`
   to the exact tag it prints: a bare family name like `gemma4` is not a tag
   Ollama resolves, and the machine this was built on runs
   `OLLAMA_MODEL=gemma4:26b-mlx`.
2. Create the venv:
   ```bash
   python3.12 -m venv .venv
   .venv/bin/pip install -r requirements.txt
   ```
3. Copy `config/.env.example` to `config/.env` and set `OLLAMA_MODEL` to your
   tag from step 1. `OLLAMA_HOST` defaults to `http://localhost:11434`.
4. For each vault, make sure it has:
   ```
   <vault>/RULES.md   -- folder structure, page format, citation rules
   <vault>/raw/       -- source documents
   <vault>/wiki/       -- maintained pages (index.md, log.md created on first run)
   ```
   Keep the vault out of `~/Documents`, `~/Desktop` and `~/Downloads`. macOS
   gates those behind a consent prompt keyed to the interpreter's exact path,
   so a `brew upgrade python@3.12` silently revokes it — and the next scheduled
   run then blocks on a dialog nobody is there to answer. That cost one run
   seven hours on 2026-08-13.
5. Test manually before scheduling:
   ```bash
   .venv/bin/python wiki_ingest.py --vault ~/Vaults/llm-wiki-learnings
   .venv/bin/python wiki_query.py --vault ~/Vaults/llm-wiki-learnings "some question"
   ```
6. Load the launchd plist (see below).

## Using it day to day

Once a vault is set up and scheduled, this is the actual workflow:

1. **Drop raw material into `<vault>/raw/`.** Notes, transcripts, exported
   docs — whatever the vault ingests. That's the only manual step; you
   don't organize it or tell the agent it's there.

   Text sources only. A binary — a PDF, a screenshot, anything with a NUL byte
   in its first 8 KB — is skipped rather than decoded, and the run logs which
   ones it skipped. That is deliberate: a PNG read as text returns hundreds of
   thousands of replacement characters, and the model then writes what the
   *filename* implies and cites the file for every claim (see `_is_text_source`
   in `agent/wiki_tools.py`). OCR it first, or transcribe it, then drop the
   text in.
2. **Ingestion runs on its own schedule** (the vault's launchd `.plist`).
   To see it happen right away instead of waiting — e.g. just after
   dropping a file in — trigger it on demand:
   ```bash
   launchctl start local.wikiagent.<vault-name>-ingest
   ```
   or run it directly, bypassing launchd entirely:
   ```bash
   .venv/bin/python wiki_ingest.py --vault ~/Vaults/llm-wiki-learnings
   ```
3. **Check what it did.** `<vault>/wiki/log.md` is the agent's own account of
   what it read, what it changed, and any judgment calls it made (it runs
   unattended, so it never stops to ask). Read it as a summary, not as
   evidence: the model writes it, so it is only as reliable as the run that
   produced it. `logs/wiki_ingest.<vault-name>.log` is the record Python
   writes — one line per tool call, with its arguments and result — and it is
   what to read when a run's behaviour is actually in question. A source is
   only marked done once a write actually lands — a page write, or the
   `log.md` entry that closes out the source. If the local model reads a file
   but returns without writing anything at all (a transient no-op that happens
   on dense sources), the run re-attempts it a few times, then leaves it for
   the next scheduled run — so a stuck file clears itself without intervention.

   Note the gap that leaves: a run that wrote *some* pages and then died still
   counts as done and is never retried. That is the hole `wiki_lint.py`'s
   source-coverage check exists to report after the fact — see step 6.
4. **Browse the result in Obsidian.** `<vault>/wiki/` is plain markdown —
   open the vault normally in the Obsidian app and start from
   `wiki/index.md`, the table of contents the agent maintains.
5. **Ask it a question directly**, any time, without waiting for a
   scheduled run:
   ```bash
   .venv/bin/python wiki_query.py --vault ~/Vaults/llm-wiki-learnings "What have we learned about scoping agent tool schemas?"
   ```
   This is read-only — it answers by citing the specific wiki pages it
   drew from and never edits anything.
6. **Audit the wiki** for problems that creep in over time:
   ```bash
   .venv/bin/python wiki_lint.py --vault ~/Vaults/llm-wiki-learnings
   ```
   Report-only by default — without `--fix` it never writes to the vault.
   Structural checks (broken and self links, orphan pages, index gaps, source
   coverage, page format, misspelled slugs, duplicate titles, template twins,
   lens integrity) run in Python, so they are instant, free, and cannot miss a
   page. Exits non-zero when it finds something.

   Source coverage is the one that catches an ingest that stopped early: a
   source is recorded in `wiki/.ingested.json` as soon as any write lands, so a
   run that wrote one page and died is never retried. A dated capture that owes
   a page named after it and does not have one is reported here. Re-run it by
   removing its name from `wiki/.ingested.json`.

   Add `--deep` for the checks code can't do — contradictions between pages,
   one concept split across two names, out-of-scope pages, and claims a newer
   source has overtaken. That pass calls the model, so it costs a couple of
   minutes.

   Add `--fix` to apply the two mechanical fixes that need no judgment:
   stripping self-links, and de-linking index entries pointing at pages that
   no longer exist. This is the only way `wiki_lint.py` writes to the vault,
   and it is deliberately narrow — every judgment call (which of two
   overlapping pages survives, whether an orphan earns its page, a bad date, an
   invented citation) is left for you.

   Add `--json` to get the structural findings as one JSON object on stdout
   instead of prose, for a caller that renders its own UI (LocalLLMAgent's
   `/wiki/lint` view shells out to exactly this):

   ```bash
   .venv/bin/python wiki_lint.py --vault ~/Vaults/llm-wiki-learnings --json
   ```

   Structural pass only, and it deliberately writes no log — this path is
   driven by a button, and a click that logged "Starting wiki lint run" would
   invent a scheduled run that never happened in the file the dashboard parses
   for run history. `--deep` is refused rather than ignored, since a page load
   cannot wait on a multi-minute model conversation. `--fix` *is* honoured, so
   `--json --fix` writes to the vault like the prose path does. The exit code
   means the same thing either way: 1 means findings exist, not that the run
   failed.

   A vault can audit itself on a schedule too, with a lint plist you write the
   same way as an ingest one (there is no lint template — copy
   `launchd/template.plist.txt` and point `ProgramArguments` at `wiki_lint.py`).
   The setup this was built against runs Sunday 10:00, after that morning's
   ingest, and lands its report in `logs/<vault-name>-lint.launchd.log`. That
   plist is also where `LLM_PROVIDER=gemini` gets set, because the judgment
   pass is where model quality decides whether the findings are worth reading —
   a deliberate, per-job choice, and the one place vault content leaves the
   machine. The daily ingests stay on the local default.

   The prose report is for a human; alongside it the run is logged through
   `setup_logger` like the ingest and snapshot jobs are — run boundaries,
   per-section counts, and the judgment pass's tool calls, in
   `logs/wiki_lint.<vault-name>.log`. Findings log at INFO, never WARNING: a
   weekly audit result is something you read, not something that should page
   you. A run that finds problems still exits non-zero, but it logs as a
   completed run — the lint worked.
7. **Back the vault up to git**, if it has a remote:
   ```bash
   .venv/bin/python vault_snapshot.py --vault ~/Vaults/llm-wiki-learnings
   ```
   Stages everything, commits as `Vault snapshot <date>: <n> files`, pushes.
   Exits without committing when nothing changed, so it is safe to schedule
   as often as you like. A failed run pushes an ntfy alert naming the cause
   (see below) — an offsite backup that quietly stops backing up is the one
   failure you will not notice on your own.

   **Use an SSH remote, not HTTPS.** An HTTPS remote authenticates through
   git's `osxkeychain` helper, and a job scheduled overnight wakes the Mac
   into *dark wake*, where the keychain refuses any access that could need
   UI. The helper fails with `failed to get: -25320` (`errSecInDarkWake`),
   git falls back to asking for a username, and `GIT_TERMINAL_PROMPT=0`
   turns that into a hard failure — every night, while the commits keep
   succeeding. An SSH remote with a passphrase-less key reads the key file
   directly and touches no keychain, so it works in dark wake:

   ```bash
   git -C ~/Vaults/llm-wiki-learnings remote set-url origin git@github.com:<you>/<vault>.git
   ```

   This is a separate job from the ingest, deliberately: a vault has two
   writers — the scheduled ingest and you editing in Obsidian — and a step
   at the end of `wiki_ingest.py` would miss every hand edit made on a day
   with no new `raw/` sources, and would let an ingest failure cost you the
   backup of those edits too.

   It never force-pushes. A failed push is almost always a non-fast-forward
   from a second machine, which wants a human choosing a side; the commit is
   already safe locally, so the job just logs and stops.

## Adding a new vault

1. Create the vault folder with `RULES.md`, `raw/`, `wiki/`.
2. Copy `launchd/template.plist.txt` to a new
   `launchd/local.wikiagent.<vault-name>-ingest.plist`, filling in
   the placeholders (repo path, `--vault` path, `Label`, log filename).
   Concrete `.plist` files are gitignored — they stay local to your machine,
   so there is nothing to commit.
3. Validate it before loading — a plist with stray text (e.g. anything
   before the `<?xml ...?>` declaration) parses fine as a copy but fails
   `launchctl load` with an opaque `Input/output error`:
   ```bash
   plutil -lint launchd/local.wikiagent.<vault-name>-ingest.plist
   ```
4. Copy it into `~/Library/LaunchAgents/` — note the `~`. That's your
   per-user LaunchAgents directory; `/Library/LaunchAgents` (no `~`) is a
   different, root-owned system directory, and `cp` there fails with
   `Permission denied`. Then load it:
   ```bash
   cp launchd/local.wikiagent.<vault-name>-ingest.plist ~/Library/LaunchAgents/
   launchctl load ~/Library/LaunchAgents/local.wikiagent.<vault-name>-ingest.plist
   ```
5. Optionally, if the vault has a git remote, repeat steps 2–4 with
   `launchd/template-snapshot.plist.txt` for a `<vault-name>-snapshot` job
   that commits and pushes the vault daily.

No Python changes required — this is the whole point of the vault-path
parameterization in `agent/wiki_tools.py`.

### Why scheduled jobs must run through `.venv/bin/python`

A vault under `~/Documents` (or `~/Desktop`) is protected by macOS TCC, and
the grant is per-binary. The `.venv` interpreter has one; a launchd-spawned
`/bin/bash` does not. A shell script scheduled the same way fails on every
vault file with `Operation not permitted` — including `git`, which reports
the vault as "not a git repository" — while the identical script run from
your terminal works, because the terminal has its own grant. Keep the
`ProgramArguments` interpreter as `.venv/bin/python` and this never bites.

## Scheduling — launchd

Each vault has its own `.plist` in `launchd/`, copied to
`~/Library/LaunchAgents/` and loaded with `launchctl load`. Same convention
as `LocalLLMAgent`.

```bash
# check status
launchctl print gui/$(id -u)/local.wikiagent.<vault-name>-ingest

# trigger on demand (bypasses the schedule, useful for testing)
launchctl start local.wikiagent.<vault-name>-ingest

# reload after editing a .plist
launchctl unload ~/Library/LaunchAgents/local.wikiagent.<vault-name>-ingest.plist
cp launchd/local.wikiagent.<vault-name>-ingest.plist ~/Library/LaunchAgents/
launchctl load ~/Library/LaunchAgents/local.wikiagent.<vault-name>-ingest.plist
```

Logs land in `logs/wiki_ingest.<vault-name>.log` (structured, written by
the script) and `logs/<vault-name>-ingest.launchd.log` (raw stdout/stderr).

Both are capped. The structured log rotates at 5 MB, keeping three backups.
The launchd log can't rotate — launchd opens it when the job starts and the
job's stdout *is* that descriptor, so renaming it would send the rest of the
run's output to a file nobody reads — so the script instead rewrites its tail
in place at startup, keeping the last 1 MB once it passes 5 MB. That needs the
script to know which file launchd chose, which is why the `.plist` repeats the
path in `EnvironmentVariables` as `WIKI_LAUNCHD_LOG`; if you leave it out,
nothing breaks and nothing gets trimmed.

## When a run goes wrong

A scheduled ingest is bounded twice over, because an unbounded one does real
damage beyond being late: Ollama runs with `OLLAMA_NUM_PARALLEL=1`, so a run
that keeps retrying holds the only slot every other consumer of that Ollama is
queued behind. On 2026-08-03 a wedged MLX runner had the 09:00 ingest retrying
until 11:54 — nearly three hours during which nothing else could get a model
call through.

- **A wall-clock budget for the whole run**, 45 minutes by default
  (`WIKI_RUN_BUDGET_MINUTES` in `config/.env`, or `--budget-minutes` for a
  one-off). The deadline is enforced from inside the HTTP retry loop, not just
  between sources, since that is where the hours actually go: the script won't
  start a backoff it can't finish, and shortens a request timeout that would
  overshoot.
- **A retry ceiling per source** (8), counted across all of that source's
  attempts and loop iterations. The per-call cap alone was never enough —
  30 iterations x 5 HTTP attempts x 3 ingest attempts is up to 450 retries for
  one file.

Either limit stops the whole run, logs an error, and exits non-zero. Work
already done is kept: finished sources stay marked, unfinished ones stay
unmarked and are picked up by the next scheduled run.

A stopped or failed run also pushes a phone alert via ntfy, so it doesn't wait
on someone opening a log. Set `NTFY_URL` (and usually `NTFY_TOKEN`) in
`config/.env` — the same self-hosted server `LocalLLMAgent` alerts through.
Leaving `NTFY_URL` unset switches push off; runs still log and still exit
non-zero. Delivery is best-effort and never masks the failure it reports.

## Running the tests

Deterministic unit tests cover the file I/O, path safety and reserved-name
guards in `agent/wiki_tools.py`, every structural check in `wiki_lint.py`, the
agent loop's dispatch/retry/provider plumbing in `agent/loop.py`, the raw-file
sort step in `wiki_ingest.py`, and the run budget and failure alerting in
`agent/budget.py`, `wiki_ingest.py`, `wiki_lint.py` and `vault_snapshot.py`.
The LLM HTTP layer is mocked, so no Ollama or Gemini is needed and nothing hits
the network — and the suite pins its own model settings, so it does not depend
on whether you have a `config/.env`.

```bash
.venv/bin/pip install -r requirements-dev.txt
.venv/bin/pytest
```

## What's NOT here

- No Anthropic/Claude API usage at runtime — Claude Code was only used to
  *write* this code. The one external API this can call is Gemini, and only
  when `LLM_PROVIDER=gemini` is set for that run (see the top of this file).
- `config/.env` and `logs/*.log*` are gitignored — the trailing `*` covers the
  rotated `.log.1` backups, which contain every tool call's full result and so
  hold verbatim vault content.
