# Security

## Reporting a vulnerability

Use GitHub's private vulnerability reporting on this repo — **Security** tab →
**Report a vulnerability**. That keeps the report private until there's a fix,
which a public issue does not. There's no email address here on purpose; the
GitHub form is the only channel.

This is a personal project maintained in spare time. Expect a reply in days,
not hours, and no backported fixes — only `main` is supported.

## What leaves your machine

By default, nothing. The ingest talks to Ollama on `localhost` and writes only
to the vault.

Three exceptions, all in your control:

- **`LLM_PROVIDER=gemini`** sends the vault pages and raw sources that run
  reads to Google's API. Nothing in this repo turns it on: `DEFAULT_PROVIDER`
  is `ollama`, `config/.env.example` ships the line commented out, and neither
  plist template sets it. Switching it on is a choice you make — per run
  (`LLM_PROVIDER=gemini .venv/bin/python wiki_ingest.py …`), or per job by
  adding it to a plist's `EnvironmentVariables`. Concrete plists are
  gitignored, so that choice never travels with a clone; if you make it for a
  scheduled job, vault content goes to a third party on that job's schedule.
- **ntfy failure alerts** (`NTFY_URL`) carry the job name and an error string,
  truncated to 500 characters. In practice that error is a filename or a path.
  It is not *guaranteed* to be content-free, though: the detail passed to
  `notify_failure` is whatever exception ended the run, and a few carry a
  fragment of what they were reading — a `JSONDecodeError` from a malformed
  model response quotes part of that response, which mid-run is the page being
  written. Treat it as "filenames and paths, with a small tail risk", not as a
  guarantee.
- **`vault_snapshot.py`** pushes the vault to whatever git remote *the vault*
  has. That remote is yours to choose; make sure a private vault has a private
  remote.

## Treat `raw/` as untrusted input

This is the one that matters. Files you drop into `<vault>/raw/` are fed
verbatim to a tool-calling model that holds write access to `<vault>/wiki/`.
Anything in a source document is therefore reaching a model that can act on it,
and a document written to manipulate that model is a real attack, not a
theoretical one.

What contains it: `_safe_page_path` and `_safe_raw_path` in
`agent/wiki_tools.py` resolve every model-supplied filename and reject anything
landing outside `<vault>/wiki` and `<vault>/raw`. Both are tested directly
against `../`, `../../etc/passwd` and absolute paths; because they compare
against `.resolve()`d paths, a symlink pointing out of the vault is caught by
the same check. The exposed tools are all vault file I/O — no shell, no
network, and no path argument that bypasses those two guards. (Nine distinct
tools are advertised across the three ingest stages — `PLAN_TOOL_SCHEMAS`,
`CREATE_PAGE_TOOL_SCHEMAS`, `UPDATE_PAGE_TOOL_SCHEMAS` and `LOG_TOOL_SCHEMAS`
in `agent/wiki_tools.py`. No stage sees all nine; each is offered only what its
one job needs. The dispatch resolves a tenth, `read_index`, which is
deliberately callable-but-unadvertised so a vault's `RULES.md` naming it in
prose still works. It is a read, and it is inside the same guards.)

The sharpest of those limits is the split between the two stage-2 sets. Which
one a unit is offered is decided by whether its page is already on disk, not by
what the model or its plan asked for: a unit updating an existing page holds
`edit_wiki_page` and no tool that can replace a file, and a unit creating a new
one holds `write_wiki_page` and no way to edit. An update therefore cannot
throw the existing page away — the worst it can do is add wrong text, which the
diff still shows.

Note what that does not reach. `write_wiki_page` takes the page name as an
argument, so a unit that was given it — because *its* page did not exist — can
still name a different page that does, and overwrite it. The split constrains
which tool each unit gets, not which file the tool is pointed at.

`write_wiki_page` additionally refuses the two filenames Python owns,
`index.md` and `log.md` (`RESERVED` in `agent/wiki_tools.py`). Both sit inside
`wiki/`, so the path guards admit them; without the name check that tool was a
way to truncate the table of contents that `update_index` exists to protect, or
the operation log below.

What does **not** contain it: nothing stops a model from being argued into
writing wrong or attacker-chosen *content* at a legitimate path inside the
vault, including overwriting a page you trust. The blast radius is the vault's
contents, and that is the guarantee — file the untrusted material in a vault
you don't treat as authoritative.

### Which record to trust afterwards

`wiki/log.md` is written by the model, from the text it passes to `append_log`.
It is a useful narrative of what a run thought it was doing, but it is **not
evidence**: a source written to manipulate the model is also, by the same
means, writing that account of itself.

The record outside the model's reach is the structured log,
`logs/wiki_ingest.<vault>.log`. Python writes one line per dispatch — `tool_call
<name>(<args>) -> <result>` — from the actual call, before the model sees the
outcome and with no tool that can edit it. If a run's behaviour is in question,
that is the file to read. Each logged argument and result is clipped at 2,000
characters (`_LOG_VALUE_MAX_CHARS` in `agent/loop.py`) so the log keeps the
shape of a run rather than a second copy of the vault — but 2,000 characters of
a source or a page is still source and page content, which is why `logs/*.log*`
is gitignored.

Binaries in `raw/` are refused rather than decoded, for a related reason
documented at `_is_text_source`: a PNG decoded with `errors="replace"` yields
noise that the model confidently narrates into cited, entirely fabricated
claims.

## Secrets and local state

- `config/.env` holds `GEMINI_API_KEY` and `NTFY_TOKEN`. It is gitignored and
  has never been committed.
- Concrete `launchd/*.plist` files are gitignored — they carry absolute paths
  and may carry an API key in `EnvironmentVariables`. Only the placeholder
  templates are tracked.
- `logs/*.log*` is gitignored, trailing `*` included: rotated `.log.1` backups
  hold every tool call's full result, meaning verbatim raw-source and wiki-page
  content.

If you fork this, re-check those four rules before your first push.

## Scope

A single-user CLI plus launchd jobs. No network listener, no daemon, no auth,
no multi-tenancy — the security boundary is your user account and macOS TCC.
It is not hardened for shared machines or for ingesting material from people
you don't trust.
