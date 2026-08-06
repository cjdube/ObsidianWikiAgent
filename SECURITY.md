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
  reads to Google's API. Nothing sets it by default, but the weekly lint job in
  `launchd/template.plist.txt` does set it for its judgment pass — so if you
  schedule that job as shipped, vault content goes to a third party every week.
  Drop the key from the plist to keep it local.
- **ntfy failure alerts** (`NTFY_URL`) carry the job name and an error string.
  The error can include a source filename or path, so a filename is the most
  that escapes this way — never page content.
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
the same check. The seven exposed tools are all vault file I/O — no shell, no
network, and no path argument that bypasses those two guards.

What does **not** contain it: nothing stops a model from being argued into
writing wrong or attacker-chosen *content* at a legitimate path inside the
vault, including overwriting a page you trust. The blast radius is the vault's
contents, and that is the guarantee — file the untrusted material in a vault
you don't treat as authoritative, and read `wiki/log.md`, which is the agent's
own account of what it changed and why.

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
