# ObsidianWikiAgent

A fully local tool that maintains one or more Obsidian "LLM wikis" — Andrej
Karpathy's raw-notes-in, LLM-organized-wiki-out pattern — using a **local
LLM served by Ollama**. No Anthropic/Claude API calls at runtime.

Vault-agnostic by design: the script has no idea what subject any given
vault covers. Every vault supplies its own `RULES.md` (folder structure,
page format, citation rules); the Python is the same for all of them.

Sibling project to `LocalLLMAgent` (same Ollama-calling pattern, same
launchd-scheduling convention) but with no dependency on it — this repo only
knows about vaults, not calendars or email.

## Architecture

```
launchd (per-vault .plist, timed)
   -> python wiki_ingest.py --vault <path>
       -> reads <vault>/RULES.md as the system prompt
       -> finds raw/ sources not yet in wiki/.ingested.json
       -> for each: tool-calling loop against Ollama's /api/chat
          (read_raw_file, read/write_wiki_page, read/update_index,
           append_log) files it into wiki/, unattended
       -> marks it ingested, logs everything to logs/wiki_ingest.<vault>.log
```

`wiki_query.py --vault <path> "question"` is the read side — manual and
on-demand only, since a question needs a live human to ask it.

### `agent/loop.py`

Same `run_agent()` / `complete_text()` pair as `LocalLLMAgent` — a
tool-calling loop against Ollama's `/api/chat`, capped at 6 iterations.

### `agent/wiki_tools.py`

Vault-path-parameterized file I/O against `raw/` and `wiki/` — every
function takes `vault_path` explicitly, so the same functions serve any
vault. `INGEST_TOOL_SCHEMAS` / `QUERY_TOOL_SCHEMAS` are the OpenAI-style
tool schemas passed to `run_agent`.

## Setup (from scratch)

1. Install [Ollama](https://ollama.com) and pull a tool-calling-capable
   model (e.g. `ollama pull gemma4`). Verify: `ollama list`.
2. Create the venv:
   ```bash
   python3.12 -m venv .venv
   .venv/bin/pip install -r requirements.txt
   ```
3. Copy `config/.env.example` to `config/.env` — defaults to
   `OLLAMA_MODEL=gemma4`, `OLLAMA_HOST=http://localhost:11434`.
4. For each vault, make sure it has:
   ```
   <vault>/RULES.md   -- folder structure, page format, citation rules
   <vault>/raw/       -- source documents
   <vault>/wiki/       -- maintained pages (index.md, log.md created on first run)
   ```
5. Test manually before scheduling:
   ```bash
   .venv/bin/python wiki_ingest.py --vault ~/Documents/llm-wiki-[vault]
   .venv/bin/python wiki_query.py --vault ~/Documents/llm-wiki-[vault] "some question"
   ```
6. Load the launchd plist (see below).

## Adding a new vault

1. Create the vault folder with `RULES.md`, `raw/`, `wiki/`.
2. Copy `launchd/template.plist.txt` to a new
   `com.cjdube.wikiagent.<vault-name>-ingest.plist`, filling in the
   `--vault` path, `Label`, and log filename.
3. `cp` it to `~/Library/LaunchAgents/` and `launchctl load` it.

No Python changes required — this is the whole point of the vault-path
parameterization in `agent/wiki_tools.py`.

## Scheduling — launchd

Each vault has its own `.plist` in `launchd/`, copied to
`~/Library/LaunchAgents/` and loaded with `launchctl load`. Same convention
as `LocalLLMAgent`.

```bash
# check status
launchctl print gui/$(id -u)/com.cjdube.wikiagent.<vault-name>-ingest

# trigger on demand (bypasses the schedule, useful for testing)
launchctl start com.cjdube.wikiagent.<vault-name>-ingest

# reload after editing a .plist
launchctl unload ~/Library/LaunchAgents/com.cjdube.wikiagent.<vault-name>-ingest.plist
cp launchd/com.cjdube.wikiagent.<vault-name>-ingest.plist ~/Library/LaunchAgents/
launchctl load ~/Library/LaunchAgents/com.cjdube.wikiagent.<vault-name>-ingest.plist
```

Logs land in `logs/wiki_ingest.<vault-name>.log` (structured, written by
the script) and `logs/<vault-name>-ingest.launchd.log` (raw stdout/stderr).

## What's NOT here

- No Anthropic/Claude API usage at runtime — Claude Code was only used to
  *write* this code.
- No lint/audit command yet — `RULES.md`'s lint workflow is a natural
  follow-on reusing `agent/wiki_tools.py`, not built yet.
- `config/.env` and `logs/*.log` are gitignored.
