# AGENTS.md

Repository-specific guidance for coding agents. Generic personal preferences belong in user-level configuration, not here.

## Project boundaries

- This is a vault-agnostic, local-first engine for maintaining Obsidian LLM wikis. Keep subject matter and personal data out of the Python code.
- Each vault's `RULES.md` is runtime policy for that vault's scope, folders, page format, and citations. It is not a coding-agent instruction file and must remain separate from this `AGENTS.md`.
- Ollama is the local default. `LLM_PROVIDER=gemini` is an explicit privacy boundary: raw sources and wiki pages read by that run leave the machine. Preserve that opt-in behavior.
- Claude Code and Codex are development tools only. Neither is a runtime dependency or provider.

## Working method

- Evidence: Treat logs and dated notes as leads. Confirm a problem against the current file, current vault state, and intervening commits before reporting or fixing it.
- Environment: Run project commands through `.venv/bin/python`; run the suite with `.venv/bin/pytest`. The system Python may be too old.
- Measurement: Change one performance variable at a time, land it, then compare a named before/after metric before proposing the next optimization.
- Instructions: Put prerequisites and commands in the order they must execute. During a long investigation, surface a short checkpoint before more slow or machine-intensive work.
- Git: When the user explicitly asks for a commit or push, work directly on `main` and do not create a branch or pull request unless requested. Keep one idea per commit and explain the reason in the commit body.

## Architectural invariants

- Preserve the staged ingest: plan without writes, execute one isolated conversation per planned page, then write one source-level log entry.
- Keep the pending-source queue oldest-first so stable filename prefixes cannot starve later sources.
- Choose create versus update from the page's current existence on disk. Existing pages receive only the edit path; new pages receive only the create path.
- Keep `**Sources**`, `**Last updated**`, and index descriptions under deterministic Python ownership. Do not move whole-document rewriting back into model prompts.
- Keep model-visible catalog and tool results bounded by the answer, not by total vault size. Avoid tools that return an entire growing index or page catalogue when a search or section list will do.
- Preserve the execute stage's verified-link boundary: it may link only to names supplied by the plan. Missing links are lintable; invented links are damage.

## Verification

- Run `.venv/bin/pytest` after code changes.
- After ingest, write-path, prompt, or tool-schema changes, run the real ingest against a fresh disposable vault copy built from `git archive`, never against the live vault. Compare wall clock, truncation/retry counts, new lint findings, and per-page insertions/deletions with the baseline.
- For documentation-only changes, validate links and run `git diff --check`; the Python suite is not required.
- Before declaring completion, inspect the final diff and account for every changed line.

## Context to load on demand

- Ingest performance, page writes, link scope, escaped-content handling, read-path scaling, cross-source synthesis, live-vault operations, or LocalLLMAgent integration: read [`docs/agent-context.md`](docs/agent-context.md) before planning or changing behavior. It records dated decisions, measurements, resolved incidents, and watch items; reverify historical claims against current state.

