# AGENTS.md

Repository-specific guidance for coding agents. The working method below belongs here too: Codex reads this file and does not read user-level configuration. Keep preferences that have nothing to do with this repository out of it.

## Project boundaries

- This is a vault-agnostic, local-first engine for maintaining Obsidian LLM wikis. Keep subject matter and personal data out of the Python code.
- Each vault's `RULES.md` is runtime policy for that vault's scope, folders, page format, and citations. It is not a coding-agent instruction file and must remain separate from this `AGENTS.md`.
- Ollama is the local default. `LLM_PROVIDER=gemini` is an explicit privacy boundary: raw sources and wiki pages read by that run leave the machine. Preserve that opt-in behavior.
- Claude Code and Codex are development tools only. Neither is a runtime dependency or provider.

## Working method

- Evidence: Treat logs and dated notes as leads. A report must describe the code as it is now, not as a log described it. Give the date of any evidence you cite.
- Environment: Run project commands through `.venv/bin/python`; run the suite with `.venv/bin/pytest`. The system Python may be too old.
- Measurement: Every performance claim needs a named before/after number, and that number must point to one change.
- Instructions: Write steps that run top to bottom with no reordering. Do not go silent during long work; say what you have found so far. A checkpoint may carry leads you have not confirmed yet, as long as it says they are unconfirmed.
- Git: When the user explicitly asks for a commit or push, work directly on `main` and do not create a branch or pull request unless requested. Make each commit revertible on its own, and say why in the body.

## Architectural invariants

- Preserve the staged ingest: plan without writes, execute one isolated conversation per planned page, then write one source-level log entry.
- Keep the pending-source queue oldest-first so stable filename prefixes cannot starve later sources.
- Choose create versus update from the page's current existence on disk. Existing pages receive only the edit path; new pages receive only the create path.
- Keep `**Sources**`, `**Last updated**`, and index descriptions under deterministic Python ownership. Do not move whole-document rewriting back into model prompts.
- Keep model-visible catalog and tool results bounded by the answer, not by total vault size. Avoid tools that return an entire growing index or page catalogue when a search or section list will do.
- One exception, measured 2026-09-11: the deep lint's judgment pass may call `list_wiki_pages`. It is the only pass whose job is the whole vault, a bare name list is 4.9% of the window at 609 pages where the index with summaries is 35%, and withholding it held Tier B recall at 1.0 of 12 against 4.2 with it. The exception is for that one pass and that one tool; the read path keeps search only. Re-measure before widening it, and watch the name list against the window as the vault grows.
- Preserve the execute stage's verified-link boundary: it may link only to names supplied by the plan. Missing links are lintable; invented links are damage.

## Verification

- Run `.venv/bin/pytest` after code changes.
- Run the real ingest against a fresh disposable vault copy built from `git archive`, never against the live vault, whenever a change can alter what gets written to a page. Compare wall clock, truncation/retry counts, new lint findings, and per-page insertions/deletions with the baseline.
- For documentation-only changes, validate links and run `git diff --check`; the Python suite is not required.
- Every changed line must trace to the request. Inspect the final diff for lines that do not before declaring completion.

## Context to load on demand

- Ingest performance, page writes, link scope, escaped-content handling, read-path scaling, cross-source synthesis, live-vault operations, and LocalLLMAgent integration are already decided and measured in [`docs/agent-context.md`](docs/agent-context.md). Read it before planning in those areas, so you do not re-open a settled question. It records dated decisions, measurements, resolved incidents, and watch items; reverify historical claims against current state.

