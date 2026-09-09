# Removing content from a wiki

Use this when material reached the wiki that should not be in it — something
leaked into a raw source and the ingest processed it.

There is no command for this. Every step is manual. Do the steps in the order
below: the raw source goes first, because a source left in `raw/` can be read
again.

Set the vault path once, then paste the commands as written:

```bash
export VAULT=~/Vaults/llm-wiki-learnings
```

## 1. Find every copy

```bash
grep -rn "the text to remove" "$VAULT"
```

The content is usually in four places:

- `raw/<folder>/<Source>.md` — the source document
- `wiki/<page>.md` — one or more pages written from it
- `wiki/index.md` — the table-of-contents line for those pages
- `wiki/log.md` — the entry that records the ingest

## 2. Delete the raw source

```bash
rm "$VAULT/raw/<folder>/<Source>.md"
```

Do this first. A source still in `raw/` can be ingested again.

## 3. Leave `wiki/.ingested.json` alone

Do not remove the source name from that file. The name is the record that says
the source is done. Remove the name, and the next run reads the source again.

A ledger entry for a deleted raw file is harmless. `check_source_coverage`
only looks at sources that are still in `raw/`.

## 4. Remove the wiki content

- A whole page is bad: `rm "$VAULT/wiki/<bad-page>.md"`
- Only part of a good page is bad: open that page and delete those lines. Keep
  the rest of the page.

## 5. Delete the index line

Open `$VAULT/wiki/index.md`. Find the line that holds `[[bad-page]]` and delete
the whole line.

Do this by hand. `wiki_lint --fix` only flattens the dead link to plain text.
The page name and its description would stay in the index.

## 6. Delete the log entry

Open `$VAULT/wiki/log.md`. Delete the entry that names the deleted source.

## 7. Clean the `**Sources**` lines

Other pages can still cite the deleted source by name. Find them:

```bash
grep -rln "<Source>.md" "$VAULT/wiki/"
```

Remove that filename from the `**Sources**` line of each page found.

## 8. Check the result

```bash
cd /Users/craigdube/Projects/ObsidianWikiAgent && .venv/bin/python wiki_lint.py --vault "$VAULT"
```

Lint reports broken links and orphan pages. Repair those links by hand. Lint
does not repair links in page bodies.

## 9. Decide what to do about git history

The vault is a git repository with a remote. Deleting a file today does not
delete it from the git history, and it does not delete it from the remote.

- **Commit the deletion and leave the history.** This is the usual choice. It
  is sufficient when the repository is private and the content was unwanted,
  not secret.
- **Rewrite the history with `git filter-repo`, then force push.** This is slow
  and it breaks every other clone of the vault. Use it only for a true secret:
  a password, an API key, or another person's private data.

## Note on the Gemini provider

If the ingest ran with `LLM_PROVIDER=gemini`, the source text already left the
machine. Deleting these files does not undo that.
