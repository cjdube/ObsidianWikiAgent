# Seeded defect pack for the deep-lint model comparison

This directory is test data for `tools/compare_lint_models.py`. Nothing in the
running system reads it, and no vault job touches it.

## What it is

A small, self-contained wiki with **twelve planted defects** — three in each of
the four categories the judgment pass claims to cover (`LINT_WRAPPER` in
`wiki_lint.py`): contradictions, duplicate concepts, out-of-scope pages, and
outdated claims.

Every planted defect is invisible to the structural pass on purpose. Titles
differ where the subjects duplicate, every link resolves, every page is indexed
and has an inbound link, and every `**Sources**` line cites a file that exists.
So a `wiki_lint.py --vault <copy>` run without `--deep` must report **zero**
findings. If it does not, the seed is leaking into the structural score and the
judgment number means nothing. `compare_lint_models.py --check` asserts this.

## Why twelve, and why a standalone vault

The 2026-09-03 comparison (`docs/agent-context.md`) planted four defects in a
533-page vault and scored 4 of 12 against 2 of 12. Two problems with that:

- Four defects is inside the noise. Three trials cannot separate two models.
- The judgment pass has only `search_wiki_pages` and `read_wiki_page`. It has
  no way to list the vault, so on 533 pages recall is mostly sampling luck.

So this pack is its own ~35-page vault. It is small enough that a thorough
auditor can plausibly reach every page, which makes the score measure judgment
rather than luck. Use `--inject` to plant the same defects in a copy of a real
vault once a winner is picked; that run is the acceptance test, not the
discriminator.

## Reachability, and why the out-of-scope pages read oddly

Search matches a page's slug, title and summary. A page nothing plausibly
searches for cannot be found by any model, and would score zero for all of
them. So each out-of-scope page's summary names an in-scope term. That is also
how out-of-scope pages really arrive: a single source mixes subjects and the
ingest files the wrong half.

## Files

- `RULES.md` — the fixture vault's own scope and page format. The out-of-scope
  defects are defined against this file, so read it before adding one.
- `seed.json` — the manifest. One entry per defect: the pages it spans and the
  slugs a correct finding must name.
- `source.md` — the single raw source every page cites. Deliberately not
  date-named, so `check_source_coverage` ignores it.
- `pages/` — the page bodies. Defect pages and filler, in one flat directory.

## Adding a defect

1. Add the page or pages to `pages/`. Give each a real title, a `**Summary**`,
   a `**Sources**: source.md` line, and a past `**Last updated**` date.
2. Add an entry to `seed.json`.
3. Run `.venv/bin/python tools/compare_lint_models.py --check`. It must report
   zero structural findings. A duplicate-concept pair especially: give the two
   pages genuinely different prose, or `check_template_twins` will catch them
   in code and the defect stops being a judgment test.

## Adding filler

Filler is not padding. Precision needs pages a model can be wrong about. Keep
each filler page ordinary, in scope, and internally consistent.
