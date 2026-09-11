# Agent Engineering Notes — wiki rules

A small knowledge base about building and running LLM agents. It is a test
fixture: every page in it is invented. Do not treat any claim here as fact
about a real system.

## Scope

In scope:

- LLM inference, context windows, quantization and local model serving
- Retrieval: embeddings, chunking, vector indexes, ranking
- Agent mechanics: tool calling, retries, run budgets, schedulers, evaluation
- The engineering craft those depend on — logging, testing, configuration

Out of scope — never create a page for these, whatever the source says:

- Social events, team outings, parties, and catering
- Volunteer days, charity drives, and fundraising
- Employee benefits, payroll, parking, and human-resources administration
- Facilities, travel booking, and office logistics

A source may mix in-scope and out-of-scope material. Ingest only the in-scope
part and silently skip the rest.

## Page format

Every page has, in this order:

- A `# Title` heading in human-readable words, not the filename.
- A `**Summary**:` line, one sentence.
- A `**Sources**:` line naming the raw files the page was built from.
- A `**Last updated**:` line, an ISO date.
- The body, in `## ` sections.

Link related pages with `[[slug]]`. Attribute claims in the body with
`(source: <filename>)`.

## What earns a page

One page per concept. Two pages must never cover the same concept — merge them
instead. A page must not contradict another page: when a newer source changes a
fact, update every page that states the old one.

## Reporting

Report findings as a numbered list. Each finding names the specific pages it
concerns and suggests a fix.
