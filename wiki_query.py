"""Ask a question against a vault's wiki, on demand.

Manual only — never scheduled, since a question needs a live human to ask
it. Uses the same vault-agnostic pattern as wiki_ingest.py: reads the
vault's own RULES.md for its Q&A conventions and answers with page
citations, using only the read-only wiki tools.

Usage:
    python wiki_query.py --vault ~/Vaults/llm-wiki-learnings "What have we learned about scoping agent tool schemas?"
"""

import argparse
import sys
from pathlib import Path

from agent.loop import run_agent
from agent.wiki_tools import QUERY_TOOL_SCHEMAS, query_dispatch

ANSWER_WRAPPER = """

You are answering a single question from the vault's owner, on demand. Follow \
the question-answering rules above: read wiki/index.md first to find relevant \
pages, read those pages, synthesize an answer, and cite the specific wiki pages \
you drew from by name. If the answer is not in the wiki, say so clearly instead \
of guessing."""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vault", required=True, help="Path to the Obsidian vault.")
    parser.add_argument("question", help="The question to ask the wiki.")
    args = parser.parse_args()

    rules_path = Path(args.vault) / "RULES.md"
    if not rules_path.is_file():
        print(f"error: {rules_path} not found", file=sys.stderr)
        return 1
    rules = rules_path.read_text(encoding="utf-8")

    answer = run_agent(
        system_prompt=rules + ANSWER_WRAPPER,
        user_prompt=args.question,
        tools=QUERY_TOOL_SCHEMAS,
        dispatch=query_dispatch(args.vault),
    )
    print(answer)
    return 0


if __name__ == "__main__":
    sys.exit(main())
