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

from agent.loop import INCOMPLETE_PREFIX, run_agent
from agent.wiki_tools import QUERY_TOOL_SCHEMAS, query_dispatch

# agent.loop's default is 6, which is sized for a short exchange and not for the
# workflow below: read the index, then read the pages you need, then answer. One
# tool call per turn makes that the index plus four pages before the loop gives
# up — and on a vault of a few hundred pages a question spanning five is
# ordinary. 15 is proportionate for "index plus several pages". There is no
# unattended-cost argument for keeping it tight here the way there is for the
# ingest: this runs when a human asks it to and that human is waiting.
MAX_QUERY_ITERATIONS = 15

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
        max_iterations=MAX_QUERY_ITERATIONS,
    )
    print(answer)
    # A loop that ran out of turns returns its marker as the answer, so without
    # this a truncated run exits 0 and reads as a real reply to anything
    # scripting this.
    if answer.startswith(INCOMPLETE_PREFIX):
        print(
            f"error: the model used all {MAX_QUERY_ITERATIONS} tool calls "
            f"without answering — ask something narrower, or raise "
            f"MAX_QUERY_ITERATIONS in wiki_query.py.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
