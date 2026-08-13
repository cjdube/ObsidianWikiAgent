"""Shared machinery for the vault entrypoints.

config/.env is loaded here because this is the one module that runs whichever
part of the package an entrypoint reaches for — including `python -m agent.loop
list-models`, which imports no other module in this package.

It used to be loaded as a side effect of importing agent/loop.py or
agent/notify.py. That covered all four entrypoints, but only by accident of
their import graphs: vault_snapshot.py got its NTFY_URL solely because it
imports agent.notify, so dropping that one import would have switched its
failure alerting off in silence.

override=False (the default), so a real environment variable still beats the
file — which is what makes `LLM_PROVIDER=gemini python wiki_ingest.py ...` work
as a per-run opt-in.
"""

from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / "config" / ".env")
