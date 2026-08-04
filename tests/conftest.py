"""Shared fixtures for the test suite.

Everything runs against a throwaway vault under pytest's tmp_path — no real
vault, no network, no Ollama/Gemini. The `vault` fixture returns a small
helper object so tests can seed raw sources and wiki pages in a line or two.
"""

import sys
from pathlib import Path

import pytest

# Import the package under test without installing it.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


RULES_MD = """\
# Test vault rules

## Raw folders

- daily-notes Notes captured day to day.
- misc Anything that doesn't fit elsewhere.

## Page format

Every page has a title, Summary, Sources, and Last updated line.
"""


class Vault:
    """A minimal on-disk vault plus helpers to populate it."""

    def __init__(self, root: Path):
        self.root = root
        self.path = str(root)
        (root / "raw").mkdir(parents=True, exist_ok=True)
        (root / "wiki").mkdir(parents=True, exist_ok=True)
        (root / "RULES.md").write_text(RULES_MD, encoding="utf-8")

    def raw(self, name: str, content: str = "raw content", subdir: str = "") -> Path:
        d = self.root / "raw" / subdir if subdir else self.root / "raw"
        d.mkdir(parents=True, exist_ok=True)
        p = d / name
        p.write_text(content, encoding="utf-8")
        return p

    def page(self, name: str, content: str) -> Path:
        p = self.root / "wiki" / (name if name.endswith(".md") else f"{name}.md")
        p.write_text(content, encoding="utf-8")
        return p

    def index(self, content: str) -> Path:
        p = self.root / "wiki" / "index.md"
        p.write_text(content, encoding="utf-8")
        return p


@pytest.fixture
def vault(tmp_path):
    return Vault(tmp_path)


@pytest.fixture(autouse=True)
def _isolate_logs(tmp_path, monkeypatch):
    """Keep test runs out of the repo's real logs/.

    Every entrypoint's main() calls setup_logger, which resolves
    agent.common.LOGS_DIR at call time and hands the path to a
    RotatingFileHandler. Without this, a test that exercises main() writes a
    fixture-named log into the production directory — which is how
    logs/vault_snapshot.{nothing,plain,repo2}.log got there. A per-file stub
    (see test_wiki_ingest.py) is the convention; this is the backstop that
    makes a missed one harmless.
    """
    from agent import common

    monkeypatch.setattr(common, "LOGS_DIR", tmp_path)


@pytest.fixture(autouse=True)
def _clear_budget():
    """agent/budget.py holds the run's deadline and retry ceiling in module
    state, so a test that starts one would otherwise leak it into the next."""
    from agent import budget

    budget.reset()
    yield
    budget.reset()
