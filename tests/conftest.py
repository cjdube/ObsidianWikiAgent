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
