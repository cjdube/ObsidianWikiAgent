"""Shared fixtures for the test suite.

Everything runs against a throwaway vault under pytest's tmp_path — no real
vault, no network, no Ollama/Gemini. The `vault` fixture returns a small
helper object so tests can seed raw sources and wiki pages in a line or two.
"""

import logging
import sys
from pathlib import Path

import pytest

# Import the package under test without installing it.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent import common as _common  # noqa: E402

# The real logs/, captured before _isolate_logs can redirect it.
_REAL_LOGS_DIR = _common.LOGS_DIR


class _ProductionLogWrite(BaseException):
    """A test opened a log handler on the repo's real logs/ directory."""


_orig_file_handler_init = logging.FileHandler.__init__


def _guarded_file_handler_init(self, filename, *args, **kwargs):
    """Refuse any handler pointed at the real logs/, permanently.

    _isolate_logs below redirects agent.common.LOGS_DIR, which covers every
    handler built the way setup_logger builds one. It does not cover a handler
    built any other way, and a redirect that gets bypassed is silent — the write
    simply lands in production, which is how logs/vault_snapshot.{nothing,plain,
    repo2}.log got there in the first place. This turns that from a stray file
    into a failure in the test that caused it.

    Installed at import rather than as a fixture so it also covers collection,
    and raising BaseException so a bare `except Exception` cannot eat it.
    """
    if Path(filename).resolve().parent == _REAL_LOGS_DIR:
        raise _ProductionLogWrite(
            f"a test tried to open the production log {filename} — the logs/ "
            f"redirect in tests/conftest.py is not in effect for whatever built "
            f"this handler."
        )
    _orig_file_handler_init(self, filename, *args, **kwargs)


logging.FileHandler.__init__ = _guarded_file_handler_init


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
def _hermetic_model_env(monkeypatch):
    """Pin the model settings the loop reads, so the suite does not depend on
    whether the developer happens to have a config/.env.

    agent/__init__.py loads that file on import, so before this fixture the
    suite passed here and would have failed on a clean checkout the moment
    _ollama_model stopped defaulting to a hardcoded tag. Tests that care about
    a specific value still monkeypatch over these.
    """
    monkeypatch.setenv("OLLAMA_MODEL", "test-model")
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_MODEL", raising=False)


@pytest.fixture(autouse=True)
def _clear_budget():
    """agent/budget.py holds the run's deadline and retry ceiling in module
    state, so a test that starts one would otherwise leak it into the next."""
    from agent import budget

    budget.reset()
    yield
    budget.reset()


# --- Suite-wide egress guards ----------------------------------------------
#
# The three fixtures above isolate; these three refuse. The difference matters:
# a redirect that a future test bypasses is silent, and the thing it was meant
# to stop happens anyway. logs/vault_snapshot.{nothing,plain,repo2}.log are what
# that looks like when it is only a log. It looks worse when it is a push to a
# phone at 3am, or a real request to the Ollama the ingest is using.
#
# Each guard raises a BaseException subclass, not an Exception. The code they
# guard is written to degrade rather than crash — agent/notify.py:59 catches
# bare Exception so a push outage can't mask the failure it reports, and
# _post_with_retry catches RequestException so a slow local model doesn't fail a
# run. An ordinary error raised inside either would be swallowed and the test
# would pass having proved nothing.


class _NtfyEgress(BaseException):
    """A test reached the real ntfy server."""


class _ModelEgress(BaseException):
    """A test reached the real model API."""


@pytest.fixture(autouse=True)
def _block_ntfy_egress(monkeypatch):
    """NTFY_URL is a real, reachable topic in config/.env, and agent/__init__.py
    loads that file at import — so nothing about running the suite makes the push
    path safe by itself. Tests that exercise a failure path stub notify_failure
    by hand today; this is what makes a missed stub loud instead of a
    notification.
    """
    from agent import notify

    def _refuse(*args, **kwargs):
        raise _NtfyEgress(
            "a test tried to POST to the real ntfy server — stub "
            "notify_failure (or agent.notify.notify) in the test that did this."
        )

    monkeypatch.setattr(notify.requests, "post", _refuse)


@pytest.fixture(autouse=True)
def _block_model_egress(monkeypatch):
    """agent/loop.py sends every Ollama and Gemini call through one pooled
    Session, so blocking its post and get covers both providers and the model
    discovery call. Ollama runs on this machine and answers on localhost, so an
    unstubbed call here does not fail fast — it succeeds, slowly, against the
    model the real ingest is using.
    """
    from agent import loop

    def _refuse(*args, **kwargs):
        raise _ModelEgress(
            "a test tried to call the real model API — stub loop._session.post, "
            "loop._post_with_retry, or the run_agent/complete_text the code "
            "under test calls."
        )

    monkeypatch.setattr(loop._session, "post", _refuse)
    monkeypatch.setattr(loop._session, "get", _refuse)
