"""One JSON line per model call, appended to logs/usage.jsonl.

Nothing in this repo reads this file. Something outside it does, which makes
the field names a contract rather than a convenience — in particular `agent`
is the literal string "wiki", and a renamed key is a reader that silently
stops counting this repo's calls rather than one that fails.

There is deliberately no "how many tokens did I use" command here. The ledger
is written in this repo and read in another, on purpose: a reader on this side
would be a second answer to the same question, free to disagree with the first.

Written under an flock because several launchd jobs (ingest, lint, snapshot)
can overlap and this is the repo's first shared-write file. A single append of
a ~300-byte line would survive on O_APPEND alone; the prune rewrite below
would not.
"""

import fcntl
import json
import logging
import os
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from agent import common

LEDGER_NAME = "usage.jsonl"

# Size is the trigger, age is the rule: the file is only rewritten once it is
# big, and the rewrite then keeps whatever is recent. Doing it the other way
# would mean parsing every row on every model call.
_DEFAULT_MAX_BYTES = 5_000_000
_DEFAULT_RETENTION_DAYS = 90

# USD per *million* tokens, as (input, output), keyed by model-name prefix;
# longest prefix wins. Ollama never consults this — it short-circuits to 0.0.
#
# These rates go stale. The provider console is the billing record; this table
# is an estimate for spotting a run that cost more than expected.
#
# Thinking tokens are not added on top: Gemini already counts them inside
# candidatesTokenCount, so adding them would double-bill exactly the calls
# that reason the most.
_PRICES = {
    "gemini-2.5-flash": (0.30, 2.50),
    "gemini-2.5-pro": (1.25, 10.00),
    "gemini-3.6-flash": (0.30, 2.50),
    "gemini-3.7-flash": (0.30, 2.50),
}

_logger = logging.getLogger(__name__)


def ledger_path() -> Path:
    """Resolved per call, not bound at import.

    agent.common.LOGS_DIR is what tests/conftest.py redirects to tmp_path. A
    module-level `LEDGER_PATH = LOGS_DIR / name` would capture the real path
    at import time and write production rows from the test suite, which is the
    same failure that put logs/vault_snapshot.*.log in the repo.
    """
    return common.LOGS_DIR / LEDGER_NAME


def estimate_cost(
    backend: str,
    model: str,
    prompt_tokens: Optional[int],
    output_tokens: Optional[int],
) -> Optional[float]:
    """USD for one call, or None when the model is not in the table.

    None and 0.0 are different answers — "we don't know" versus "it was free".
    The reader counts the nulls separately and shows that count beside the
    total, which is how a stale price table announces itself. This repo can
    reach any model list_gemini_models() returns, so unpriced rows are likely;
    that is fine, as long as they are visible.
    """
    if backend == "ollama":
        return 0.0
    name = model or ""
    matches = [p for p in _PRICES if name.startswith(p)]
    if not matches:
        return None
    in_rate, out_rate = _PRICES[max(matches, key=len)]
    return round(
        (prompt_tokens or 0) / 1_000_000 * in_rate
        + (output_tokens or 0) / 1_000_000 * out_rate,
        6,
    )


def record(
    *,
    task: str,
    caller: Optional[str],
    backend: str,
    model: str,
    duration_ms: int,
    prompt_tokens: Optional[int] = None,
    output_tokens: Optional[int] = None,
    thinking_tokens: Optional[int] = None,
    num_ctx: Optional[int] = None,
    finish_reason: Optional[str] = None,
    tools_offered: int = 0,
    ok: bool = True,
    error: Optional[str] = None,
) -> None:
    """Append one row. Never raises.

    A ledger is bookkeeping; a run that fails because bookkeeping failed has
    traded the work for the record of it. Every failure here is a debug line.
    """
    try:
        row = {
            "ts": datetime.now().isoformat(timespec="seconds"),
            "agent": "wiki",
            "task": task,
            "caller": caller,
            "backend": backend,
            "model": model,
            "prompt_tokens": prompt_tokens,
            "output_tokens": output_tokens,
            "thinking_tokens": thinking_tokens,
            "num_ctx": num_ctx,
            "duration_ms": duration_ms,
            "finish_reason": finish_reason,
            "tools_offered": tools_offered,
            "ok": ok,
            "error": error,
            "cost_usd": estimate_cost(backend, model, prompt_tokens, output_tokens),
        }
        path = ledger_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with _locked(path):
            with open(path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(row) + "\n")
            _prune(path)
    except Exception:
        _logger.debug("usage ledger write failed", exc_info=True)


@contextmanager
def _locked(path: Path):
    """Exclusive lock on a sibling .lock file.

    The lock is not taken on the ledger itself because _prune replaces that
    file: an fd holding a lock on the old inode guards nothing once the new
    one is in place.
    """
    lock_path = path.with_suffix(path.suffix + ".lock")
    with open(lock_path, "w") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, "") or default)
    except ValueError:
        return default


def _prune(path: Path) -> None:
    """Past WIKI_USAGE_MAX_BYTES, keep only rows inside the retention window.

    A row whose `ts` cannot be read is kept: an unparseable line is a bug in
    the writer, and deleting the evidence of it on a size trigger is the wrong
    way to find out.
    """
    max_bytes = _env_int("WIKI_USAGE_MAX_BYTES", _DEFAULT_MAX_BYTES)
    if path.stat().st_size <= max_bytes:
        return

    days = _env_int("WIKI_USAGE_RETENTION_DAYS", _DEFAULT_RETENTION_DAYS)
    cutoff = datetime.now() - timedelta(days=days)
    kept = [
        line
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and _keep(line, cutoff)
    ]
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text("".join(f"{line}\n" for line in kept), encoding="utf-8")
    tmp.replace(path)


def _keep(line: str, cutoff: datetime) -> bool:
    try:
        return datetime.fromisoformat(json.loads(line)["ts"]) >= cutoff
    except Exception:
        return True
