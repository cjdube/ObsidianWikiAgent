"""Push a short failure alert to the user's phone via a self-hosted ntfy server.

The scheduled jobs here are unattended and their logs are only read when
something already looks wrong — which is the wrong order. On 2026-08-03 a
wedged run burned nearly three hours and the only evidence was a line in
logs/learnings-ingest.launchd.log that nobody had reason to open.

Mirrors LocalLLMAgent's agent/tools/notify.py (same server, same topic
convention) rather than importing it: these are separate repos with separate
venvs, and this one needs only the plaintext publish path. It does not carry
that version's email fallback, so an ntfy outage loses the alert — acceptable
here, where the run is scheduled daily and a stuck run is visible again the
next morning.

Config (config/.env):
    NTFY_URL   full topic URL, e.g. http://mac-mini.tailnet.ts.net:2586/wiki-alerts
    NTFY_TOKEN publish token for that topic (Bearer). Optional but expected,
               since the self-hosted server runs auth-default-access: deny-all.

Leaving NTFY_URL unset switches push off; the jobs still log and still exit
non-zero.
"""

import logging
import os
from typing import Optional

import requests

_TIMEOUT_S = 10
_MAX_MESSAGE_CHARS = 500


def notify(message: str, title: str = None, priority: str = None) -> dict:
    """POST a notification to the configured ntfy topic. Returns {"ok": True}
    or {"error": ...} — never raises, so a push outage can't mask the failure
    it is trying to report."""
    url = os.getenv("NTFY_URL")
    if not url:
        return {"error": "NTFY_URL not set in config/.env"}

    headers = {}
    token = os.getenv("NTFY_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    if title:
        headers["Title"] = title
    if priority:
        headers["Priority"] = priority

    try:
        resp = requests.post(
            url,
            data=message[:_MAX_MESSAGE_CHARS].encode("utf-8"),
            headers=headers,
            timeout=_TIMEOUT_S,
        )
        resp.raise_for_status()
    except Exception as e:
        return {"error": f"{e.__class__.__name__}: {e}"}

    return {"ok": True}


def notify_failure(
    job_name: str, detail: object, logger: Optional[logging.Logger] = None
) -> None:
    """Push a one-line failure alert for a scheduled job (best-effort)."""
    result = notify(
        message=f"{job_name} failed: {detail}",
        title=f"WikiAgent: {job_name} failed",
        priority="high",
    )
    if logger and result.get("error"):
        logger.warning(f"Failure push via ntfy did not send: {result['error']}")
