"""The usage ledger: one JSON line per model call, in logs/usage.jsonl.

Nothing in this repo reads that file, so these tests are the only thing that
holds the record shape still. A renamed field here is a reader outside this
repo that silently stops counting.
"""

import json
import logging
from datetime import datetime, timedelta

import pytest

from agent import loop, usage_ledger

# Captured verbatim from a real Ollama /api/chat call on 2026-09-01
# (gemma4:26b-mlx, num_predict=8 so the reply would hit the cap). Not a
# hand-written dict on purpose: `eval_count` is a field this repo had never
# read, and a fixture invented alongside the code that reads it proves only
# that the two agree with each other.
OLLAMA_CAPTURE = json.loads(
    '{"model":"gemma4:26b-mlx","created_at":"2026-09-01T23:06:21.440243Z",'
    '"message":{"role":"assistant","content":"","thinking":"*   Input: "},'
    '"done":true,"done_reason":"length","total_duration":6453108875,'
    '"load_duration":5834544750,"prompt_eval_count":22,'
    '"prompt_eval_duration":500284750,"eval_count":8,"eval_duration":117349041}'
)


class FakeResp:
    def __init__(self, json_data):
        self._json = json_data

    def json(self):
        return self._json


def rows():
    path = usage_ledger.ledger_path()
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def replies(monkeypatch, *bodies):
    """Answer each POST with the next body, repeating the last one forever."""
    queue = list(bodies)

    def fake_post(*args, **kwargs):
        return FakeResp(queue.pop(0) if len(queue) > 1 else queue[0])

    monkeypatch.setattr(loop, "_post_with_retry", fake_post)


# --- pricing ---------------------------------------------------------------


def test_an_unknown_model_is_unpriced_not_free():
    """None and 0.0 are different answers. A model missing from the table
    means the table is stale, and 0.0 would report that as a free call."""
    assert usage_ledger.estimate_cost("gemini", "gemini-9.9-ultra", 1000, 1000) is None


def test_ollama_is_free_without_consulting_the_table():
    assert usage_ledger.estimate_cost("ollama", "gemma4:26b-mlx", 99_000, 99_000) == 0.0


def test_the_longest_matching_prefix_wins():
    """'gemini-2.5-pro' also starts with nothing else in the table today, but
    the rule has to hold before a shorter prefix is ever added."""
    assert usage_ledger.estimate_cost(
        "gemini", "gemini-2.5-pro-preview", 1_000_000, 0
    ) == 1.25


def test_thinking_tokens_are_not_added_on_top():
    """Gemini counts them inside candidatesTokenCount already."""
    priced = usage_ledger.estimate_cost("gemini", "gemini-2.5-flash", 0, 1_000_000)
    assert priced == 2.50


# --- reading the response bodies -------------------------------------------


def test_ollama_eval_count_lands_in_output_tokens(monkeypatch):
    replies(monkeypatch, OLLAMA_CAPTURE)

    loop.run_agent("sys", "user", tools=[], dispatch={}, provider="ollama",
                   max_iterations=1)

    row = rows()[0]
    assert row["output_tokens"] == 8
    assert row["prompt_tokens"] == 22
    assert row["finish_reason"] == "length"
    assert row["backend"] == "ollama"
    assert row["cost_usd"] == 0.0


def test_gemini_camelcase_keys_are_read(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    monkeypatch.setenv("GEMINI_MODEL", "gemini-2.5-flash")
    replies(monkeypatch, {
        "candidates": [
            {"content": {"parts": [{"text": "done"}]}, "finishReason": "STOP"}
        ],
        "usageMetadata": {
            "promptTokenCount": 100,
            "candidatesTokenCount": 40,
            "thoughtsTokenCount": 25,
        },
    })

    loop.run_agent("sys", "user", tools=[], dispatch={}, provider="gemini")

    row = rows()[0]
    assert (row["prompt_tokens"], row["output_tokens"]) == (100, 40)
    assert row["thinking_tokens"] == 25
    assert row["finish_reason"] == "STOP"
    assert row["cost_usd"] == pytest.approx(0.0001300)


def test_the_sdk_snake_case_names_are_not_what_rest_returns(monkeypatch):
    """This repo calls the REST endpoint, not the Python SDK. If the reader
    were written against the SDK's attribute names it would record nulls
    forever and look like a provider that stopped reporting usage."""
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    monkeypatch.setenv("GEMINI_MODEL", "gemini-2.5-flash")
    replies(monkeypatch, {
        "candidates": [{"content": {"parts": [{"text": "done"}]}}],
        "usage_metadata": {"prompt_token_count": 100, "candidates_token_count": 40},
    })

    loop.run_agent("sys", "user", tools=[], dispatch={}, provider="gemini")

    row = rows()[0]
    assert row["prompt_tokens"] is None
    assert row["output_tokens"] is None


def test_a_blocked_gemini_response_records_the_failure(monkeypatch):
    """No candidates raises inside the loop; the row still has to be written,
    because a run that costs a prompt and returns nothing is the expensive
    kind of failure."""
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    monkeypatch.setenv("GEMINI_MODEL", "gemini-2.5-flash")
    replies(monkeypatch, {"promptFeedback": {"blockReason": "SAFETY"}})

    with pytest.raises(RuntimeError):
        loop.run_agent("sys", "user", tools=[], dispatch={}, provider="gemini")

    # The POST itself succeeded, so the row is an ok row with the counts the
    # body carried — which is none.
    assert rows()[0]["prompt_tokens"] is None


# --- one row per call ------------------------------------------------------


def test_a_three_iteration_loop_writes_three_rows(monkeypatch):
    """Not one row per run: the prompt is re-sent in full every iteration, and
    that growth is the thing the ledger exists to show."""
    tool_turn = {"message": {"tool_calls": [{"function": {"name": "t", "arguments": {}}}]}}
    replies(monkeypatch, tool_turn, tool_turn, {"message": {"content": "final"}})

    loop.run_agent("sys", "user", tools=[{"function": {"name": "t"}}],
                   dispatch={"t": lambda **kw: {"ok": True}}, provider="ollama")

    written = rows()
    assert len(written) == 3
    assert [r["caller"] for r in written] == ["run", "run", "run"]
    assert [r["tools_offered"] for r in written] == [1, 1, 1]


def test_a_one_shot_call_is_recorded_as_complete_text(monkeypatch):
    monkeypatch.setenv("OLLAMA_NUM_CTX", "12345")
    replies(monkeypatch, {"message": {"content": "ok"}})

    loop.complete_text("sys", "user", provider="ollama")

    row = rows()[0]
    assert row["caller"] == "complete_text"
    assert row["tools_offered"] == 0
    # The window the call actually asked for, not the code default: a row that
    # cannot say how big the context was cannot explain the prompt size.
    assert row["num_ctx"] == 12345


def test_the_task_is_the_logger_name(monkeypatch):
    replies(monkeypatch, {"message": {"content": "ok"}})

    loop.complete_text("sys", "user", provider="ollama",
                       logger=logging.getLogger("wiki_ingest.some-vault"))

    assert rows()[0]["task"] == "wiki_ingest.some-vault"


def test_a_call_with_no_logger_is_task_unknown(monkeypatch):
    replies(monkeypatch, {"message": {"content": "ok"}})

    loop.complete_text("sys", "user", provider="ollama")

    assert rows()[0]["task"] == "unknown"


def test_a_failed_call_is_recorded_as_not_ok(monkeypatch):
    def boom(*args, **kwargs):
        raise TimeoutError("model took too long")

    monkeypatch.setattr(loop, "_post_with_retry", boom)

    with pytest.raises(TimeoutError):
        loop.complete_text("sys", "user", provider="ollama")

    row = rows()[0]
    assert row["ok"] is False
    assert row["error"] == "TimeoutError: model took too long"


def test_every_row_carries_the_agreed_fields(monkeypatch):
    """The field names are a contract with a reader outside this repo."""
    replies(monkeypatch, OLLAMA_CAPTURE)

    loop.complete_text("sys", "user", provider="ollama")

    assert set(rows()[0]) == {
        "ts", "agent", "task", "caller", "backend", "model", "prompt_tokens",
        "output_tokens", "thinking_tokens", "num_ctx", "duration_ms",
        "finish_reason", "tools_offered", "ok", "error", "cost_usd",
    }
    assert rows()[0]["agent"] == "wiki"


# --- the writer itself -----------------------------------------------------


def test_record_swallows_an_oserror(tmp_path, monkeypatch):
    """A ledger is bookkeeping. A run that dies because bookkeeping died has
    traded the work for the record of it."""
    blocked = tmp_path / "a-directory"
    blocked.mkdir()
    monkeypatch.setattr(usage_ledger, "ledger_path", lambda: blocked)

    usage_ledger.record(task="t", caller="run", backend="ollama", model="m",
                        duration_ms=1)

    assert blocked.is_dir()


def test_the_prune_keeps_recent_rows_and_drops_old_ones(monkeypatch):
    monkeypatch.setenv("WIKI_USAGE_MAX_BYTES", "1")
    monkeypatch.setenv("WIKI_USAGE_RETENTION_DAYS", "30")
    old = (datetime.now() - timedelta(days=90)).isoformat(timespec="seconds")
    usage_ledger.ledger_path().write_text(
        json.dumps({"ts": old, "agent": "wiki"}) + "\n"
        + "not json at all\n",
        encoding="utf-8",
    )

    usage_ledger.record(task="t", caller="run", backend="ollama", model="m",
                        duration_ms=1)

    written = usage_ledger.ledger_path().read_text(encoding="utf-8").splitlines()
    assert old not in written[0]
    # An unreadable ts is kept: it is a bug in the writer, and a size trigger
    # is the wrong moment to delete the evidence of one.
    assert written[0] == "not json at all"
    assert len(written) == 2


def test_the_suite_never_writes_the_real_ledger(monkeypatch):
    """tests/conftest.py redirects agent.common.LOGS_DIR, and ledger_path()
    resolves against it per call rather than binding it at import.

    What this asserts is that the call left the real ledger untouched, not that
    the file is absent. logs/usage.jsonl is production data on any machine that
    actually runs the agent — the scheduled ingest appends to it — so the
    absence form passed only on a checkout that had never run one, and started
    failing here on 2026-09-02. Size is enough: an appended row always grows it.
    """
    from agent import common

    real = common._ROOT / "logs" / "usage.jsonl"
    before = real.stat().st_size if real.exists() else None

    replies(monkeypatch, {"message": {"content": "ok"}})
    loop.complete_text("sys", "user", provider="ollama")

    assert usage_ledger.ledger_path().exists()
    assert usage_ledger.ledger_path().parent == common.LOGS_DIR
    assert (real.stat().st_size if real.exists() else None) == before
