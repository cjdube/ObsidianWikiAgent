"""Tests for the provider-agnostic agent loop in agent/loop.py.

No network: requests.post (and the retry helper, for run_agent) is mocked, and
time.sleep is neutralized so backoff paths run instantly.
"""

import logging

import requests

from agent import loop


class FakeResp:
    def __init__(self, status=200, headers=None, json_data=None):
        self.status_code = status
        self.headers = headers or {}
        self._json = json_data or {}

    def json(self):
        return self._json

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.exceptions.HTTPError(f"HTTP {self.status_code}")


def _sequenced(items):
    """A fake callable that yields the given results (raising any Exception)."""
    it = iter(items)

    def _call(*args, **kwargs):
        r = next(it)
        if isinstance(r, Exception):
            raise r
        return r

    return _call


# --- _dispatch_tool --------------------------------------------------------


def test_dispatch_unknown_tool():
    result = loop._dispatch_tool("nope", {}, {}, None)
    assert result == {"error": "unknown tool 'nope'"}


def test_dispatch_tool_exception_is_caught():
    def boom(**kwargs):
        raise RuntimeError("kaboom")

    result = loop._dispatch_tool("boom", {}, {"boom": boom}, None)
    assert "raised" in result["error"]
    assert "kaboom" in result["error"]


def test_dispatch_log_clips_a_huge_result(caplog):
    """read_raw_file returns whole sources. Logged verbatim, one dense run can
    turn over the whole 5 MB x 3 rotation and take the run history with it."""
    def big(**kwargs):
        return {"content": "x" * 50_000}

    logger = logging.getLogger("clip-result")
    with caplog.at_level(logging.INFO, logger="clip-result"):
        loop._dispatch_tool("big", {}, {"big": big}, logger)

    line = caplog.text
    assert len(line) < 5_000
    assert "+48000 chars" in line


def test_dispatch_log_clips_each_argument_separately(caplog):
    """The page name is the part worth keeping, and it must survive however the
    model happened to order the arguments."""
    logger = logging.getLogger("clip-args")
    args = {"content": "y" * 50_000, "name": "speakers-bureau"}
    with caplog.at_level(logging.INFO, logger="clip-args"):
        loop._dispatch_tool("write", args, {"write": lambda **kw: {"written": "ok"}}, logger)

    assert "speakers-bureau" in caplog.text
    assert len(caplog.text) < 5_000


def test_dispatch_log_leaves_a_short_call_intact(caplog):
    logger = logging.getLogger("clip-none")
    with caplog.at_level(logging.INFO, logger="clip-none"):
        loop._dispatch_tool("t", {"name": "a"}, {"t": lambda **kw: {"ok": True}}, logger)

    assert "tool_call t({'name': 'a'}) -> {\"ok\": true}" in caplog.text
    assert "chars]" not in caplog.text


def test_dispatch_drops_extra_kwarg_for_fixed_signature():
    def fixed(a):
        return {"a": a}

    result = loop._dispatch_tool("fixed", {"a": 1, "hallucinated": 2}, {"fixed": fixed}, None)
    assert result == {"a": 1}


def test_dispatch_keeps_extra_kwarg_for_var_kwargs():
    def kw(**kwargs):
        return kwargs

    result = loop._dispatch_tool("kw", {"a": 1, "b": 2}, {"kw": kw}, None)
    assert result == {"a": 1, "b": 2}


def test_dispatch_missing_required_arg_reports_error():
    def fixed(a):
        return {"a": a}

    result = loop._dispatch_tool("fixed", {}, {"fixed": fixed}, None)
    assert "error" in result


# --- _provider -------------------------------------------------------------


def test_provider_precedence(monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    assert loop._provider() == "ollama"
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    assert loop._provider() == "gemini"
    assert loop._provider("Ollama") == "ollama"  # explicit wins, normalized


def test_provider_rejects_a_typo_for_both_entrypoints(monkeypatch):
    """run_agent used to raise while complete_text fell through to Ollama, so a
    typo ran the ingest's sort step locally and then died on the first source."""
    monkeypatch.setenv("LLM_PROVIDER", "gemeni")
    for call in (
        lambda: loop.run_agent("s", "u", tools=[], dispatch={}),
        lambda: loop.complete_text("s", "u"),
    ):
        try:
            call()
            assert False, "expected ValueError"
        except ValueError as e:
            assert "gemeni" in str(e)


# --- model configuration ---------------------------------------------------


def test_ollama_model_raises_rather_than_defaulting(monkeypatch):
    """The old default was 'gemma4', which README calls out as a name Ollama
    does not resolve — five retries with backoff and no mention of .env."""
    monkeypatch.delenv("OLLAMA_MODEL", raising=False)
    try:
        loop._ollama_model()
        assert False, "expected RuntimeError"
    except RuntimeError as e:
        assert "OLLAMA_MODEL" in str(e)
        assert "config/.env" in str(e)


def test_ollama_model_explicit_beats_env(monkeypatch):
    monkeypatch.setenv("OLLAMA_MODEL", "from-env")
    assert loop._ollama_model() == "from-env"
    assert loop._ollama_model("explicit") == "explicit"


def test_gemini_model_raises_for_complete_text_too(monkeypatch):
    """complete_text skipped this check and built 'models/None:generateContent'."""
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    monkeypatch.delenv("GEMINI_MODEL", raising=False)
    try:
        loop.complete_text("s", "u", provider="gemini")
        assert False, "expected RuntimeError"
    except RuntimeError as e:
        assert "GEMINI_MODEL" in str(e)


# --- _to_function_declarations ---------------------------------------------


def test_to_function_declarations_drops_empty_parameters():
    tools = [
        {"function": {"name": "noargs", "parameters": {"type": "object", "properties": {}}}},
        {"function": {"name": "withargs", "parameters": {"type": "object",
                                                          "properties": {"x": {"type": "string"}}}}},
    ]
    decls = loop._to_function_declarations(tools)
    assert "parameters" not in decls[0]
    assert "parameters" in decls[1]


# --- _post_with_retry ------------------------------------------------------


def test_gemini_calls_honour_their_own_timeout(monkeypatch):
    """The Ollama path has honoured OLLAMA_TIMEOUT all along; this one took
    _post_with_retry's hardcoded 120 with no way to change it."""
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    monkeypatch.setenv("GEMINI_MODEL", "m")
    monkeypatch.setenv("GEMINI_TIMEOUT", "900")
    seen = {}

    def fake_post(url, json=None, headers=None, timeout=None):
        seen["timeout"] = timeout
        return FakeResp(200, json_data={"candidates": [{"content": {"parts": [{"text": "hi"}]}}]})

    monkeypatch.setattr(loop._session, "post", fake_post)
    loop.run_agent("s", "u", tools=[], dispatch={}, provider="gemini")
    assert seen["timeout"] == 900


def test_gemini_timeout_defaults_when_unset(monkeypatch):
    monkeypatch.delenv("GEMINI_TIMEOUT", raising=False)
    assert loop._gemini_timeout() == 120


def test_post_with_retry_success(monkeypatch):
    monkeypatch.setattr(loop._session, "post", _sequenced([FakeResp(200, json_data={"ok": 1})]))
    resp = loop._post_with_retry("http://x", {})
    assert resp.json() == {"ok": 1}


def test_post_with_retry_retries_on_429(monkeypatch):
    monkeypatch.setattr(loop.time, "sleep", lambda s: None)
    monkeypatch.setattr(
        loop._session, "post",
        _sequenced([FakeResp(429, headers={"Retry-After": "0"}), FakeResp(200, json_data={"ok": 2})]),
    )
    resp = loop._post_with_retry("http://x", {})
    assert resp.json() == {"ok": 2}


def test_post_with_retry_retries_network_error(monkeypatch):
    monkeypatch.setattr(loop.time, "sleep", lambda s: None)
    monkeypatch.setattr(
        loop._session, "post",
        _sequenced([requests.exceptions.ConnectionError("boom"), FakeResp(200, json_data={"ok": 3})]),
    )
    resp = loop._post_with_retry("http://x", {})
    assert resp.json() == {"ok": 3}


def test_post_with_retry_gives_up(monkeypatch):
    monkeypatch.setattr(loop.time, "sleep", lambda s: None)
    monkeypatch.setattr(loop._session, "post", _sequenced([FakeResp(503)] * loop._MAX_HTTP_ATTEMPTS))
    try:
        loop._post_with_retry("http://x", {})
        assert False, "expected HTTPError"
    except requests.exceptions.HTTPError:
        pass


# --- run_agent (ollama) ----------------------------------------------------


def test_run_agent_ollama_dispatches_then_returns_text(monkeypatch):
    payloads = [
        {"message": {"tool_calls": [{"function": {"name": "t", "arguments": {}}}]}},
        {"message": {"content": "final answer"}},
    ]
    monkeypatch.setattr(
        loop, "_post_with_retry",
        lambda *a, **k: FakeResp(200, json_data=payloads.pop(0)),
    )
    calls = []
    dispatch = {"t": lambda **kw: calls.append(1) or {"ok": True}}

    result = loop.run_agent("sys", "user", tools=[], dispatch=dispatch, provider="ollama")
    assert result == "final answer"
    assert calls == [1]


def test_run_agent_ollama_hits_iteration_cap(monkeypatch):
    monkeypatch.setattr(
        loop, "_post_with_retry",
        lambda *a, **k: FakeResp(
            200, json_data={"message": {"tool_calls": [{"function": {"name": "t", "arguments": {}}}]}}
        ),
    )
    dispatch = {"t": lambda **kw: {"ok": True}}
    result = loop.run_agent("sys", "user", tools=[], dispatch=dispatch, provider="ollama", max_iterations=2)
    assert result.startswith("[incomplete:")


def test_run_agent_rejects_unknown_provider():
    try:
        loop.run_agent("s", "u", tools=[], dispatch={}, provider="nope")
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_ollama_calls_send_num_ctx(monkeypatch):
    """num_ctx must reach Ollama on every call — context overflow is silent,
    so a dropped option degrades output with nothing to catch it."""
    monkeypatch.setenv("OLLAMA_NUM_CTX", "12345")
    seen = []

    def fake_post(url, payload, **kwargs):
        seen.append(payload)
        return FakeResp(200, json_data={"message": {"content": "done"}})

    monkeypatch.setattr(loop, "_post_with_retry", fake_post)

    loop.run_agent("sys", "user", tools=[], dispatch={}, provider="ollama")
    loop.complete_text("sys", "user", provider="ollama")

    assert len(seen) == 2
    for payload in seen:
        assert payload["options"]["num_ctx"] == 12345


def test_think_false_reaches_ollama(monkeypatch):
    """A stage that transcribes rather than judges turns reasoning off, and the
    field has to actually land on the request for that to mean anything — the
    reasoning block shares num_predict with the reply, so leaving it on is what
    cuts a page write short (see _run_ollama)."""
    seen = []

    def fake_post(url, payload, **kwargs):
        seen.append(payload)
        return FakeResp(200, json_data={"message": {"content": "done"}})

    monkeypatch.setattr(loop, "_post_with_retry", fake_post)

    loop.run_agent("sys", "user", tools=[], dispatch={}, provider="ollama", think=False)

    assert seen[0]["think"] is False


def test_think_is_omitted_by_default(monkeypatch):
    """Saying nothing must send nothing. A model with no notion of thinking
    should see exactly the request it saw before the parameter existed, rather
    than a field it has to ignore."""
    seen = []

    def fake_post(url, payload, **kwargs):
        seen.append(payload)
        return FakeResp(200, json_data={"message": {"content": "done"}})

    monkeypatch.setattr(loop, "_post_with_retry", fake_post)

    loop.run_agent("sys", "user", tools=[], dispatch={}, provider="ollama")
    loop.complete_text("sys", "user", provider="ollama")

    assert len(seen) == 2
    for payload in seen:
        assert "think" not in payload


def test_think_is_ignored_on_the_gemini_path(monkeypatch):
    """Gemini has no such field, and the caller setting it is describing the
    local model rather than stating a requirement. So the run proceeds without
    it — the weekly `wiki_lint --deep` pass shares run_agent and sets
    LLM_PROVIDER=gemini."""
    seen = []

    def fake_post(url, payload, **kwargs):
        seen.append(payload)
        return FakeResp(
            200,
            json_data={"candidates": [{"content": {"parts": [{"text": "done"}]}}]},
        )

    monkeypatch.setenv("GEMINI_API_KEY", "k")
    monkeypatch.setenv("GEMINI_MODEL", "m")
    monkeypatch.setattr(loop, "_post_with_retry", fake_post)

    result = loop.run_agent(
        "sys", "user", tools=[], dispatch={}, provider="gemini", think=False
    )

    assert result == "done"
    assert "think" not in seen[0]


def test_ollama_calls_send_num_predict(monkeypatch):
    """Ollama's default output length is unlimited, so an uncapped call can
    generate until the client times out — which cost a whole ingest run."""
    monkeypatch.setenv("OLLAMA_NUM_PREDICT", "777")
    seen = []

    def fake_post(url, payload, **kwargs):
        seen.append(payload)
        return FakeResp(200, json_data={"message": {"content": "done"}})

    monkeypatch.setattr(loop, "_post_with_retry", fake_post)

    loop.run_agent("sys", "user", tools=[], dispatch={}, provider="ollama")
    loop.complete_text("sys", "user", provider="ollama")

    assert len(seen) == 2
    for payload in seen:
        assert payload["options"]["num_predict"] == 777


def test_num_predict_defaults_when_unset(monkeypatch):
    monkeypatch.delenv("OLLAMA_NUM_PREDICT", raising=False)
    assert loop._ollama_options()["num_predict"] == 2000


def test_ollama_tool_results_carry_the_tool_name(monkeypatch):
    """Several calls in one turn arrive as an ordered list; without a name the
    model has to infer which result belongs to which call."""
    payloads = [
        {"message": {"tool_calls": [
            {"function": {"name": "a", "arguments": {}}},
            {"function": {"name": "b", "arguments": {}}},
        ]}},
        {"message": {"content": "done"}},
    ]
    seen = []

    def fake_post(url, payload, **kwargs):
        seen.append(payload["messages"])
        return FakeResp(200, json_data=payloads.pop(0))

    monkeypatch.setattr(loop, "_post_with_retry", fake_post)
    dispatch = {"a": lambda **kw: {"r": "a"}, "b": lambda **kw: {"r": "b"}}
    loop.run_agent("sys", "user", tools=[], dispatch=dispatch, provider="ollama")

    tool_messages = [m for m in seen[-1] if m.get("role") == "tool"]
    assert [m["tool_name"] for m in tool_messages] == ["a", "b"]


# --- context-window warning ------------------------------------------------


class _Recorder:
    def __init__(self):
        self.warnings = []
        self.infos = []

    def warning(self, msg):
        self.warnings.append(msg)

    def info(self, msg):
        self.infos.append(msg)


def _quiet_post(monkeypatch):
    monkeypatch.setattr(
        loop, "_post_with_retry",
        lambda *a, **k: FakeResp(200, json_data={"message": {"content": "done"}}),
    )


def test_warns_when_the_opening_prompt_already_fills_half_the_window(monkeypatch):
    """Overflow is silent — Ollama drops the oldest messages — so the only way
    anyone learns num_ctx is too small is if the run says so."""
    monkeypatch.setenv("OLLAMA_NUM_CTX", "1000")
    _quiet_post(monkeypatch)
    logger = _Recorder()

    loop.run_agent("x" * 8000, "u", tools=[], dispatch={}, provider="ollama", logger=logger)

    assert any("num_ctx=1000" in w for w in logger.warnings)
    assert any("OLLAMA_NUM_CTX" in w for w in logger.warnings)


def test_no_warning_when_the_prompt_fits_comfortably(monkeypatch):
    monkeypatch.setenv("OLLAMA_NUM_CTX", "32768")
    _quiet_post(monkeypatch)
    logger = _Recorder()

    loop.run_agent("short system", "short user", tools=[], dispatch={},
                   provider="ollama", logger=logger)

    assert logger.warnings == []


# --- output-cap warning ----------------------------------------------------


def test_warns_when_a_reply_stops_at_the_cap(monkeypatch):
    """A capped reply that says nothing trades a loud failure for a quiet
    wrong answer — a half-written page looks like a finished one."""
    monkeypatch.setenv("OLLAMA_NUM_PREDICT", "2000")
    monkeypatch.setattr(
        loop, "_post_with_retry",
        lambda *a, **k: FakeResp(200, json_data={
            "message": {"content": "cut off mid-"},
            "done_reason": "length",
            "prompt_eval_count": 23808,
        }),
    )
    logger = _Recorder()

    loop.run_agent("sys", "user", tools=[], dispatch={}, provider="ollama", logger=logger)

    assert any("num_predict=2000" in w for w in logger.warnings)
    assert any("23808" in w for w in logger.warnings)


def test_measures_the_real_prompt_every_iteration(monkeypatch):
    """The opening-prompt estimate cannot see growth from tool results, which
    is where all of it happens. prompt_eval_count is Ollama's own count."""
    monkeypatch.setenv("OLLAMA_NUM_CTX", "32768")
    payloads = [
        {"message": {"tool_calls": [{"function": {"name": "t", "arguments": {}}}]},
         "prompt_eval_count": 4000},
        {"message": {"content": "done"}, "prompt_eval_count": 9000},
    ]
    monkeypatch.setattr(
        loop, "_post_with_retry",
        lambda *a, **k: FakeResp(200, json_data=payloads.pop(0)),
    )
    logger = _Recorder()

    loop.run_agent("sys", "user", tools=[], dispatch={"t": lambda **kw: {}},
                   provider="ollama", logger=logger)

    sizes = [i for i in logger.infos if "prompt" in i]
    assert "prompt reached 4000 tokens on iteration 1" in sizes[0]
    assert "prompt reached 9000 tokens on iteration 2" in sizes[1]
    assert logger.warnings == []


def test_warns_when_the_measured_prompt_crosses_the_fraction(monkeypatch):
    """The failing run sat at 23808/32768 = 73% and said nothing."""
    monkeypatch.setenv("OLLAMA_NUM_CTX", "32768")
    monkeypatch.setattr(
        loop, "_post_with_retry",
        lambda *a, **k: FakeResp(200, json_data={
            "message": {"content": "done"}, "prompt_eval_count": 23808,
        }),
    )
    logger = _Recorder()

    loop.run_agent("sys", "user", tools=[], dispatch={}, provider="ollama", logger=logger)

    assert any("23808 tokens" in w and "73%" in w for w in logger.warnings)
    assert any("OLLAMA_NUM_CTX" in w for w in logger.warnings)


def test_prompt_size_is_silent_when_ollama_does_not_report_it(monkeypatch):
    monkeypatch.setattr(
        loop, "_post_with_retry",
        lambda *a, **k: FakeResp(200, json_data={"message": {"content": "done"}}),
    )
    logger = _Recorder()

    loop.run_agent("sys", "user", tools=[], dispatch={}, provider="ollama", logger=logger)

    assert logger.warnings == []
    assert not [i for i in logger.infos if "prompt" in i]


def test_no_cap_warning_on_a_normal_stop(monkeypatch):
    monkeypatch.setattr(
        loop, "_post_with_retry",
        lambda *a, **k: FakeResp(200, json_data={
            "message": {"content": "done"}, "done_reason": "stop",
        }),
    )
    logger = _Recorder()

    loop.run_agent("sys", "user", tools=[], dispatch={}, provider="ollama", logger=logger)

    assert logger.warnings == []


def test_a_truncated_reply_does_not_run_its_tool_call(monkeypatch):
    """A reply that stops at num_predict stops *mid-call*, and Ollama still
    hands back the arguments it managed to parse — so a half-written page
    arrives looking like a complete call. Observed 4 times in the 2026-08-20
    ingest: each was saved only by 'name' being emitted after 'content' and so
    falling off the end, which turned the truncation into a confusing
    missing-argument TypeError. Emitted the other way round it would have
    written the truncated body over a real page."""
    ran = []
    payloads = [
        {"message": {"tool_calls": [{"function": {
            "name": "write_wiki_page",
            "arguments": {"name": "ollama.md", "content": "# Ollama\n\ncut off mid-"},
        }}]}, "done_reason": "length"},
        {"message": {"content": "done"}, "done_reason": "stop"},
    ]
    sent = []

    def fake_post(url, payload, **kwargs):
        sent.append(payload)
        return FakeResp(200, json_data=payloads.pop(0))

    monkeypatch.setattr(loop, "_post_with_retry", fake_post)
    logger = _Recorder()

    loop.run_agent(
        "sys", "user", tools=[],
        dispatch={"write_wiki_page": lambda **kw: ran.append(kw) or {"written": "x"}},
        provider="ollama", logger=logger,
    )

    assert ran == [], "the truncated call must not reach the tool"
    fed_back = [m for m in sent[1]["messages"] if m.get("role") == "tool"]
    assert len(fed_back) == 1
    assert "cut off" in fed_back[0]["content"]
    assert "write_wiki_page" in fed_back[0]["content"]


def test_a_complete_reply_still_runs_its_tool_call(monkeypatch):
    """The truncation guard keys off done_reason, so the ordinary path has to
    keep working — including the 'stop' Ollama sends on a well-formed call."""
    ran = []
    payloads = [
        {"message": {"tool_calls": [{"function": {
            "name": "write_wiki_page",
            "arguments": {"name": "ollama.md", "content": "# Ollama\n"},
        }}]}, "done_reason": "stop"},
        {"message": {"content": "done"}, "done_reason": "stop"},
    ]
    monkeypatch.setattr(
        loop, "_post_with_retry",
        lambda *a, **k: FakeResp(200, json_data=payloads.pop(0)),
    )

    loop.run_agent(
        "sys", "user", tools=[],
        dispatch={"write_wiki_page": lambda **kw: ran.append(kw) or {"written": "x"}},
        provider="ollama",
    )

    assert ran == [{"name": "ollama.md", "content": "# Ollama\n"}]


def test_a_reply_cut_off_before_any_tool_call_is_not_a_final_answer(monkeypatch):
    """The other half of the truncation case: the cut lands *before* the model
    emits anything parseable, so Ollama returns done_reason=length with no tool
    calls and no content. Verified live on 2026-08-20 against gemma4:26b-mlx at
    num_predict 600 and 1200. Returning that empty string as the final answer
    made wiki_ingest log 'produced no wiki writes' and burn a whole retry of the
    source, so the loop has to nudge and keep going instead."""
    payloads = [
        {"message": {"content": ""}, "done_reason": "length"},
        {"message": {"content": "done"}, "done_reason": "stop"},
    ]
    sent = []

    def fake_post(url, payload, **kwargs):
        sent.append(payload)
        return FakeResp(200, json_data=payloads.pop(0))

    monkeypatch.setattr(loop, "_post_with_retry", fake_post)
    logger = _Recorder()

    answer = loop.run_agent(
        "sys", "user", tools=[], dispatch={}, provider="ollama", logger=logger,
    )

    assert answer == "done", "the truncated turn must not end the loop"
    # The payload holds the live message list, so read the user turns rather
    # than the tail: by now the second reply has been appended too.
    said = [m["content"] for m in sent[1]["messages"] if m.get("role") == "user"]
    assert len(said) == 2, "the nudge should be the only message added"
    assert "cut off" in said[1]


def test_a_run_that_only_ever_truncates_reports_itself_incomplete(monkeypatch):
    """A caller has to be able to tell a truncated run from a genuine no-op, so
    the nudge must not turn into a silent empty answer when it never lands."""
    monkeypatch.setattr(
        loop, "_post_with_retry",
        lambda *a, **k: FakeResp(200, json_data={
            "message": {"content": ""}, "done_reason": "length",
        }),
    )

    answer = loop.run_agent(
        "sys", "user", tools=[], dispatch={}, provider="ollama", max_iterations=3,
    )

    assert answer.startswith(loop.INCOMPLETE_PREFIX)


def test_an_empty_reply_that_stopped_normally_is_still_the_answer(monkeypatch):
    """The nudge keys off done_reason, so an ordinary empty answer must still
    come straight back rather than costing another round trip."""
    calls = []

    def fake_post(url, payload, **kwargs):
        calls.append(payload)
        return FakeResp(200, json_data={
            "message": {"content": ""}, "done_reason": "stop",
        })

    monkeypatch.setattr(loop, "_post_with_retry", fake_post)

    answer = loop.run_agent("sys", "user", tools=[], dispatch={}, provider="ollama")

    assert answer == ""
    assert len(calls) == 1


# --- _retry_delay ----------------------------------------------------------


def test_retry_after_as_an_http_date_does_not_crash_the_run():
    """RFC 9110 lets Retry-After be a date rather than a count of seconds, and
    float() on one raises. That ValueError was raised inside the retry loop,
    where nothing catches it, so a header whose whole purpose is to slow the
    client down instead killed the source with an error naming neither the
    header nor the server."""
    delay = loop._retry_delay("Wed, 21 Oct 2015 07:28:00 GMT", attempt=0)
    assert 1.0 <= delay <= 2.0, "should fall back to the computed backoff"


def test_retry_after_is_capped_rather_than_obeyed():
    """The computed backoff has always been capped at _MAX_BACKOFF_S; a
    server-supplied one was not. budget.before_retry does not save this —
    wiki_query.py starts no budget and wiki_lint starts one only for --deep —
    so an interactive question would have sat silently for the full hour."""
    delay = loop._retry_delay("3600", attempt=0)
    assert delay <= loop._MAX_BACKOFF_S + 1


def test_retry_after_is_honoured_when_it_is_shorter_than_the_cap():
    """Capping must not become ignoring: a server asking for 5 seconds gets 5,
    not the cap and not the exponential backoff."""
    delay = loop._retry_delay("5", attempt=4)
    assert 5.0 <= delay <= 6.0


def test_a_negative_retry_after_never_becomes_a_negative_sleep():
    """time.sleep raises on a negative number, so a malformed header must not
    reach it."""
    assert loop._retry_delay("-100", attempt=0) >= 0.0


def test_computed_backoff_is_capped_too():
    """The ceiling belongs to the delay, not to who chose it."""
    assert loop._retry_delay(None, attempt=20) <= loop._MAX_BACKOFF_S + 1


def test_a_429_with_a_date_retry_after_still_retries(monkeypatch):
    """The end-to-end shape of the bug: the crash happened inside
    _post_with_retry, so the fix has to be visible from there."""
    slept = []
    monkeypatch.setattr(loop.time, "sleep", slept.append)
    monkeypatch.setattr(
        loop._session, "post",
        _sequenced([
            FakeResp(429, headers={"Retry-After": "Wed, 21 Oct 2015 07:28:00 GMT"}),
            FakeResp(200, json_data={"ok": 1}),
        ]),
    )

    resp = loop._post_with_retry("http://x", {})

    assert resp.json() == {"ok": 1}
    assert len(slept) == 1 and slept[0] <= loop._MAX_BACKOFF_S + 1


# --- Gemini truncation -----------------------------------------------------


def test_a_truncated_gemini_reply_does_not_run_its_tool_call(monkeypatch):
    """The same cut the Ollama path has guarded since 2026-08-20, under the
    other provider's name for it. Gemini reports finishReason=MAX_TOKENS and
    still hands back the part of the call it had written, so a half-written
    page arrives looking complete. wiki_lint --deep is read-only today, but
    LLM_PROVIDER=gemini is a documented opt-in for a full vault rebuild and
    that path writes."""
    ran = []
    payloads = [
        {"candidates": [{
            "finishReason": "MAX_TOKENS",
            "content": {"parts": [{"functionCall": {
                "name": "write_wiki_page",
                "args": {"name": "colima.md", "content": "# Colima\n\ncut off mid-"},
            }}]},
        }]},
        {"candidates": [{"content": {"parts": [{"text": "done"}]}}]},
    ]
    sent = []

    def fake_post(url, payload, **kwargs):
        sent.append(payload)
        return FakeResp(200, json_data=payloads.pop(0))

    monkeypatch.setenv("GEMINI_API_KEY", "k")
    monkeypatch.setenv("GEMINI_MODEL", "m")
    monkeypatch.setattr(loop, "_post_with_retry", fake_post)

    answer = loop.run_agent(
        "sys", "user", tools=[],
        dispatch={"write_wiki_page": lambda **kw: ran.append(kw) or {"written": "x"}},
        provider="gemini",
    )

    assert ran == [], "the truncated call must not reach the tool"
    assert answer == "done"
    # The payload holds the live contents list, so scan it rather than take
    # the tail: by now the second reply has been appended too.
    fed_back = [
        p["functionResponse"]
        for turn in sent[1]["contents"]
        for p in turn["parts"]
        if "functionResponse" in p
    ]
    assert len(fed_back) == 1
    assert fed_back[0]["name"] == "write_wiki_page"
    assert "cut off" in fed_back[0]["response"]["error"]


def test_a_complete_gemini_reply_still_runs_its_tool_call(monkeypatch):
    """The guard keys off finishReason, so the ordinary path — including the
    STOP Gemini sends on a well-formed call — has to keep working."""
    ran = []
    payloads = [
        {"candidates": [{
            "finishReason": "STOP",
            "content": {"parts": [{"functionCall": {
                "name": "write_wiki_page",
                "args": {"name": "colima.md", "content": "# Colima\n"},
            }}]},
        }]},
        {"candidates": [{"content": {"parts": [{"text": "done"}]}}]},
    ]

    monkeypatch.setenv("GEMINI_API_KEY", "k")
    monkeypatch.setenv("GEMINI_MODEL", "m")
    monkeypatch.setattr(
        loop, "_post_with_retry",
        lambda *a, **k: FakeResp(200, json_data=payloads.pop(0)),
    )

    loop.run_agent(
        "sys", "user", tools=[],
        dispatch={"write_wiki_page": lambda **kw: ran.append(kw) or {"written": "x"}},
        provider="gemini",
    )

    assert ran == [{"name": "colima.md", "content": "# Colima\n"}]


def test_a_gemini_reply_cut_off_before_any_tool_call_is_not_a_final_answer(monkeypatch):
    """The other half, in Gemini's message shape: the cut lands before anything
    parseable, so returning the empty text as the answer would read downstream
    as a model that chose to do nothing."""
    payloads = [
        {"candidates": [{"finishReason": "MAX_TOKENS", "content": {"parts": []}}]},
        {"candidates": [{"content": {"parts": [{"text": "done"}]}}]},
    ]
    sent = []

    def fake_post(url, payload, **kwargs):
        sent.append(payload)
        return FakeResp(200, json_data=payloads.pop(0))

    monkeypatch.setenv("GEMINI_API_KEY", "k")
    monkeypatch.setenv("GEMINI_MODEL", "m")
    monkeypatch.setattr(loop, "_post_with_retry", fake_post)

    answer = loop.run_agent(
        "sys", "user", tools=[], dispatch={}, provider="gemini",
    )

    assert answer == "done", "the truncated turn must not end the loop"
    # The payload holds the live contents list, so read the user turns rather
    # than the tail: by now the second reply has been appended too.
    said = [t for t in sent[1]["contents"] if t["role"] == "user"]
    assert len(said) == 2, "the nudge should be the only turn added"
    assert "cut off" in said[1]["parts"][0]["text"], "Gemini wants parts, not content"
