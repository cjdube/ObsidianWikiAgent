"""Tests for the provider-agnostic agent loop in agent/loop.py.

No network: requests.post (and the retry helper, for run_agent) is mocked, and
time.sleep is neutralized so backoff paths run instantly.
"""

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


def test_post_with_retry_success(monkeypatch):
    monkeypatch.setattr(loop.requests, "post", _sequenced([FakeResp(200, json_data={"ok": 1})]))
    resp = loop._post_with_retry("http://x", {})
    assert resp.json() == {"ok": 1}


def test_post_with_retry_retries_on_429(monkeypatch):
    monkeypatch.setattr(loop.time, "sleep", lambda s: None)
    monkeypatch.setattr(
        loop.requests, "post",
        _sequenced([FakeResp(429, headers={"Retry-After": "0"}), FakeResp(200, json_data={"ok": 2})]),
    )
    resp = loop._post_with_retry("http://x", {})
    assert resp.json() == {"ok": 2}


def test_post_with_retry_retries_network_error(monkeypatch):
    monkeypatch.setattr(loop.time, "sleep", lambda s: None)
    monkeypatch.setattr(
        loop.requests, "post",
        _sequenced([requests.exceptions.ConnectionError("boom"), FakeResp(200, json_data={"ok": 3})]),
    )
    resp = loop._post_with_retry("http://x", {})
    assert resp.json() == {"ok": 3}


def test_post_with_retry_gives_up(monkeypatch):
    monkeypatch.setattr(loop.time, "sleep", lambda s: None)
    monkeypatch.setattr(loop.requests, "post", _sequenced([FakeResp(503)] * loop._MAX_HTTP_ATTEMPTS))
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
    assert "prompt 4000 tokens on iteration 1" in sizes[0]
    assert "prompt 9000 tokens on iteration 2" in sizes[1]
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
