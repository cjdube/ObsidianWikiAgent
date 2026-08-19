"""Tool-calling agent loop against a local Ollama model or the Gemini API.

Drives a conversation: send messages + tool schemas to the model, dispatch any
tool calls to local Python functions, feed results back, repeat until the
model returns a final text response or the iteration cap is hit.

The loop itself is provider-agnostic — only the wire format differs, so each
provider owns its own request/history translation and shares the tool-dispatch
step. Ollama is the default: the scheduled daily ingests run locally. Gemini is
opt-in per run (LLM_PROVIDER=gemini) for the work where synthesis quality
decides whether the output is worth reading — one-off bulk jobs like a full
vault rebuild, and the weekly wiki_lint --deep judgment pass, which sets it in
its .plist. Note that anything it reads leaves the machine.
"""

import inspect
import json
import logging
import os
import random
import time
from typing import Callable, Optional

import requests

from agent import budget

MAX_TOOL_ITERATIONS = 6

DEFAULT_PROVIDER = "ollama"
GEMINI_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta"

# The Gemini free tier rate-limits aggressively, and a vault rebuild is
# hundreds of calls; without backoff a single 429 would fail the whole source
# and burn the retry budget in wiki_ingest.
_RETRY_STATUSES = (429, 500, 502, 503, 504)
_MAX_HTTP_ATTEMPTS = 5


def _provider(explicit: Optional[str] = None) -> str:
    name = (explicit or os.getenv("LLM_PROVIDER") or DEFAULT_PROVIDER).strip().lower()
    if name not in ("ollama", "gemini"):
        # Checked in one place so complete_text and run_agent agree. They used
        # to differ: run_agent raised, while complete_text fell through to its
        # Ollama branch, so a typo in LLM_PROVIDER ran the ingest's sort step
        # locally and then failed on the first source with an unrelated-looking
        # error.
        raise ValueError(f"unknown provider '{name}' (expected 'ollama' or 'gemini')")
    return name


def _ollama_model(explicit: Optional[str] = None) -> str:
    """The Ollama tag to call, or a message saying how to set one.

    No default. The obvious one — a bare family name like 'gemma4' — is not a
    tag Ollama resolves, so it buys nothing but a 404 five times over with
    backoff, several minutes after the run started and with no mention of
    config/.env in the error. The Gemini path already raises with a remedy
    (see _gemini_key); this matches it.
    """
    model = explicit or os.getenv("OLLAMA_MODEL")
    if not model:
        raise RuntimeError(
            "OLLAMA_MODEL is not set — add the exact tag `ollama list` prints "
            "to config/.env (e.g. OLLAMA_MODEL=gemma4:26b-mlx). A bare family "
            "name is not a tag Ollama resolves."
        )
    return model


def _incomplete(max_iterations: int, logger: Optional[logging.Logger]) -> str:
    """The both-providers answer when the loop runs out of iterations."""
    if logger:
        logger.warning(
            f"agent loop exceeded max_iterations={max_iterations} without a "
            "final answer — returning best-effort partial result"
        )
    return (
        f"[incomplete: hit max_iterations={max_iterations} tool calls without "
        "reaching a final answer]"
    )


def _dispatch_tool(
    fn_name: str,
    fn_args: dict,
    dispatch: dict[str, Callable[..., dict]],
    logger: Optional[logging.Logger],
) -> dict:
    """Run one tool call and return its result dict. Never raises: a failing
    tool is reported back to the model as an error result so it can recover,
    rather than killing the run."""
    fn = dispatch.get(fn_name)
    if fn is None:
        result = {"error": f"unknown tool '{fn_name}'"}
    else:
        try:
            params = inspect.signature(fn).parameters
            takes_var_kwargs = any(
                p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()
            )
            # Local models occasionally hallucinate an extra argument that
            # isn't in the tool's schema (e.g. passing update_index a 'name'
            # kwarg copied from the sibling write_wiki_page tool). Drop
            # anything the function doesn't declare rather than failing the
            # call outright — a genuinely missing required argument still
            # raises normally and is reported back to the model.
            if not takes_var_kwargs:
                fn_args = {k: v for k, v in fn_args.items() if k in params}
            result = fn(**fn_args)
        except Exception as e:
            result = {"error": f"tool '{fn_name}' raised: {e}"}
    if logger:
        logger.info(f"tool_call {fn_name}({fn_args}) -> {json.dumps(result)}")
    return result


def _post_with_retry(
    url: str,
    payload: dict,
    headers: Optional[dict] = None,
    timeout: int = 120,
    logger: Optional[logging.Logger] = None,
) -> requests.Response:
    """POST, retrying transient failures with exponential backoff + jitter.

    Honors Retry-After when the server sends it (Gemini does on quota errors).
    Network-level failures (timeouts, connection errors — routine for a large
    local model under load) get the same backoff as retryable HTTP statuses,
    rather than failing the call on the first slow response.

    Every wait here is also charged against the run budget (agent/budget.py).
    _MAX_HTTP_ATTEMPTS caps one call's retries, which is not the same as
    capping the run's: this helper is invoked once per loop iteration, per
    ingest attempt, per source, so a wedged server stays under this cap while
    the totals run to hours. budget.before_retry is what actually stops that.
    """
    for attempt in range(1, _MAX_HTTP_ATTEMPTS + 1):
        budget.check("model request")
        try:
            resp = requests.post(
                url, json=payload, headers=headers,
                timeout=budget.clamp_timeout(timeout),
            )
        except requests.exceptions.RequestException as e:
            if attempt == _MAX_HTTP_ATTEMPTS:
                raise
            delay = min(2 ** attempt, 60) + random.uniform(0, 1)
            reason = f"{e.__class__.__name__} from model API"
            budget.before_retry(delay, reason)
            if logger:
                logger.warning(
                    f"{reason} — retrying in "
                    f"{delay:.1f}s (attempt {attempt}/{_MAX_HTTP_ATTEMPTS})"
                )
            time.sleep(delay)
            continue
        if resp.status_code not in _RETRY_STATUSES:
            resp.raise_for_status()
            return resp
        if attempt == _MAX_HTTP_ATTEMPTS:
            resp.raise_for_status()
        retry_after = resp.headers.get("Retry-After")
        delay = float(retry_after) if retry_after else min(2 ** attempt, 60)
        delay += random.uniform(0, 1)
        reason = f"HTTP {resp.status_code} from model API"
        budget.before_retry(delay, reason)
        if logger:
            logger.warning(
                f"{reason} — retrying in "
                f"{delay:.1f}s (attempt {attempt}/{_MAX_HTTP_ATTEMPTS})"
            )
        time.sleep(delay)
    raise RuntimeError("unreachable")


def run_agent(
    system_prompt: str,
    user_prompt: str,
    tools: list[dict],
    dispatch: dict[str, Callable[..., dict]],
    model: str = None,
    host: str = None,
    logger: Optional[logging.Logger] = None,
    max_iterations: int = MAX_TOOL_ITERATIONS,
    provider: str = None,
) -> str:
    """Run the tool-calling loop and return the model's final text response.

    tools: list of OpenAI-style tool schemas (see agent/wiki_tools.py TOOL_SCHEMA).
           Translated per-provider as needed.
    dispatch: {function_name: callable} mapping — callable takes the parsed
              arguments dict via **kwargs and returns a JSON-serializable dict.
    max_iterations: raise for workflows that legitimately need many tool
              calls (e.g. wiki ingest, which can touch 10-15 pages per source).
    provider: 'ollama' (default) or 'gemini'; falls back to $LLM_PROVIDER.
    """
    name = _provider(provider)
    if name == "gemini":
        return _run_gemini(
            system_prompt, user_prompt, tools, dispatch, model, logger, max_iterations
        )
    return _run_ollama(
        system_prompt, user_prompt, tools, dispatch, model, host, logger, max_iterations
    )


def _ollama_options() -> dict:
    """The `options` block sent with every Ollama call.

    `num_ctx` is pinned rather than left to the server default because the
    consequence of getting it wrong is silent. A run's context is the vault's
    RULES.md, plus the whole of wiki/index.md, plus every tool result the loop
    accumulates — measured at ~22K tokens on a 228-page vault and growing with
    it. Overflow doesn't error: Ollama drops the oldest messages, so the model
    answers from a truncated view and the output just quietly gets worse.
    32768 matches the default of the Ollama this was built against (0.32.3);
    naming it here means an upstream change to that default cannot start
    truncating runs without anyone noticing.

    `num_predict` caps how much one reply may be. Ollama's default for
    /api/chat is unlimited, so a repetition loop generates until the context
    fills or the client hangs up. On 2026-08-19 that cost a whole ingest run:
    one call ran away, four ReadTimeout retries resent the identical prompt
    (deterministic, not a flake) at exactly 10 minutes each, and the 45-minute
    run budget expired with 0/2 sources done. A step here is a tool call or a
    page write; neither legitimately needs 2000 tokens, so a runaway now ends
    in about a minute and the run keeps its budget. Truncation at the cap is
    logged (see _warn_if_reply_hit_the_cap) rather than silent.
    """
    return {
        "num_ctx": int(os.getenv("OLLAMA_NUM_CTX", "32768")),
        "num_predict": int(os.getenv("OLLAMA_NUM_PREDICT", "2000")),
    }


def _warn_if_reply_hit_the_cap(
    data: dict, logger: Optional[logging.Logger]
) -> None:
    """Say something when a reply stopped because it hit num_predict.

    Without this the cap trades a loud failure for a quiet wrong answer: a
    half-written page or a truncated tool call looks like a normal response.
    Ollama reports the reason as done_reason == 'length'.
    """
    if logger is None or data.get("done_reason") != "length":
        return
    logger.warning(
        f"Model reply stopped at num_predict="
        f"{_ollama_options()['num_predict']} (prompt was "
        f"{data.get('prompt_eval_count', '?')} tokens) — the reply is cut off. "
        f"Raise OLLAMA_NUM_PREDICT in config/.env if this was a legitimate "
        f"long answer rather than a repetition loop."
    )


# Rough bytes-per-token for prose and markdown. Only ever decides whether to
# log a warning, so being 20% out costs nothing.
#
# It is more than 20% out: the 2026-08-19 prompt measured 67,483 chars for the
# 23,808 tokens Ollama counted, i.e. ~2.8, because wiki-links and code
# tokenize denser than prose. Left at 4 deliberately — this constant now only
# feeds the pre-flight guess in _warn_if_context_is_tight, and _log_prompt_size
# reports Ollama's own count from the first response onward. Changing it would
# retune a guess that a measurement has already replaced.
_CHARS_PER_TOKEN = 4

# Fraction of num_ctx at which a measured prompt is worth a warning. The prompt
# still has to hold the reply (num_predict) and at least one more tool result,
# and on this vault a single page read is ~4,500 tokens — so at 70% of a 32768
# window the ~9,800 tokens left is one big read away from overflow. The run
# that prompted all this sat at 73% with nothing in any log to say so.
_CONTEXT_WARN_FRACTION = 0.70


def _log_prompt_size(
    data: dict,
    iteration: int,
    num_ctx: int,
    logger: Optional[logging.Logger],
) -> None:
    """Record how big the prompt actually was, as Ollama itself counted it.

    _warn_if_context_is_tight can only see the opening prompt, which is the one
    part of a run that does not grow. That is why the 2026-08-19 run passed it
    in silence: system + user was ~3,500 tokens, but by the twelfth call the
    accumulated tool results had taken the prompt to 23,808 — 73% of the
    window, and 38% of it one wiki page that had been read twice.

    prompt_eval_count comes back on every non-streaming response for free, so
    this is a measurement rather than an estimate, and it lands once per
    iteration — which is exactly where the growth happens.
    """
    if logger is None:
        return
    count = data.get("prompt_eval_count")
    if count is None:
        # Not every Ollama build reports it, and there is nothing to say if so.
        return
    fill = count / num_ctx
    message = (
        f"prompt {count} tokens on iteration {iteration} "
        f"({fill:.0%} of num_ctx={num_ctx})"
    )
    if fill >= _CONTEXT_WARN_FRACTION:
        logger.warning(
            f"{message} — overflow is silent (Ollama drops the oldest "
            f"messages). Raise OLLAMA_NUM_CTX in config/.env, or cut what the "
            f"loop feeds back in."
        )
    else:
        logger.info(message)


def _warn_if_context_is_tight(
    system_prompt: str,
    user_prompt: str,
    num_ctx: int,
    logger: Optional[logging.Logger],
) -> None:
    """Say something when the opening prompt already fills much of the window.

    The overflow itself is silent by design (see _ollama_options), and that is
    the whole problem: Ollama drops the oldest messages, so a vault's RULES.md
    leaves the conversation first and the model finishes the run answering from
    a truncated view with nothing in any log to say so.

    Measured before any tool result is appended, because that is the part of a
    run that cannot shrink. An ingest then adds the source, every page it reads
    and every page it writes — so a start that already fills half the window
    will not fit, and the half is a deliberately early alarm rather than a
    prediction.
    """
    if logger is None:
        return
    estimate = (len(system_prompt) + len(user_prompt)) // _CHARS_PER_TOKEN
    if estimate * 2 > num_ctx:
        logger.warning(
            f"Prompt is ~{estimate} tokens before any tool results, against "
            f"num_ctx={num_ctx}. Ollama drops the oldest messages on overflow "
            f"rather than failing — raise OLLAMA_NUM_CTX in config/.env."
        )


def _run_ollama(
    system_prompt: str,
    user_prompt: str,
    tools: list[dict],
    dispatch: dict[str, Callable[..., dict]],
    model: str,
    host: str,
    logger: Optional[logging.Logger],
    max_iterations: int,
) -> str:
    model = _ollama_model(model)
    host = host or os.getenv("OLLAMA_HOST", "http://localhost:11434")
    timeout = int(os.getenv("OLLAMA_TIMEOUT", "300"))
    options = _ollama_options()
    _warn_if_context_is_tight(system_prompt, user_prompt, options["num_ctx"], logger)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    for iteration in range(max_iterations):
        resp = _post_with_retry(
            f"{host}/api/chat",
            {
                "model": model,
                "messages": messages,
                "tools": tools,
                "stream": False,
                "options": options,
            },
            timeout=timeout,
            logger=logger,
        )
        data = resp.json()
        _log_prompt_size(data, iteration + 1, options["num_ctx"], logger)
        _warn_if_reply_hit_the_cap(data, logger)
        message = data["message"]
        messages.append(message)

        tool_calls = message.get("tool_calls") or []
        if not tool_calls:
            return message.get("content", "")

        for call in tool_calls:
            fn_name = call["function"]["name"]
            fn_args = call["function"].get("arguments", {})
            result = _dispatch_tool(fn_name, fn_args, dispatch, logger)
            # tool_name matters once a turn carries several calls: without it
            # the model gets an ordered list of unlabelled results and has to
            # infer which is which. The Gemini path has always named them.
            messages.append({
                "role": "tool",
                "tool_name": fn_name,
                "content": json.dumps(result),
            })

    return _incomplete(max_iterations, logger)


def _gemini_key() -> str:
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        raise RuntimeError(
            "GEMINI_API_KEY is not set — add it to config/.env to use "
            "LLM_PROVIDER=gemini."
        )
    return key


def _gemini_model(explicit: Optional[str] = None) -> str:
    model = explicit or os.getenv("GEMINI_MODEL")
    if not model:
        raise RuntimeError(
            "GEMINI_MODEL is not set — add it to config/.env (see "
            "`python -m agent.loop list-models` for what your key can reach)."
        )
    return model


def _to_function_declarations(tools: list[dict]) -> list[dict]:
    """OpenAI-style tool schemas -> Gemini functionDeclarations.

    The inner 'function' object is already nearly Gemini's shape. The one
    incompatibility: Gemini rejects an empty parameters object, which is how
    the no-argument tools (list_wiki_pages, read_index) declare themselves —
    those must omit 'parameters' entirely.
    """
    declarations = []
    for tool in tools:
        fn = dict(tool["function"])
        params = fn.get("parameters") or {}
        if not params.get("properties"):
            fn.pop("parameters", None)
        declarations.append(fn)
    return declarations


def _run_gemini(
    system_prompt: str,
    user_prompt: str,
    tools: list[dict],
    dispatch: dict[str, Callable[..., dict]],
    model: str,
    logger: Optional[logging.Logger],
    max_iterations: int,
) -> str:
    model = _gemini_model(model)

    url = f"{GEMINI_ENDPOINT}/models/{model}:generateContent"
    headers = {"x-goog-api-key": _gemini_key()}
    payload_base = {
        "systemInstruction": {"parts": [{"text": system_prompt}]},
        "tools": [{"functionDeclarations": _to_function_declarations(tools)}],
    }

    contents = [{"role": "user", "parts": [{"text": user_prompt}]}]

    for iteration in range(max_iterations):
        resp = _post_with_retry(
            url, {**payload_base, "contents": contents}, headers=headers, logger=logger
        )
        data = resp.json()

        candidates = data.get("candidates") or []
        if not candidates:
            # Prompt blocked by a safety filter, or an empty generation.
            reason = data.get("promptFeedback", {}).get("blockReason", "no candidates")
            raise RuntimeError(f"Gemini returned no candidates: {reason}")

        content = candidates[0].get("content") or {}
        parts = content.get("parts") or []
        contents.append({"role": "model", "parts": parts})

        calls = [p["functionCall"] for p in parts if "functionCall" in p]
        if not calls:
            return "".join(p.get("text", "") for p in parts)

        responses = []
        for call in calls:
            fn_name = call["name"]
            fn_args = call.get("args") or {}
            result = _dispatch_tool(fn_name, fn_args, dispatch, logger)
            responses.append(
                {"functionResponse": {"name": fn_name, "response": result}}
            )
        contents.append({"role": "user", "parts": responses})

    return _incomplete(max_iterations, logger)


def complete_text(
    system_prompt: str,
    user_prompt: str,
    model: str = None,
    host: str = None,
    provider: str = None,
) -> str:
    """Single-turn, tool-free completion — for tasks where the caller
    assembles the surrounding structure itself rather than trusting the
    model to produce it."""
    name = _provider(provider)

    if name == "gemini":
        # Same validation _run_gemini does. Skipping it here built a URL of
        # 'models/None:generateContent' and returned an opaque 404.
        model = _gemini_model(model)
        resp = _post_with_retry(
            f"{GEMINI_ENDPOINT}/models/{model}:generateContent",
            {
                "systemInstruction": {"parts": [{"text": system_prompt}]},
                "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
            },
            headers={"x-goog-api-key": _gemini_key()},
        )
        candidates = resp.json().get("candidates") or []
        if not candidates:
            return ""
        parts = (candidates[0].get("content") or {}).get("parts") or []
        return "".join(p.get("text", "") for p in parts).strip()

    model = _ollama_model(model)
    host = host or os.getenv("OLLAMA_HOST", "http://localhost:11434")
    resp = _post_with_retry(
        f"{host}/api/chat",
        {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
            "options": _ollama_options(),
        },
        timeout=int(os.getenv("OLLAMA_TIMEOUT", "300")),
    )
    data = resp.json()
    # No logger is passed in here, so use the module's — a truncated
    # single-turn answer is exactly as silent as a truncated loop turn.
    _warn_if_reply_hit_the_cap(data, logging.getLogger(__name__))
    return data["message"].get("content", "").strip()


def list_gemini_models() -> list[str]:
    """Model ids the configured key can actually reach, for generateContent.

    Model availability moves faster than this code; ask the API rather than
    hardcoding an id that may not exist.
    """
    resp = requests.get(
        f"{GEMINI_ENDPOINT}/models",
        headers={"x-goog-api-key": _gemini_key()},
        timeout=30,
    )
    resp.raise_for_status()
    return sorted(
        m["name"].removeprefix("models/")
        for m in resp.json().get("models", [])
        if "generateContent" in m.get("supportedGenerationMethods", [])
    )


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "list-models":
        for m in list_gemini_models():
            print(m)
    else:
        print("usage: python -m agent.loop list-models", file=sys.stderr)
        sys.exit(1)
