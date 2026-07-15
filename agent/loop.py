"""Tool-calling agent loop against a local Ollama model or the Gemini API.

Drives a conversation: send messages + tool schemas to the model, dispatch any
tool calls to local Python functions, feed results back, repeat until the
model returns a final text response or the iteration cap is hit.

The loop itself is provider-agnostic — only the wire format differs, so each
provider owns its own request/history translation and shares the tool-dispatch
step. Ollama is the default: the scheduled daily ingests run locally, and
Gemini is opt-in per run (LLM_PROVIDER=gemini) for one-off bulk work like a
full vault rebuild, where synthesis quality matters more than cost.
"""

import json
import logging
import os
import random
import time
from pathlib import Path
from typing import Callable, Optional

import requests
from dotenv import load_dotenv

_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_ROOT / "config" / ".env")

MAX_TOOL_ITERATIONS = 6

DEFAULT_PROVIDER = "ollama"
GEMINI_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta"

# The Gemini free tier rate-limits aggressively, and a vault rebuild is
# hundreds of calls; without backoff a single 429 would fail the whole source
# and burn the retry budget in wiki_ingest.
_RETRY_STATUSES = (429, 500, 502, 503, 504)
_MAX_HTTP_ATTEMPTS = 5


def _provider(explicit: Optional[str] = None) -> str:
    return (explicit or os.getenv("LLM_PROVIDER") or DEFAULT_PROVIDER).strip().lower()


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
    """
    for attempt in range(1, _MAX_HTTP_ATTEMPTS + 1):
        resp = requests.post(url, json=payload, headers=headers, timeout=timeout)
        if resp.status_code not in _RETRY_STATUSES:
            resp.raise_for_status()
            return resp
        if attempt == _MAX_HTTP_ATTEMPTS:
            resp.raise_for_status()
        retry_after = resp.headers.get("Retry-After")
        delay = float(retry_after) if retry_after else min(2 ** attempt, 60)
        delay += random.uniform(0, 1)
        if logger:
            logger.warning(
                f"HTTP {resp.status_code} from model API — retrying in "
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
    if name == "ollama":
        return _run_ollama(
            system_prompt, user_prompt, tools, dispatch, model, host, logger, max_iterations
        )
    raise ValueError(f"unknown provider '{name}' (expected 'ollama' or 'gemini')")


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
    model = model or os.getenv("OLLAMA_MODEL", "gemma4")
    host = host or os.getenv("OLLAMA_HOST", "http://localhost:11434")

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    for iteration in range(max_iterations):
        resp = _post_with_retry(
            f"{host}/api/chat",
            {"model": model, "messages": messages, "tools": tools, "stream": False},
            logger=logger,
        )
        data = resp.json()
        message = data["message"]
        messages.append(message)

        tool_calls = message.get("tool_calls") or []
        if not tool_calls:
            return message.get("content", "")

        for call in tool_calls:
            fn_name = call["function"]["name"]
            fn_args = call["function"].get("arguments", {})
            result = _dispatch_tool(fn_name, fn_args, dispatch, logger)
            messages.append({"role": "tool", "content": json.dumps(result)})

    raise RuntimeError(f"agent loop exceeded max_iterations={max_iterations} without a final answer")


def _gemini_key() -> str:
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        raise RuntimeError(
            "GEMINI_API_KEY is not set — add it to config/.env to use "
            "LLM_PROVIDER=gemini."
        )
    return key


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
    model = model or os.getenv("GEMINI_MODEL")
    if not model:
        raise RuntimeError(
            "GEMINI_MODEL is not set — add it to config/.env (see "
            "`python -m agent.loop list-models` for what your key can reach)."
        )

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

    raise RuntimeError(f"agent loop exceeded max_iterations={max_iterations} without a final answer")


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
        model = model or os.getenv("GEMINI_MODEL")
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

    model = model or os.getenv("OLLAMA_MODEL", "gemma4")
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
        },
    )
    return resp.json()["message"].get("content", "").strip()


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
