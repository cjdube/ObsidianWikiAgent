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
import ipaddress
from urllib.parse import urlparse
import random
import time
from typing import Callable, Optional

import requests

from agent import budget, usage_ledger

MAX_TOOL_ITERATIONS = 6

DEFAULT_PROVIDER = "ollama"
GEMINI_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta"

# The Gemini free tier rate-limits aggressively, and a vault rebuild is
# hundreds of calls; without backoff a single 429 would fail the whole source
# and burn the retry budget in wiki_ingest.
_RETRY_STATUSES = (429, 500, 502, 503, 504)
_MAX_HTTP_ATTEMPTS = 5

# One connection pool for the process. A run is dozens of calls to the same host
# — 30 iterations per source, per source — and every requests.post() opened a
# fresh connection for each. Against localhost that is noise; against Gemini it
# is a TLS handshake per call. Single-threaded, so a shared Session is safe.
_session = requests.Session()


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


def _ollama_host(explicit: Optional[str] = None) -> str:
    """Resolve Ollama's endpoint without silently sending vault data off-box."""
    host = (explicit or os.getenv("OLLAMA_HOST") or "http://localhost:11434").strip()
    parsed = urlparse(host)
    hostname = parsed.hostname
    if not hostname:
        raise ValueError(f"invalid OLLAMA_HOST '{host}'")
    try:
        loopback = ipaddress.ip_address(hostname).is_loopback
    except ValueError:
        loopback = hostname.lower() == "localhost"
    if not loopback and os.getenv("ALLOW_REMOTE_OLLAMA", "").lower() not in {"1", "true", "yes"}:
        raise RuntimeError(
            "OLLAMA_HOST points to a remote server; set ALLOW_REMOTE_OLLAMA=1 "
            "only if sending vault contents off this machine is intentional"
        )
    return host.rstrip("/")


# run_agent returns a string either way, so this is how a caller tells a real
# answer from a loop that ran out of turns. Named rather than spelled out at
# each end, so the two cannot drift.
INCOMPLETE_PREFIX = "[incomplete:"


def _incomplete(max_iterations: int, logger: Optional[logging.Logger]) -> str:
    """The both-providers answer when the loop runs out of iterations."""
    if logger:
        logger.warning(
            f"agent loop exceeded max_iterations={max_iterations} without a "
            "final answer — returning best-effort partial result"
        )
    return (
        f"{INCOMPLETE_PREFIX} hit max_iterations={max_iterations} tool calls "
        "without reaching a final answer]"
    )


# How much of one tool argument or result value reaches the structured log.
#
# That log is the record outside the model's reach (see SECURITY.md), so it has
# to keep the shape of every call: which tool, with which page name, and whether
# it worked. It does not have to keep a second verbatim copy of a file that is
# already on disk — and keeping one cost the log its own history. read_raw_file
# returns whole sources (one observed source was 366 KB) and write_wiki_page
# takes whole pages, so a single dense run writes several MB into a 5 MB x 3
# rotation and can age out the entire retention window in a day, taking with it
# exactly the forensic record the rotation exists to preserve.
_LOG_VALUE_MAX_CHARS = 2000


def _clip(value: str) -> str:
    if len(value) <= _LOG_VALUE_MAX_CHARS:
        return value
    dropped = len(value) - _LOG_VALUE_MAX_CHARS
    return f"{value[:_LOG_VALUE_MAX_CHARS]}… [+{dropped} chars]"


def _clip_values(payload: dict) -> dict:
    """`payload` with every long string value shortened, keys untouched.

    Per value rather than over the whole rendered dict: clipping the rendering
    would let one 20 KB page body push the page's *name* out of the line, and
    the name is the part worth keeping."""
    return {k: _clip(v) if isinstance(v, str) else v for k, v in payload.items()}


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
        except budget.BudgetExceeded:
            # The run is over, not this tool call. Reported back to the model as
            # a tool error it would be swallowed twice over: the loop would keep
            # going, and the SIGALRM watchdog that raised it is one-shot
            # (agent/budget.py), so it can never fire again — leaving only the
            # cooperative checks it exists to backstop. That matters here
            # specifically because the failure the watchdog was written for is a
            # blocking read() (2026-08-13, 433 minutes behind a macOS consent
            # prompt), and every file read in a run happens inside this call.
            raise
        except Exception as e:
            result = {"error": f"tool '{fn_name}' raised: {e}"}
    if logger:
        # A refused call is logged at WARNING so it carries the level its
        # content already claims. These are not fatal — the model is handed the
        # error and usually recovers — but they are the trail a stuck stage
        # leaves, and at INFO they were indistinguishable from the successful
        # calls around them in a log that runs to thousands of lines.
        line = (
            f"tool_call {fn_name}({_clip_values(fn_args)}) -> "
            f"{json.dumps(_clip_values(result))}"
        )
        logger.warning(line) if "error" in result else logger.info(line)
    return result


def _truncated_call(
    fn_name: str, logger: Optional[logging.Logger], cap: str
) -> dict:
    """The result to feed back for a tool call that arrived cut off.

    A reply that stops at its output limit stops wherever it happened to be,
    and if that is inside a tool call the provider still returns whatever
    arguments it managed to parse. The call then arrives looking complete while
    missing every character the model had not written yet.

    `cap` names the limit that was hit, and reaches the log line only. Ollama
    can say which num_predict it was; Gemini reports MAX_TOKENS without saying
    what the number was, so it can only name the limit. What goes back to the
    model is the same either way, because the remedy is.

    Running it is the danger. write_wiki_page replaces whole files, so a
    truncated 'content' is a half page written over a real one — and nothing
    downstream would notice, because a short page is not a malformed page.
    The 2026-08-20 ingest hit this four times and got away with it only by
    accident: the model emits 'content' before 'name', so the cut took 'name'
    with it and the call died on a missing-argument TypeError instead. That
    TypeError was also actively misleading — it sent the model back to re-emit
    the same oversized page, which truncated again, three times in a row.

    So refuse the call and say why. The model can act on this one: the fix is
    a shorter page, not a re-send.
    """
    if logger:
        logger.warning(
            f"Refused a truncated '{fn_name}' call — the reply hit {cap} "
            f"mid-call, so its arguments are incomplete. Not running it."
        )
    return {
        "error": (
            f"tool '{fn_name}' was not run: your reply was cut off at the "
            f"reply-length limit before you finished writing the call, so its "
            f"arguments are missing text. Do not send the same call again — it "
            f"will be cut off in the same place. Send a shorter one instead: "
            f"for a page, write a more concise page, or split it into two "
            f"pages linked to each other."
        )
    }


def _truncated_reply_nudge(logger: Optional[logging.Logger], cap: str) -> str:
    """The text to send back when a reply was cut off before any tool call.

    The other half of the truncation case. _truncated_call covers the cut that
    lands *inside* a call; this one lands before the model wrote anything
    parseable, so the provider reports the length stop with no tool calls and
    no content at all. Verified live on 2026-08-20 against gemma4:26b-mlx at
    num_predict 600 and 1200.

    Returns bare text rather than a whole message, because the two providers
    disagree about the envelope — Ollama wants {"role", "content"}, Gemini
    wants {"role", "parts": [{"text"}]} — while the words are the same for
    both. Each caller wraps it in its own shape.

    The loop used to return that empty content as the final answer, which reads
    downstream as a model that chose to do nothing: wiki_ingest logged
    'produced no wiki writes' and spent one of its three attempts on the source
    for a reply that never finished a sentence. So say what happened and let the
    model try again shorter — the turn is not over.
    """
    if logger:
        logger.warning(
            f"Reply was cut off at {cap} before any tool call — nudging the "
            f"model to answer shorter rather than treating the empty reply as "
            f"a final answer."
        )
    return (
        "Your last reply was cut off at the reply-length limit before you "
        "finished it, so none of it reached me. Answer again, shorter: "
        "make one tool call at a time, and keep any page you write "
        "concise or split it into two pages linked to each other."
    )


# Ceiling on one backoff, whoever picked it. The computed backoff has always
# been capped; a server-supplied Retry-After was not, and the run budget only
# catches an over-long one while a budget is running — wiki_query.py starts
# none, and wiki_lint starts one only for --deep. So an interactive question
# could sit silently for the full hour a `Retry-After: 3600` asks for.
#
# Capping it is safe because nothing here is obliged to obey the server's
# number: _MAX_HTTP_ATTEMPTS bounds the attempts and the caller retries the
# whole source anyway. Waiting an hour to be polite to a quota that resets on
# its own schedule buys nothing a shorter wait does not.
_MAX_BACKOFF_S = 60


def _retry_delay(retry_after: Optional[str], attempt: int) -> float:
    """How long to wait before the next attempt, capped from both ends.

    Retry-After is allowed to be an HTTP date rather than a count of seconds
    (RFC 9110), and float() on 'Wed, 21 Oct 2015 07:28:00 GMT' raises. That
    happened inside the retry loop, where nothing catches it, so a header whose
    entire purpose is to slow the client down would instead kill the source
    with a ValueError that named neither the header nor the server.

    The date form is not parsed, only refused: every value is clamped to
    _MAX_BACKOFF_S anyway, so any date far enough out to matter would be
    clamped to the same number the fallback already gives. Negative and absurd
    values fall out the same way.

    Jitter is added last so two clients told the same number do not return in
    lockstep.
    """
    delay = min(2 ** attempt, _MAX_BACKOFF_S)
    if retry_after:
        try:
            delay = max(0.0, min(float(retry_after), _MAX_BACKOFF_S))
        except ValueError:
            pass
    return delay + random.uniform(0, 1)


def _post_with_retry(
    url: str,
    payload: dict,
    headers: Optional[dict] = None,
    timeout: int = 120,
    logger: Optional[logging.Logger] = None,
) -> requests.Response:
    """POST, retrying transient failures with exponential backoff + jitter.

    Honors Retry-After when the server sends it (Gemini does on quota errors),
    up to _MAX_BACKOFF_S — see _retry_delay for why the server's number is
    capped rather than obeyed. Network-level failures (timeouts, connection errors — routine for a large
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
            resp = _session.post(
                url, json=payload, headers=headers,
                timeout=budget.clamp_timeout(timeout),
            )
        except requests.exceptions.RequestException as e:
            if attempt == _MAX_HTTP_ATTEMPTS:
                raise
            delay = _retry_delay(None, attempt)
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
        delay = _retry_delay(resp.headers.get("Retry-After"), attempt)
        reason = f"HTTP {resp.status_code} from model API"
        budget.before_retry(delay, reason)
        if logger:
            logger.warning(
                f"{reason} — retrying in "
                f"{delay:.1f}s (attempt {attempt}/{_MAX_HTTP_ATTEMPTS})"
            )
        time.sleep(delay)
    raise RuntimeError("unreachable")


def _task_name(logger: Optional[logging.Logger]) -> str:
    """What the ledger records a call as having been for.

    setup_logger names each entrypoint's logger after the job and vault
    (wiki_ingest.<vault>, wiki_lint.<vault>), so the logger already carries the
    attribution. A call made with no logger at all has none to carry, and
    "agent.loop" would be a column that says only which file made the request.

    getattr rather than .name: the loop accepts anything logger-shaped, and the
    suite passes recorders that implement the four level methods and nothing
    else. A ledger field is not worth failing a run over.
    """
    return getattr(logger, "name", None) or "unknown"


def _usage_from(data: dict, backend: str) -> dict:
    """Token counts out of one response body, per provider.

    Gemini here is the REST endpoint, not the Python SDK: the JSON is
    camelCase and the SDK's snake_case attribute names match nothing. Reading
    them would produce a ledger of nulls that looks like a quiet provider
    rather than a bug, which is why tests/test_usage_ledger.py asserts the
    snake_case shape yields nulls.
    """
    if backend == "gemini":
        usage = data.get("usageMetadata") or {}
        candidates = data.get("candidates") or []
        return {
            "prompt_tokens": usage.get("promptTokenCount"),
            "output_tokens": usage.get("candidatesTokenCount"),
            "thinking_tokens": usage.get("thoughtsTokenCount"),
            "finish_reason": candidates[0].get("finishReason") if candidates else None,
        }
    return {
        "prompt_tokens": data.get("prompt_eval_count"),
        # eval_count comes back on every non-streaming response for free and
        # this repo had never read it — output size was logged nowhere at all.
        "output_tokens": data.get("eval_count"),
        "thinking_tokens": None,
        "finish_reason": data.get("done_reason"),
    }


def _post_and_record(
    url: str,
    payload: dict,
    *,
    backend: str,
    model: str,
    caller: str,
    timeout: int,
    headers: Optional[dict] = None,
    logger: Optional[logging.Logger] = None,
    num_ctx: Optional[int] = None,
    tools_offered: int = 0,
) -> dict:
    """POST one model call, append its usage row, return the parsed body.

    Not a true seam — the four call sites still branch on provider before they
    get here — but it is the one place the response-to-row mapping lives, and
    that mapping is what rots when a fifth call site appears.

    One row per call, so a 30-iteration ingest writes 30 rows. Collapsing them
    into one per run would hide the growth the ledger exists to show: the
    prompt is re-sent in full every iteration.
    """
    t0 = time.monotonic()
    common_fields = {
        "task": _task_name(logger),
        "caller": caller,
        "backend": backend,
        "model": model,
        "num_ctx": num_ctx,
        "tools_offered": tools_offered,
    }
    try:
        resp = _post_with_retry(
            url, payload, headers=headers, timeout=timeout, logger=logger
        )
        data = resp.json()
    except Exception as e:
        usage_ledger.record(
            duration_ms=int((time.monotonic() - t0) * 1000),
            ok=False,
            error=f"{e.__class__.__name__}: {e}",
            **common_fields,
        )
        raise
    usage_ledger.record(
        duration_ms=int((time.monotonic() - t0) * 1000),
        **_usage_from(data, backend),
        **common_fields,
    )
    return data


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
    think: Optional[bool] = None,
) -> str:
    """Run the tool-calling loop and return the model's final text response.

    tools: list of OpenAI-style tool schemas (see agent/wiki_tools.py TOOL_SCHEMA).
           Translated per-provider as needed.
    dispatch: {function_name: callable} mapping — callable takes the parsed
              arguments dict via **kwargs and returns a JSON-serializable dict.
    max_iterations: raise for workflows that legitimately need many tool
              calls (e.g. wiki ingest, which can touch 10-15 pages per source).
    provider: 'ollama' (default) or 'gemini'; falls back to $LLM_PROVIDER.
    think: see _run_ollama. Ollama only — Gemini has no equivalent field, and a
           caller that sets it is describing the local model's behaviour, not
           stating a requirement the run should fail without. So it is ignored
           there rather than raising: the weekly `wiki_lint --deep` pass sets
           LLM_PROVIDER=gemini in its own .plist and shares this entry point.
    """
    name = _provider(provider)
    if name == "gemini":
        return _run_gemini(
            system_prompt, user_prompt, tools, dispatch, model, logger, max_iterations
        )
    return _run_ollama(
        system_prompt, user_prompt, tools, dispatch, model, host, logger,
        max_iterations, think,
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
    # "reached", not a bare count: the number is the size of the whole
    # transcript being re-sent this turn, not tokens spent on this turn alone.
    # Read as a running tally it invites the wrong fix — trimming one call's
    # output — when what grows is everything the loop has carried since turn 1.
    message = (
        f"prompt reached {count} tokens on iteration {iteration} "
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
    think: Optional[bool] = None,
) -> str:
    """`think` controls whether the model reasons before answering.

    None leaves the field off the request, which is the model's own default.
    False turns reasoning off, and for a step that transcribes rather than
    judges that is strictly better on both axes at once.

    Measured on gemma4:26b-mlx against a real wiki_ingest stage-2 conversation,
    rewriting the 95-line wiki/ollama.md, five trials:

        thinking on   252.6s  13,035 tok  cut off  kept 23/95 original lines
        thinking on   274.5s  12,903 tok  cut off  kept 52/95
        thinking off   88.1s   3,923 tok  clean    kept 94/95
        thinking off   75.1s   3,839 tok  clean    kept 94/95
        thinking off   75.1s   3,875 tok  clean    kept 94/95

    3.4x faster, and the accuracy runs the same way rather than against it. The
    reasoning block is generated into the same num_predict budget as the reply,
    so a long one leaves too little for the page body; the reply is then cut off
    mid-write, and because write_wiki_page replaces whole files, what lands is a
    short page over a real one. That is not hypothetical — the 2026-08-20 vault
    snapshot recorded wiki/local-llm-agent.md at 22 insertions and 78 deletions,
    an "update" that deleted most of the page. With thinking off nothing was cut
    off in any trial, and the token count barely moved between runs (3,839 /
    3,875 / 3,923) where thinking on ranged from 1,907 to 4,565 reasoning tokens
    on the identical task.

    None of which makes reasoning useless — it makes it a per-step choice. The
    same benchmark against stage 1, which decides *which* pages a source should
    touch, failed to call submit_plan at all in 3 of 6 trials with thinking off.
    So the caller decides, one stage at a time; see wiki_ingest.py.
    """
    model = _ollama_model(model)
    host = _ollama_host(host)
    timeout = int(os.getenv("OLLAMA_TIMEOUT", "300"))
    options = _ollama_options()
    cap = f"num_predict={options['num_predict']}"
    _warn_if_context_is_tight(system_prompt, user_prompt, options["num_ctx"], logger)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    payload = {
        "model": model,
        "tools": tools,
        "stream": False,
        "options": options,
    }
    # Omitted entirely when the caller said nothing, so a model with no notion
    # of thinking sees exactly the request it saw before this existed.
    if think is not None:
        payload["think"] = think

    for iteration in range(max_iterations):
        data = _post_and_record(
            f"{host}/api/chat",
            {**payload, "messages": messages},
            backend="ollama",
            model=model,
            caller="run",
            timeout=timeout,
            logger=logger,
            num_ctx=options["num_ctx"],
            tools_offered=len(tools),
        )
        _log_prompt_size(data, iteration + 1, options["num_ctx"], logger)
        _warn_if_reply_hit_the_cap(data, logger)
        message = data["message"]
        messages.append(message)

        tool_calls = message.get("tool_calls") or []
        if not tool_calls:
            if data.get("done_reason") == "length":
                messages.append({
                    "role": "user",
                    "content": _truncated_reply_nudge(logger, cap),
                })
                continue
            return message.get("content", "")

        # Ollama reports the cut on the whole reply, not per call, so every
        # call in a truncated turn is suspect: only the last one can actually
        # be short, but nothing in the response says which that is.
        truncated = data.get("done_reason") == "length"

        for call in tool_calls:
            fn_name = call["function"]["name"]
            fn_args = call["function"].get("arguments", {})
            if truncated:
                result = _truncated_call(fn_name, logger, cap)
            else:
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


def _gemini_timeout() -> int:
    """Seconds to wait on one Gemini call. The Ollama path has honoured
    OLLAMA_TIMEOUT all along while this one took _post_with_retry's hardcoded
    120 — and the slowest call in the repo is the --deep judgment pass, which
    reads the whole index."""
    return int(os.getenv("GEMINI_TIMEOUT", "120"))


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
        data = _post_and_record(
            url,
            {**payload_base, "contents": contents},
            backend="gemini",
            model=model,
            caller="run",
            headers=headers,
            timeout=_gemini_timeout(),
            logger=logger,
            tools_offered=len(tools),
        )

        candidates = data.get("candidates") or []
        if not candidates:
            # Prompt blocked by a safety filter, or an empty generation.
            reason = data.get("promptFeedback", {}).get("blockReason", "no candidates")
            raise RuntimeError(f"Gemini returned no candidates: {reason}")

        content = candidates[0].get("content") or {}
        parts = content.get("parts") or []
        contents.append({"role": "model", "parts": parts})

        # The same cut the Ollama path has guarded since 2026-08-20, under the
        # other provider's name for it. Gemini stops at its own output-token
        # limit and says so in finishReason, but it still returns whatever of
        # the call it had written — so a truncated write_wiki_page arrives
        # looking complete and lands half a page over a real one.
        #
        # Today the only Gemini caller in this repo is wiki_lint --deep, whose
        # tools are all read-only, so nothing can be damaged through this path
        # yet. LLM_PROVIDER=gemini is a documented per-run opt-in for a full
        # vault rebuild, and that one does write. Guarding it now costs a
        # branch; finding out later costs pages.
        truncated = candidates[0].get("finishReason") == "MAX_TOKENS"
        cap = "the model's output token limit"

        calls = [p["functionCall"] for p in parts if "functionCall" in p]
        if not calls:
            if truncated:
                contents.append({
                    "role": "user",
                    "parts": [{"text": _truncated_reply_nudge(logger, cap)}],
                })
                continue
            return "".join(p.get("text", "") for p in parts)

        responses = []
        for call in calls:
            fn_name = call["name"]
            fn_args = call.get("args") or {}
            if truncated:
                result = _truncated_call(fn_name, logger, cap)
            else:
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
    logger: Optional[logging.Logger] = None,
) -> str:
    """Single-turn, tool-free completion — for tasks where the caller
    assembles the surrounding structure itself rather than trusting the
    model to produce it.

    logger is optional but worth passing: it names the task in the usage
    ledger, and it is what lets _post_with_retry say that it is backing off.
    Without it a one-shot call is recorded as task "unknown"."""
    name = _provider(provider)

    if name == "gemini":
        # Same validation _run_gemini does. Skipping it here built a URL of
        # 'models/None:generateContent' and returned an opaque 404.
        model = _gemini_model(model)
        data = _post_and_record(
            f"{GEMINI_ENDPOINT}/models/{model}:generateContent",
            {
                "systemInstruction": {"parts": [{"text": system_prompt}]},
                "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
            },
            backend="gemini",
            model=model,
            caller="complete_text",
            headers={"x-goog-api-key": _gemini_key()},
            timeout=_gemini_timeout(),
            logger=logger,
        )
        candidates = data.get("candidates") or []
        if not candidates:
            return ""
        parts = (candidates[0].get("content") or {}).get("parts") or []
        return "".join(p.get("text", "") for p in parts).strip()

    model = _ollama_model(model)
    host = _ollama_host(host)
    options = _ollama_options()
    data = _post_and_record(
        f"{host}/api/chat",
        {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
            "options": options,
        },
        backend="ollama",
        model=model,
        caller="complete_text",
        timeout=int(os.getenv("OLLAMA_TIMEOUT", "300")),
        logger=logger,
        num_ctx=options["num_ctx"],
    )
    # Fall back to the module's logger when the caller passed none — a
    # truncated single-turn answer is exactly as silent as a truncated loop
    # turn, and this warning predates the parameter.
    _warn_if_reply_hit_the_cap(data, logger or logging.getLogger(__name__))
    return data["message"].get("content", "").strip()


def list_gemini_models() -> list[str]:
    """Model ids the configured key can actually reach, for generateContent.

    Model availability moves faster than this code; ask the API rather than
    hardcoding an id that may not exist.
    """
    resp = _session.get(
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
