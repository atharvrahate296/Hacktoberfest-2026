"""Cloud inference client for Scribble2Structure -- Gemini API.

Handles:
  1. Build one Gemini API client from the API key (module-level singleton).
  2. Build a multimodal user turn from the uploaded image + mode prompt.
  3. Call generate_content() on the configured model and read the reply.
  4. Walk a model chain, moving to the next model when one is saturated.
  5. Parse the JSON object out of the reply (with at most 1 corrective retry).
  6. Validate the Mermaid diagram (with at most 1 corrective retry).
  7. Normalize everything into an Extraction.

The configured model defaults to ``gemini-2.5-flash-lite``. It may also be a
Gemma id (Gemma 3 4B/12B/27B, Gemma 4), which share the endpoint and accept
images, so the request is built to the *safest common denominator* of both
families rather than to the capabilities of whichever one is configured:

  * No JSON mode. ``response_mime_type``/``response_schema`` are rejected by
    every Gemma id ("JSON mode is not enabled for models/<id>"). The JSON
    schema therefore lives in the prompt (app/prompts.py) and the reply is
    parsed afterwards. Gemini could honour a response schema, but relying on it
    would make the app silently model-dependent, so the prompt-driven contract
    is kept for both.
  * No system role. Gemma was not trained with one and rejects
    ``system_instruction``, so the system prompt is folded into the single user
    turn. This is accepted by Gemini too.
  * No sampler knobs unless configured. Gemma is served with the default
    sampler, so ``temperature`` and ``top_p`` are omitted unless the operator
    sets them, rather than being sent with a value Gemini would honour and
    Gemma would reject.

That conservatism costs a little of what Gemini could offer, and buys one
codebase that keeps working if ``GEMMA_MODEL`` is pointed at a Gemma id.

Images are sent inline as base64 in the request; nothing is written to disk.
"""

from __future__ import annotations

import logging
import os
import time
from threading import Lock
from typing import Any

from dotenv import load_dotenv

from . import diagram as diagram_mod
from . import prompts, validation

load_dotenv(override=False)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_MODEL = "gemini-2.5-flash-lite"

# Vision-capable ids served by the Gemini API, Gemini first. The env var is
# named GEMMA_MODEL for continuity, but any of these may be set there.
#
# Gemma 3 4B/12B/27B and Gemma 4 accept image input. The Gemma 3 1B size is
# text-only and is deliberately absent.
KNOWN_MODELS = (
    "gemini-2.5-flash-lite",
    "gemini-2.5-flash",
    "gemma-3-27b-it",
    "gemma-3-12b-it",
    "gemma-3-4b-it",
    "gemma-4-26b-a4b-it",
)

# Gemini API public endpoint and version.
DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com"
DEFAULT_API_VERSION = "v1beta"

# Enough room for a full Mermaid diagram plus the notes payload. Comfortably
# inside both ceilings: Gemma 3 caps generation at 8192 tokens and
# gemini-2.5-flash-lite at 65536.
MAX_OUTPUT_TOKENS = 4096

# How much of the model's own reply to keep for diagnosis when a run yields
# nothing. Enough to see what went wrong, short enough to return in JSON.
RAW_PREVIEW_CHARS = 1200

# Wall-clock ceiling for a single HTTP call to the Gemini API. The SDK's own
# default leaves a stalled connection waiting indefinitely, which presents to
# the user as a request that never returns. Always bounded.
def _call_timeout_ms() -> int:
    """Per-request HTTP timeout in milliseconds. Never zero."""
    return max(5_000, _int_env("GEMINI_TIMEOUT_MS", 120_000))

# Backoff for a saturated or erroring endpoint, before moving to the next model
# in the chain. Replaced by a no-op in tests so the suite never really sleeps.
_SLEEP = time.sleep
RETRY_DELAYS = (1.0, 3.0)

# Errors that mean the request itself is wrong. Walking the model chain cannot
# fix these, and retrying only burns quota, so they fail immediately.
_FATAL_MESSAGE_HINTS = (
    "json mode is not enabled",
    "developer instruction is not enabled",
    "function calling is not enabled",
    "api key not valid",
    "permission denied",
)


def _env(name: str) -> str:
    return (os.getenv(name) or "").strip()


def _int_env(name: str, fallback: int) -> int:
    try:
        return int(_env(name))
    except (ValueError, TypeError):
        return fallback


def _float_env(name: str) -> float | None:
    raw = _env(name)
    if not raw:
        return None
    try:
        return float(raw)
    except (ValueError, TypeError):
        return None


def api_key() -> str:
    """The Gemini API key.

    GEMINI_API_KEY is the documented name; the two aliases are accepted because
    that is what the Google tooling and most .env templates write.
    """
    return _env("GEMINI_API_KEY") or _env("GOOGLE_API_KEY") or _env("GEMINI_KEY")


def model_id() -> str:
    """The configured model, falling back to the shipped default."""
    return _env("GEMMA_MODEL") or _env("GEMINI_MODEL") or DEFAULT_MODEL


def model_chain() -> list[str]:
    """The configured model followed by any opt-in fallbacks, deduplicated.

    GEMMA_MODEL_FALLBACKS is a comma-separated list of ids tried in order when
    the primary is saturated, rate-limited, or missing. It is empty by default
    so a failure is reported rather than silently answered by a smaller model.
    """
    chain: list[str] = []
    for candidate in [model_id(), *_env("GEMMA_MODEL_FALLBACKS").split(",")]:
        name = candidate.strip()
        if name and name not in chain:
            chain.append(name)
    return chain


def endpoint_url() -> str:
    """The API base URL this client talks to, for /api/health and /api/config."""
    base = _env("GEMINI_BASE_URL") or DEFAULT_BASE_URL
    version = _env("GEMINI_API_VERSION") or DEFAULT_API_VERSION
    return f"{base.rstrip('/')}/{version.strip('/')}"


def key_present() -> bool:
    """True when inference can run: a Gemini API key is set."""
    return bool(api_key())


def max_output_tokens() -> int:
    return _int_env("GEMMA_MAX_OUTPUT_TOKENS", MAX_OUTPUT_TOKENS)


# ---------------------------------------------------------------------------
# Public error types
# ---------------------------------------------------------------------------


class InferenceError(RuntimeError):
    """A user-facing inference failure."""


class ModelUnavailable(InferenceError):
    """A model-level failure that may not affect the next model in the chain.

    ``retryable`` distinguishes the two cases that look alike from here:
    a saturated endpoint is worth another try after a backoff, while a model
    the API does not serve will fail identically forever.

    ``max_attempts`` caps how many times this specific failure is retried
    against the same model. A timeout that has already consumed its full HTTP
    budget should be tried once more, not three times: three 120s timeouts are
    six minutes of apparent hang before the chain moves on.
    """

    def __init__(
        self,
        message: str,
        *,
        retryable: bool = True,
        max_attempts: int | None = None,
    ):
        super().__init__(message)
        self.retryable = retryable
        self.max_attempts = max_attempts


class ConfigurationError(InferenceError):
    """A setup problem the operator can fix without restarting the process.

    Kept distinct from other failures because it must never be memoized: a key
    pasted into .env under a running --reload server should start working
    immediately, not be poisoned by the first request that lacked it.
    """


# ---------------------------------------------------------------------------
# Client singleton
# ---------------------------------------------------------------------------

_client: Any = None
_client_lock = Lock()
_client_error: str | None = None


def _build_client():
    """Construct a Gemini API client bound to the configured key and endpoint."""
    key = api_key()
    if not key:
        raise ConfigurationError(
            "No Gemini API key is set. Add GEMINI_API_KEY to your .env file "
            "(create one at https://aistudio.google.com/apikey) and restart the app."
        )

    try:
        from google import genai
        from google.genai import types
    except ImportError as exc:
        raise InferenceError(
            "The google-genai package is not installed. Run: pip install -r requirements.txt"
        ) from exc

    try:
        options: dict[str, Any] = {}
        base = _env("GEMINI_BASE_URL")
        version = _env("GEMINI_API_VERSION")
        if base:
            options["base_url"] = base
        if version:
            options["api_version"] = version
        timeout_ms = _call_timeout_ms()
        options["timeout"] = timeout_ms
        return genai.Client(
            api_key=key,
            http_options=types.HttpOptions(**options) if options else None,
        )
    except InferenceError:
        raise
    except Exception as exc:
        raise InferenceError(
            f"Could not initialize the Gemini API client: {str(exc)[:300]}"
        ) from exc


def _get_client():
    """Return the shared client, creating it on first call (thread-safe).

    A broken SDK install or an unusable endpoint is remembered for the process
    lifetime: neither fixes itself between requests, and re-attempting on every
    upload would add latency to every failure. A missing key is NOT remembered
    (see ConfigurationError).
    """
    global _client, _client_error

    if _client_error is not None:
        raise InferenceError(f"Gemini API client failed to initialize: {_client_error}")
    if _client is not None:
        return _client

    with _client_lock:
        if _client is not None:
            return _client
        if _client_error is not None:
            raise InferenceError(
                f"Gemini API client failed to initialize: {_client_error}"
            )
        try:
            _client = _build_client()
        except ConfigurationError:
            # Recoverable in place: never poison the singleton over a missing key.
            raise
        except InferenceError as exc:
            _client_error = str(exc)[:400]
            raise
        logger.info("Gemini API client ready. Endpoint: %s", endpoint_url())
        return _client


# ---------------------------------------------------------------------------
# Error translation
# ---------------------------------------------------------------------------


def _status_of(exc: BaseException) -> int | None:
    """The HTTP status carried by an SDK error, if it has one."""
    for attr in ("code", "status_code"):
        value = getattr(exc, attr, None)
        if isinstance(value, int):
            return value
    return None


def _message_of(exc: BaseException) -> str:
    message = getattr(exc, "message", None)
    if isinstance(message, str) and message.strip():
        return message
    return str(exc)


def _looks_like_timeout(exc: BaseException) -> bool:
    """True for a stalled/expired request rather than a server-side error."""
    if isinstance(exc, TimeoutError):
        return True
    name = type(exc).__name__.lower()
    if "timeout" in name or "deadline" in name:
        return True
    lowered = _message_of(exc).lower()
    return "timed out" in lowered or "timeout" in lowered


def _classify(exc: BaseException) -> InferenceError:
    """Translate an SDK error into a user-facing error.

    Model-level failures (saturation, a missing model, a 5xx) become
    ModelUnavailable so the chain moves on; request-level failures stay fatal
    because no other model would accept the request either.
    """
    status = _status_of(exc)
    message = _message_of(exc)
    lowered = message.lower()

    if any(hint in lowered for hint in _FATAL_MESSAGE_HINTS):
        return InferenceError(
            "The Gemini API rejected this request: " f"{message[:300]}"
        )
    if status in (401, 403):
        return InferenceError(
            "The Gemini API key was rejected (HTTP {0}). Check GEMINI_API_KEY in your "
            ".env file -- see https://aistudio.google.com/apikey".format(status)
        )
    if status == 404:
        # The endpoint does not serve this id. Retrying it cannot help.
        return ModelUnavailable(
            f"Model not available on the Gemini API: {message[:200]}", retryable=False
        )
    if _looks_like_timeout(exc):
        # Already consumed a full HTTP budget. One more try, then move on.
        return ModelUnavailable(
            f"The Gemini API did not respond within {_call_timeout_ms() // 1000}s.",
            retryable=True,
            max_attempts=2,
        )
    # Saturation and 5xx: the same model may well succeed shortly.
    return ModelUnavailable(message[:300], retryable=True)


def _call_once(model: str, contents: list[Any], config: Any) -> str:
    """One generate_content() call, returning the reply text."""
    client = _get_client()
    try:
        response = client.models.generate_content(
            model=model, contents=contents, config=config
        )
    except InferenceError:
        raise
    except Exception as exc:
        raise _classify(exc) from exc
    return _response_text(response, model)


def _response_text(response: Any, model: str) -> str:
    """Pull the reply text out of a response, explaining an empty one.

    An empty reply is almost always a safety filter or a finish reason rather
    than a bug, so say which instead of failing with a decoding error.
    """
    text = ""
    try:
        text = response.text or ""
    except (AttributeError, ValueError):
        text = ""

    if text.strip():
        return text.strip()

    reason = "the model returned an empty response"
    try:
        candidates = getattr(response, "candidates", None) or []
        if candidates:
            finish = getattr(candidates[0], "finish_reason", None)
            if finish:
                reason = f"the model stopped early ({finish})"
        feedback = getattr(response, "prompt_feedback", None)
        block = getattr(feedback, "block_reason", None)
        if block:
            reason = f"the request was blocked by a safety filter ({block})"
    except Exception:
        pass

    raise InferenceError(f"{reason} while using '{model}'.")


# ---------------------------------------------------------------------------
# Request assembly
# ---------------------------------------------------------------------------


def _types():
    """The SDK's schema module, or a user-facing error when it is missing."""
    try:
        from google.genai import types
    except ImportError as exc:
        raise InferenceError(
            "The google-genai package is not installed. Run: pip install -r requirements.txt"
        ) from exc
    return types


def _build_config():
    """Generation config restricted to what every supported model accepts.

    Sampler knobs are omitted unless explicitly configured, and JSON mode and
    system instructions are never requested. See the module docstring: the
    default model is Gemini, but a Gemma id must keep working from the same
    codebase, so the config stays at their intersection.
    """
    types = _types()

    kwargs: dict[str, Any] = {"max_output_tokens": max_output_tokens()}

    temperature = _float_env("GEMMA_TEMPERATURE")
    if temperature is not None:
        kwargs["temperature"] = temperature
    top_p = _float_env("GEMMA_TOP_P")
    if top_p is not None:
        kwargs["top_p"] = top_p

    # response_mime_type / response_schema are intentionally NOT set: every
    # Gemma id rejects JSON mode on this endpoint. See the module docstring.
    return types.GenerateContentConfig(**kwargs)


def _fuse(system_prompt: str, user_prompt: str) -> str:
    """Join the system prompt and the request into one user turn.

    Gemma rejects a system role outright, so the instruction block and the
    per-request text are concatenated with a rule between them. Gemini accepts
    the same shape, which keeps one request format for both families.
    """
    return f"{system_prompt.strip()}\n\n---\n\n{user_prompt.strip()}"


def _build_image_contents(image: validation.ValidatedImage, system_prompt: str, user_prompt: str):
    """A single user turn holding the image bytes and the fused prompt."""
    types = _types()

    part_image = types.Part.from_bytes(data=image.data, mime_type=image.mime)
    part_text = types.Part.from_text(text=_fuse(system_prompt, user_prompt))
    return [types.Content(role="user", parts=[part_image, part_text])]


def _build_text_contents(prompt: str):
    """A single user turn holding only text, for corrective retries."""
    types = _types()

    return [types.Content(role="user", parts=[types.Part.from_text(text=prompt)])]


# ---------------------------------------------------------------------------
# Public generation functions
# ---------------------------------------------------------------------------


def _call_chain(contents: list[Any], config: Any) -> tuple[str, str, list[str]]:
    """Run one generation, walking the model chain on model-level failures.

    Returns (text, model_used, fallbacks_skipped).
    """
    chain = model_chain()
    last: InferenceError | None = None

    for index, model in enumerate(chain):
        for attempt in range(len(RETRY_DELAYS) + 1):
            try:
                return _call_once(model, contents, config), model, chain[:index]
            except ModelUnavailable as exc:
                last = exc
                # A saturated endpoint is worth a short backoff. A model the
                # API does not serve is not: hand off at once. A failure may
                # also cap its own attempts (a timeout has already spent its
                # budget, so retrying it three times just multiplies the wait).
                ceiling = (
                    exc.max_attempts
                    if exc.max_attempts is not None
                    else len(RETRY_DELAYS) + 1
                )
                if not exc.retryable or attempt + 1 >= ceiling:
                    break
                _SLEEP(RETRY_DELAYS[min(attempt, len(RETRY_DELAYS) - 1)])
            except InferenceError:
                raise

    raise last or InferenceError("No model in the chain is available.")


def _generate(
    image: validation.ValidatedImage, system_prompt: str, user_prompt: str
) -> tuple[str, str, list[str]]:
    """Run one multimodal generation round. Returns (text, model, fallbacks)."""
    return _call_chain(_build_image_contents(image, system_prompt, user_prompt), _build_config())


def _generate_text_only(prompt: str) -> tuple[str, str, list[str]]:
    """Text-only generation for corrective retries (no image)."""
    return _call_chain(_build_text_contents(prompt), _build_config())


# ---------------------------------------------------------------------------
# Diagram repair
# ---------------------------------------------------------------------------


def _repair_diagram(
    raw_source: str, error: str, expected_type: str = "flowchart"
) -> tuple[str | None, str | None]:
    try:
        repair_content = diagram_mod.build_repair_prompt(raw_source, error, expected_type=expected_type)
        fixed, _model, _fallbacks = _generate_text_only(repair_content)
    except InferenceError as exc:
        return None, f"Diagram repair could not run: {exc}"

    try:
        validated = diagram_mod.validate_mermaid(fixed)
        return validated["mermaid"], None
    except diagram_mod.DiagramError as exc:
        return None, f"Diagram repair failed: {exc}"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _extraction_is_empty(extraction: validation.Extraction) -> bool:
    """True when nothing usable came back, so a retry could still help."""
    return not (
        extraction.diagram
        or extraction.notes
        or extraction.takeaways
        or extraction.action_items
        or extraction.sql_schema
    )


def _extract_diagram(
    payload: dict, telemetry: dict
) -> dict | None:
    """Validate the payload's diagram, with one corrective retry on failure."""
    raw_diagram = payload.get("diagram")
    raw_source, explanation, raw_type = "", "", "flowchart"
    if isinstance(raw_diagram, dict):
        raw_source = raw_diagram.get("mermaid") or ""
        explanation = raw_diagram.get("explanation") or ""
        raw_type = raw_diagram.get("type") or "flowchart"
    elif isinstance(raw_diagram, str):
        raw_source = raw_diagram

    if not raw_source:
        return None

    telemetry["diagram_attempts"] = 1
    try:
        return diagram_mod.validate_mermaid(raw_source, explanation)
    except diagram_mod.DiagramError as exc:
        fixed, repair_error = _repair_diagram(raw_source, str(exc), expected_type=raw_type)
        telemetry["diagram_attempts"] = 2
        if fixed:
            return {"type": raw_type, "mermaid": fixed, "explanation": explanation}
        telemetry["diagram_repair_error"] = repair_error
        telemetry["diagram_error"] = str(exc)
        return None


def _record_model(telemetry: dict, model_used: str, fallbacks: list[str]) -> None:
    """Fold a completed call's model choice into the response telemetry.

    Replaces rather than accumulates: the extraction on screen came from the
    last successful call, so its model is the one reported. Stacking the skip
    lists of every attempt would describe a chain walk that never happened --
    a retry that lands back on the primary must not leave the primary listed
    as skipped.
    """
    telemetry["model_used"] = model_used
    telemetry["model_fallbacks"] = list(fallbacks)


def analyze(
    image: validation.ValidatedImage, mode: str = "auto"
) -> tuple[validation.Extraction, dict]:
    """Run a full extraction using cloud inference via the Gemini API.

    Returns (extraction, telemetry).
    """
    chain = model_chain()
    telemetry: dict = {
        "model": chain[0],
        "model_chain": chain,
        "model_used": chain[0],
        "model_fallbacks": [],
        "mode": mode,
        "endpoint": endpoint_url(),
        "attempts": 0,
        "diagram_attempts": 0,
        "elapsed_ms": 0,
    }
    started = time.perf_counter()

    sys_prompt = prompts.build_system_prompt(mode)
    usr_prompt = prompts.build_user_prompt(mode)

    try:
        text, model_used, fallbacks = _generate(image, sys_prompt, usr_prompt)
    except InferenceError:
        telemetry["elapsed_ms"] = int((time.perf_counter() - started) * 1000)
        raise
    telemetry["attempts"] = 1
    _record_model(telemetry, model_used, fallbacks)

    payload = validation.extract_json_object(text)
    if payload is None:
        telemetry["attempts"] = 2
        repair = (
            "Your previous reply could not be parsed as JSON. Reply with ONLY the valid JSON "
            "object matching the schema — no markdown fences, no notes, no commentary. "
            f"Here is what you previously returned:\n\n{text[:2000]}"
        )
        try:
            text, model_used, fallbacks = _generate_text_only(repair)
            _record_model(telemetry, model_used, fallbacks)
            payload = validation.extract_json_object(text)
        except InferenceError:
            payload = None

    if payload is None:
        telemetry["elapsed_ms"] = int((time.perf_counter() - started) * 1000)
        telemetry["raw_preview"] = text[:RAW_PREVIEW_CHARS]
        extraction = validation.normalize_extraction(None, requested_mode=mode)
        extraction.warnings.append(
            "The model replied in a format this app could not parse. Please try again."
        )
        return extraction, telemetry

    validated_diagram = _extract_diagram(payload, telemetry)
    extraction = validation.normalize_extraction(
        payload, requested_mode=mode, diagram_validated=validated_diagram
    )

    # A reply that parses but carries nothing usable means the model did not
    # read the image. Retry once WITH the image, under an explicit instruction to
    # transcribe rather than describe.
    if _extraction_is_empty(extraction):
        telemetry["content_attempts"] = 1
        strict = (
            "Your previous answer contained no usable content from this image. "
            "Look at the image again and transcribe only what is actually written or drawn in it. "
            "List each box, arrow and label you can see. If a section is illegible, say so in "
            "\"uncertain_text\" instead of leaving everything empty. Do not describe the task and "
            "do not repeat any part of the JSON template."
        )
        try:
            retry_text, model_used, fallbacks = _generate(
                image, sys_prompt, f"{usr_prompt}\n\n{strict}"
            )
            telemetry["attempts"] = 2
            _record_model(telemetry, model_used, fallbacks)
            retry_payload = validation.extract_json_object(retry_text)
            if retry_payload is not None:
                retry_diagram = _extract_diagram(retry_payload, telemetry)
                retry_extraction = validation.normalize_extraction(
                    retry_payload, requested_mode=mode, diagram_validated=retry_diagram
                )
                if not _extraction_is_empty(retry_extraction):
                    extraction = retry_extraction
                    telemetry["recovered"] = True
        except InferenceError as exc:
            telemetry["content_retry_error"] = str(exc)[:200]

    if _extraction_is_empty(extraction):
        # Keep the model's own words so a failure is diagnosable instead of blank.
        telemetry["raw_preview"] = text[:RAW_PREVIEW_CHARS]
        telemetry["degraded"] = True

    telemetry["elapsed_ms"] = int((time.perf_counter() - started) * 1000)
    return extraction, telemetry