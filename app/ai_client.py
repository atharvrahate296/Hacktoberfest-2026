"""OpenRouter access for multimodal inference.

Responsibilities, in order of the happy path:
  1. POST image + prompt
  2. Parse the JSON object out of the reply
  3. Validate the Mermaid, with at most one corrective retry
  4. Normalize everything into an Extraction

The client never raises for a recoverable problem. InferenceError carries a
message that is safe to show a user, and the notes survive a dead diagram.
"""

from __future__ import annotations

import base64
import json
import os
import time

import requests
from dotenv import load_dotenv

from . import diagram as diagram_mod
from . import prompts, validation

# Local development convenience: a .env file in the project root is honoured.
# Never overrides a variable already present in the real environment.
load_dotenv(override=False)

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
TIMEOUT_SECONDS = 120
MAX_ATTEMPTS = 2  # one initial call + at most one corrective retry

# Tried in order when OPENR_MODEL_FALLBACKS is set. A 429 here means the model
# is saturated upstream -- the correct response is to try a different one, not
# to hammer the same endpoint.
DEFAULT_FALLBACKS = "nvidia/nemotron-3.5-super-vl:free"

# Some reasoning models emit a `reasoning` field alongside `content` and spend
# the whole token budget there, leaving content empty. We always request extra
# headroom so the actual answer fits.
EXTRA_TOKEN_HEADROOM = 2500


class InferenceError(RuntimeError):
    """A user-facing inference failure."""


class ModelUnavailable(InferenceError):
    """This specific model is saturated or unavailable; another may work."""


def _env(name: str) -> str:
    return (os.getenv(name) or "").strip()


def api_key() -> str:
    return _env("OPENROUTER_API_KEY")


def model_id() -> str:
    return _env("OPENROUTER_MODEL") or "google/gemma-4-26b-a4b-it:free"


def model_chain() -> list[str]:
    """Primary model first, then any configured fallbacks, de-duplicated.

    Lets the app survive a saturated free tier instead of hard-failing on a 429.
    """
    chain = [model_id()]
    raw = _env("OPENROUTER_MODEL_FALLBACKS") or DEFAULT_FALLBACKS
    for candidate in (c.strip() for c in raw.split(",")):
        if candidate and candidate not in chain:
            chain.append(candidate)
    return chain


def _data_url(data: bytes, mime: str) -> str:
    return f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"


def _call(messages: list[dict], model: str, key: str) -> str:
    """One round trip to OpenRouter. Raises InferenceError on any failure."""
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "HTTP-Referer": _env("APP_URL") or "https://github.com/rohan-kokatare/scribble2structure",
        "X-Title": _env("APP_TITLE") or "Scribble2Structure",
    }
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.1,
        "max_tokens": 2000 + EXTRA_TOKEN_HEADROOM,
    }
    try:
        resp = requests.post(
            OPENROUTER_URL,
            headers=headers,
            data=json.dumps(payload),
            timeout=TIMEOUT_SECONDS,
        )
    except requests.Timeout as exc:
        raise InferenceError(
            f"The model did not respond within {TIMEOUT_SECONDS} seconds. Try again."
        ) from exc
    except requests.RequestException as exc:
        raise InferenceError(f"Could not reach the model provider: {exc}") from exc

    if resp.status_code in (401, 403):
        raise InferenceError("The model provider rejected the API key (401). Check OPENROUTER_API_KEY.")
    if resp.status_code == 402:
        raise InferenceError(
            "The model provider reports insufficient credit (402). Use a free model "
            "id or add credit at https://openrouter.ai/credits."
        )
    if resp.status_code == 429:
        # Raised distinctly so the caller can move to the next model rather than
        # retrying an endpoint that is already saturated.
        raise ModelUnavailable(
            resp.text[:160].replace("\n", " ") or "rate limited upstream (429)"
        )
    if resp.status_code == 404:
        raise InferenceError(
            f"The model id '{model}' was not found (404). Set OPENROUTER_MODEL to a "
            "model your account can access."
        )
    if resp.status_code >= 400:
        detail = resp.text[:200].replace("\n", " ")
        raise InferenceError(f"The model provider returned {resp.status_code}: {detail}")

    try:
        body = resp.json()
    except ValueError as exc:
        raise InferenceError("The model provider returned a non-JSON response.") from exc

    if body.get("error"):
        err = body["error"]
        msg = err.get("message") if isinstance(err, dict) else str(err)
        raise InferenceError(f"The model provider reported an error: {msg}")

    try:
        content = body["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise InferenceError("The model provider returned an unexpected response shape.") from exc

    if isinstance(content, list):
        content = "".join(
            p.get("text", "") for p in content if isinstance(p, dict) and p.get("text")
        )
    if not isinstance(content, str):
        content = ""
    text = content.strip()
    if not text:
        raise InferenceError("The model returned an empty response.")
    return text


def _messages_with_image(image: validation.ValidatedImage) -> list[dict]:
    return [
        {"role": "system", "content": prompts.SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": _data_url(image.data, image.mime)},
                },
                {"type": "text", "text": prompts.USER_PROMPT},
            ],
        },
    ]


def _repair_diagram(
    raw_source: str, error: str, model: str, key: str
) -> tuple[str | None, str | None]:
    """One corrective retry. Returns (clean_source, error_message)."""
    try:
        fixed = _call(
            [
                {
                    "role": "user",
                    "content": diagram_mod.build_repair_prompt(raw_source, error),
                }
            ],
            model,
            key,
        )
    except InferenceError as exc:
        return None, f"Diagram repair could not run: {exc}"

    try:
        validated = diagram_mod.validate_mermaid(fixed)
        return validated["mermaid"], None
    except diagram_mod.DiagramError as exc:
        return None, f"Diagram repair failed: {exc}"


def analyze(image: validation.ValidatedImage) -> tuple[validation.Extraction, dict]:
    """Run a full extraction. Returns (extraction, telemetry).

    Telemetry carries timings and attempt counts for the UI's honesty panel.
    """
    key = api_key()
    if not key:
        raise InferenceError(
            "OPENROUTER_API_KEY is not set. Copy .env.example to .env, add your key, "
            "and restart the app."
        )

    chain = model_chain()
    telemetry = {
        "model": chain[0],
        "model_chain": chain,
        "model_used": None,
        "model_fallbacks": [],
        "attempts": 0,
        "diagram_attempts": 0,
        "elapsed_ms": 0,
    }
    started = time.perf_counter()

    # Walk the model chain. A 429 means this model is saturated, so moving to
    # the next one is the useful response; any other error is fatal.
    text = None
    for candidate in chain:
        try:
            text = _call(_messages_with_image(image), candidate, key)
            telemetry["model_used"] = candidate
            break
        except ModelUnavailable as exc:
            telemetry["model_fallbacks"].append({"model": candidate, "reason": str(exc)[:120]})
            continue
    if text is None:
        raise InferenceError(
            "Every configured model is currently rate limited "
            f"({len(chain)} tried: {', '.join(chain)}). Wait a minute and retry."
        )
    model = telemetry["model_used"]
    telemetry["attempts"] = 1

    payload = validation.extract_json_object(text)
    if payload is None:
        # One repair attempt for malformed JSON, per the spec.
        telemetry["attempts"] = 2
        repair = (
            "Your previous reply could not be parsed as JSON. Reply with ONLY the JSON "
            "object, no fences and no commentary. Here is the text you returned:\n\n"
            f"{text[:2000]}"
        )
        try:
            text = _call(
                [
                    {"role": "system", "content": prompts.SYSTEM_PROMPT},
                    {"role": "user", "content": repair},
                ],
                model,
                key,
            )
            payload = validation.extract_json_object(text)
        except InferenceError:
            payload = None

    if payload is None:
        telemetry["elapsed_ms"] = int((time.perf_counter() - started) * 1000)
        extraction = validation.normalize_extraction(None)
        extraction.warnings.append(
            "The model replied in a format this app could not parse. Your image was "
            "not lost — try again, or use a different model id."
        )
        return extraction, telemetry

    # Validate the diagram, allowing one repair round.
    validated_diagram = None
    raw_diagram = payload.get("diagram")
    raw_source, explanation = "", ""
    if isinstance(raw_diagram, dict):
        raw_source = raw_diagram.get("mermaid") or ""
        explanation = raw_diagram.get("explanation") or ""
    elif isinstance(raw_diagram, str):
        raw_source = raw_diagram

    if raw_source:
        telemetry["diagram_attempts"] = 1
        try:
            validated_diagram = diagram_mod.validate_mermaid(raw_source, explanation)
        except diagram_mod.DiagramError as exc:
            # Repair uses the model that produced the diagram, so the retry is
            # consistent with the original answer.
            fixed, repair_error = _repair_diagram(raw_source, str(exc), model, key)
            telemetry["diagram_attempts"] = 2
            if fixed:
                validated_diagram = {"type": "flowchart", "mermaid": fixed, "explanation": explanation}
            else:
                telemetry["diagram_repair_error"] = repair_error
                telemetry["diagram_error"] = str(exc)

    extraction = validation.normalize_extraction(
        payload, diagram_validated=validated_diagram
    )
    telemetry["elapsed_ms"] = int((time.perf_counter() - started) * 1000)
    return extraction, telemetry


def key_present() -> bool:
    return bool(api_key())
