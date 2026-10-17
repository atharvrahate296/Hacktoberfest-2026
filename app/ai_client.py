"""Hugging Face multimodal Gemma inference client.

Handles:
  1. POST image + mode-specific prompt to Hugging Face Inference API / Router / dedicated endpoint
  2. Parse the JSON object out of the reply (with at most 1 corrective retry)
  3. Validate the Mermaid diagram (with at most 1 corrective retry)
  4. Normalize everything into an Extraction

Tokens remain server-side and are never exposed client-side.
Does not fall back to OpenRouter or fabricate responses.
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

HF_ROUTER_URL = "https://router.huggingface.co/v1/chat/completions"
TIMEOUT_SECONDS = 120
MAX_ATTEMPTS = 2  # one initial call + at most one corrective retry

# Default Gemma model on Hugging Face (supports image + text inputs)
DEFAULT_GEMMA_MODEL = "google/gemma-3-27b-it"

# Fallback Gemma models if the primary is saturated upstream
DEFAULT_FALLBACKS = "google/gemma-3-12b-it,google/gemma-3-4b-it"

EXTRA_TOKEN_HEADROOM = 2500


class InferenceError(RuntimeError):
    """A user-facing inference failure."""


class ModelUnavailable(InferenceError):
    """This specific model is saturated or unavailable upstream; another may work."""


def _env(name: str) -> str:
    return (os.getenv(name) or "").strip()


def hf_token() -> str:
    """Hugging Face access token."""
    return _env("HF_TOKEN")


# Backward compatibility alias
def api_key() -> str:
    return hf_token()


def model_id() -> str:
    """Configured Hugging Face model identifier."""
    return _env("HF_MODEL_ID") or DEFAULT_GEMMA_MODEL


def endpoint_url() -> str | None:
    """Optional dedicated Hugging Face endpoint URL."""
    url = _env("HF_ENDPOINT_URL")
    return url if url else None


def model_chain() -> list[str]:
    """Primary model first, then any configured fallbacks, de-duplicated."""
    chain = [model_id()]
    raw = _env("HF_MODEL_FALLBACKS") or DEFAULT_FALLBACKS
    for candidate in (c.strip() for c in raw.split(",")):
        if candidate and candidate not in chain:
            chain.append(candidate)
    return chain


def _data_url(data: bytes, mime: str) -> str:
    return f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"


def _resolve_url(custom_endpoint: str | None, model: str) -> str:
    if custom_endpoint:
        cleaned = custom_endpoint.strip().rstrip("/")
        if cleaned.endswith("/chat/completions"):
            return cleaned
        if cleaned.endswith("/v1"):
            return f"{cleaned}/chat/completions"
        return f"{cleaned}/v1/chat/completions"
    return HF_ROUTER_URL


def _call(messages: list[dict], model: str, token: str, endpoint: str | None = None) -> str:
    """One round trip to Hugging Face Inference API / Router / dedicated endpoint."""
    url = _resolve_url(endpoint, model)
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "HTTP-Referer": _env("APP_URL") or "https://github.com/rohan1252030019-netizen/scribble2structure",
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
            url,
            headers=headers,
            data=json.dumps(payload),
            timeout=TIMEOUT_SECONDS,
        )
    except requests.Timeout as exc:
        raise InferenceError(
            f"Hugging Face inference timed out after {TIMEOUT_SECONDS} seconds. Try again with a tighter crop or smaller image."
        ) from exc
    except requests.RequestException as exc:
        raise InferenceError(f"Could not reach Hugging Face inference service: {exc}") from exc

    if resp.status_code in (401, 403):
        raise InferenceError(
            "Hugging Face rejected the access token (401/403). Check HF_TOKEN in your environment or .env file."
        )
    if resp.status_code == 404:
        raise InferenceError(
            f"The model id '{model}' or endpoint was not found on Hugging Face (404). "
            "Verify HF_MODEL_ID or your access permissions for this Gemma model."
        )
    if resp.status_code == 429:
        raise ModelUnavailable(
            resp.text[:160].replace("\n", " ") or f"Hugging Face rate limited upstream (429) for {model}."
        )
    if resp.status_code == 503:
        try:
            body = resp.json()
            err_msg = body.get("error") if isinstance(body, dict) else str(body)
            est = body.get("estimated_time") if isinstance(body, dict) else None
            est_text = f" (estimated wait: {int(est)}s)" if est else ""
            raise InferenceError(
                f"Hugging Face model is currently loading{est_text}. Please retry in a few moments."
            )
        except (ValueError, TypeError):
            raise InferenceError(
                "Hugging Face model service is temporarily unavailable (503). Please retry shortly."
            )
    if resp.status_code >= 400:
        detail = resp.text[:200].replace("\n", " ")
        raise InferenceError(f"Hugging Face returned status {resp.status_code}: {detail}")

    try:
        body = resp.json()
    except ValueError as exc:
        raise InferenceError("Hugging Face returned a non-JSON response.") from exc

    if body.get("error"):
        err = body["error"]
        msg = err.get("message") if isinstance(err, dict) else str(err)
        raise InferenceError(f"Hugging Face reported an error: {msg}")

    try:
        content = body["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise InferenceError("Hugging Face returned an unexpected response shape.") from exc

    if isinstance(content, list):
        content = "".join(
            p.get("text", "") for p in content if isinstance(p, dict) and p.get("text")
        )
    if not isinstance(content, str):
        content = ""
    text = content.strip()
    if not text:
        raise InferenceError("The Gemma model returned an empty response.")
    return text


def _messages_with_image(image: validation.ValidatedImage, mode: str = "auto") -> list[dict]:
    sys_prompt = prompts.build_system_prompt(mode)
    user_prompt = prompts.build_user_prompt(mode)
    return [
        {"role": "system", "content": sys_prompt},
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": _data_url(image.data, image.mime)},
                },
                {"type": "text", "text": user_prompt},
            ],
        },
    ]


def _repair_diagram(
    raw_source: str, error: str, model: str, token: str, endpoint: str | None, expected_type: str = "flowchart"
) -> tuple[str | None, str | None]:
    """One corrective retry for Mermaid diagrams."""
    try:
        repair_content = diagram_mod.build_repair_prompt(raw_source, error, expected_type=expected_type)
        fixed = _call(
            [
                {
                    "role": "user",
                    "content": repair_content,
                }
            ],
            model,
            token,
            endpoint=endpoint,
        )
    except InferenceError as exc:
        return None, f"Diagram repair could not run: {exc}"

    try:
        validated = diagram_mod.validate_mermaid(fixed)
        return validated["mermaid"], None
    except diagram_mod.DiagramError as exc:
        return None, f"Diagram repair failed: {exc}"


def analyze(
    image: validation.ValidatedImage, mode: str = "auto"
) -> tuple[validation.Extraction, dict]:
    """Run a full extraction using Hugging Face Gemma inference.

    Returns (extraction, telemetry).
    """
    token = hf_token()
    if not token:
        raise InferenceError(
            "HF_TOKEN is not set. Copy .env.example to .env, set HF_TOKEN with your "
            "Hugging Face access token (from https://huggingface.co/settings/tokens), "
            "and restart the app."
        )

    endpoint = endpoint_url()
    chain = model_chain()
    telemetry = {
        "model": chain[0],
        "model_chain": chain,
        "model_used": None,
        "model_fallbacks": [],
        "mode": mode,
        "endpoint": endpoint or "router.huggingface.co",
        "attempts": 0,
        "diagram_attempts": 0,
        "elapsed_ms": 0,
    }
    started = time.perf_counter()

    text = None
    messages = _messages_with_image(image, mode=mode)
    for candidate in chain:
        try:
            text = _call(messages, candidate, token, endpoint=endpoint)
            telemetry["model_used"] = candidate
            break
        except ModelUnavailable as exc:
            telemetry["model_fallbacks"].append({"model": candidate, "reason": str(exc)[:120]})
            continue

    if text is None:
        raise InferenceError(
            "Every configured Hugging Face model candidate is currently rate limited "
            f"({len(chain)} tried: {', '.join(chain)}). Wait a minute and retry."
        )

    model = telemetry["model_used"]
    telemetry["attempts"] = 1

    payload = validation.extract_json_object(text)
    if payload is None:
        # One corrective repair attempt for malformed JSON
        telemetry["attempts"] = 2
        repair = (
            "Your previous reply could not be parsed as JSON. Reply with ONLY the valid JSON "
            "object matching the schema, with no markdown fences, no notes, and no commentary. "
            f"Here is what you previously returned:\n\n{text[:2000]}"
        )
        try:
            text = _call(
                [
                    {"role": "system", "content": prompts.build_system_prompt(mode)},
                    {"role": "user", "content": repair},
                ],
                model,
                token,
                endpoint=endpoint,
            )
            payload = validation.extract_json_object(text)
        except InferenceError:
            payload = None

    if payload is None:
        telemetry["elapsed_ms"] = int((time.perf_counter() - started) * 1000)
        extraction = validation.normalize_extraction(None, requested_mode=mode)
        extraction.warnings.append(
            "The model replied in a format this app could not parse. Your image was "
            "not lost — please try again."
        )
        return extraction, telemetry

    # Validate the diagram, allowing at most one corrective retry
    validated_diagram = None
    raw_diagram = payload.get("diagram")
    raw_source, explanation = "", ""
    raw_type = "flowchart"
    if isinstance(raw_diagram, dict):
        raw_source = raw_diagram.get("mermaid") or ""
        explanation = raw_diagram.get("explanation") or ""
        raw_type = raw_diagram.get("type") or "flowchart"
    elif isinstance(raw_diagram, str):
        raw_source = raw_diagram

    if raw_source:
        telemetry["diagram_attempts"] = 1
        try:
            validated_diagram = diagram_mod.validate_mermaid(raw_source, explanation)
        except diagram_mod.DiagramError as exc:
            fixed, repair_error = _repair_diagram(
                raw_source, str(exc), model, token, endpoint, expected_type=raw_type
            )
            telemetry["diagram_attempts"] = 2
            if fixed:
                validated_diagram = {"type": raw_type, "mermaid": fixed, "explanation": explanation}
            else:
                telemetry["diagram_repair_error"] = repair_error
                telemetry["diagram_error"] = str(exc)

    extraction = validation.normalize_extraction(
        payload, requested_mode=mode, diagram_validated=validated_diagram
    )
    telemetry["elapsed_ms"] = int((time.perf_counter() - started) * 1000)
    return extraction, telemetry


def key_present() -> bool:
    """Check if Hugging Face token is provided."""
    return bool(hf_token())
