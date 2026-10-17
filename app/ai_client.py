"""Local Hugging Face transformers inference client for Scribble2Structure.

Handles:
  1. Load google/gemma-3-4b-it once via a module-level singleton (cached after
     first call; never reloads on subsequent requests).
  2. Build a multimodal chat message from the uploaded image + mode prompt.
  3. Run local generate() and decode the reply text.
  4. Parse the JSON object out of the reply (with at most 1 corrective retry).
  5. Validate the Mermaid diagram (with at most 1 corrective retry).
  6. Normalize everything into an Extraction.

No network calls are made for inference. The HF_TOKEN is only used to download
gated model weights on first load; it is never sent to an external API.
"""

from __future__ import annotations

import io
import json
import logging
import os
import time
from threading import Lock

from dotenv import load_dotenv

from . import diagram as diagram_mod
from . import prompts, validation

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

# Local development convenience: a .env file in the project root is honoured.
load_dotenv(override=False)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration helpers
# ---------------------------------------------------------------------------

DEFAULT_GEMMA_MODEL = "google/gemma-3-4b-it"

# Maximum new tokens to generate per request
MAX_NEW_TOKENS = 2048


def _env(name: str) -> str:
    return (os.getenv(name) or "").strip()


def hf_token() -> str:
    """Hugging Face access token (used only for gated model download)."""
    return _env("HF_TOKEN")


# Backward compatibility alias
def api_key() -> str:
    return hf_token()


def model_id() -> str:
    """Configured Hugging Face model identifier."""
    return _env("HF_MODEL_ID") or DEFAULT_GEMMA_MODEL


def endpoint_url() -> str | None:
    """Not applicable for local inference; returns None always."""
    return None


def key_present() -> bool:
    """
    For local inference the token is only needed to download gated weights.
    Returns True if either the weights are already cached or a token is set.
    """
    if hf_token():
        return True
    # Check if transformers cache already has the model weights
    try:
        from huggingface_hub import try_to_load_from_cache  # type: ignore
        result = try_to_load_from_cache(model_id(), "config.json")
        return result is not None
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Public error types (preserved for main.py compatibility)
# ---------------------------------------------------------------------------


class InferenceError(RuntimeError):
    """A user-facing inference failure."""


class ModelUnavailable(InferenceError):
    """This specific model is saturated or unavailable; another may work."""


# ---------------------------------------------------------------------------
# Model singleton – loaded once per process
# ---------------------------------------------------------------------------

_model = None
_processor = None
_device: str = "cpu"
_load_lock = Lock()
_load_error: str | None = None


def _get_model_and_processor():
    """Return (model, processor), loading from disk on first call.

    Thread-safe: concurrent requests block until the first load finishes.
    Raises InferenceError on any loading failure.
    """
    global _model, _processor, _device, _load_error

    if _load_error is not None:
        raise InferenceError(f"Model failed to load previously: {_load_error}")

    if _model is not None and _processor is not None:
        return _model, _processor

    with _load_lock:
        # Double-checked locking
        if _model is not None and _processor is not None:
            return _model, _processor
        if _load_error is not None:
            raise InferenceError(f"Model failed to load previously: {_load_error}")

        try:
            import torch  # type: ignore
            from transformers import AutoProcessor  # type: ignore

            mid = model_id()
            token = hf_token() or None

            logger.info("Loading local model %s …", mid)

            # Determine dtype and device
            if torch.cuda.is_available():
                dtype = torch.bfloat16
                _device = "auto"
                logger.info("CUDA available – loading with device_map='auto', bfloat16")
            else:
                dtype = torch.float32
                _device = "cpu"
                logger.info("No CUDA – loading on CPU with float32 (slow but functional)")

            # Try vision-capable model classes in order of preference
            model_obj = None
            load_kwargs = dict(
                torch_dtype=dtype,
                device_map=_device if _device != "cpu" else None,
                token=token,
            )

            for cls_name in (
                "Gemma3ForConditionalGeneration",
                "AutoModelForImageTextToText",
                "AutoModelForVision2Seq",
            ):
                try:
                    if cls_name == "Gemma3ForConditionalGeneration":
                        from transformers import Gemma3ForConditionalGeneration  # type: ignore
                        model_obj = Gemma3ForConditionalGeneration.from_pretrained(
                            mid, **load_kwargs
                        )
                    elif cls_name == "AutoModelForImageTextToText":
                        from transformers import AutoModelForImageTextToText  # type: ignore
                        model_obj = AutoModelForImageTextToText.from_pretrained(
                            mid, **load_kwargs
                        )
                    else:
                        from transformers import AutoModelForVision2Seq  # type: ignore
                        model_obj = AutoModelForVision2Seq.from_pretrained(
                            mid, **load_kwargs
                        )
                    logger.info("Loaded model with class %s", cls_name)
                    break
                except (ImportError, OSError, ValueError) as exc:
                    logger.debug("Class %s failed: %s", cls_name, exc)
                    continue

            if model_obj is None:
                raise InferenceError(
                    f"Could not load '{mid}' with any supported model class. "
                    "Ensure transformers>=4.50 and the model weights are available."
                )

            proc = AutoProcessor.from_pretrained(mid, token=token)

            if _device == "cpu":
                model_obj = model_obj.to("cpu")

            model_obj.eval()

            _model = model_obj
            _processor = proc
            logger.info("Model %s ready.", mid)
            return _model, _processor

        except InferenceError:
            raise
        except MemoryError as exc:
            _load_error = "Out of memory while loading model weights."
            raise InferenceError(_load_error) from exc
        except Exception as exc:
            _load_error = str(exc)[:300]
            raise InferenceError(
                f"Failed to load local model '{model_id()}': {_load_error}"
            ) from exc


# ---------------------------------------------------------------------------
# Image helper
# ---------------------------------------------------------------------------


def _pil_from_validated(image: validation.ValidatedImage):
    """Convert a ValidatedImage (bytes) to a PIL Image for the processor."""
    try:
        from PIL import Image  # type: ignore
        return Image.open(io.BytesIO(image.data)).convert("RGB")
    except Exception as exc:
        raise InferenceError(f"Could not decode image for model input: {exc}") from exc


# ---------------------------------------------------------------------------
# Core generation
# ---------------------------------------------------------------------------


def _generate(
    pil_image,
    system_prompt: str,
    user_prompt: str,
) -> str:
    """Run one round of local generation. Returns raw text."""
    import torch  # type: ignore

    model, processor = _get_model_and_processor()

    # Build the chat messages list with an image attachment
    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": system_prompt}],
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": pil_image},
                {"type": "text", "text": user_prompt},
            ],
        },
    ]

    try:
        # apply_chat_template produces the token ids + pixel values
        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
    except Exception:
        # Fallback: some processor versions use a different API
        try:
            text_part = processor.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False
            )
            inputs = processor(
                text=[text_part],
                images=[pil_image],
                return_tensors="pt",
                padding=True,
            )
        except Exception as exc2:
            raise InferenceError(
                f"Processor could not encode the image and prompt: {exc2}"
            ) from exc2

    # Move tensors to the model device
    try:
        if hasattr(model, "device") and str(model.device) != "cpu":
            inputs = {k: v.to(model.device) for k, v in inputs.items() if hasattr(v, "to")}
        elif _device != "cpu":
            dev = next(model.parameters()).device
            inputs = {k: v.to(dev) for k, v in inputs.items() if hasattr(v, "to")}
    except Exception:
        pass  # leave on default device; generation may still succeed

    input_len = inputs.get("input_ids", torch.tensor([])).shape[-1] if "input_ids" in inputs else 0

    try:
        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                temperature=None,
                top_p=None,
            )
    except RuntimeError as exc:
        msg = str(exc)
        if "out of memory" in msg.lower() or "CUDA out of memory" in msg:
            raise InferenceError(
                "GPU ran out of memory during generation. Try a smaller image or "
                "restart the server to free memory."
            ) from exc
        raise InferenceError(f"Generation failed: {msg[:200]}") from exc
    except Exception as exc:
        raise InferenceError(f"Generation failed unexpectedly: {str(exc)[:200]}") from exc

    # Decode only the newly generated tokens (exclude the prompt)
    try:
        new_tokens = generated[:, input_len:]
        text = processor.batch_decode(
            new_tokens,
            skip_special_tokens=True,
        )[0].strip()
    except Exception as exc:
        raise InferenceError(f"Token decoding failed: {exc}") from exc

    if not text:
        raise InferenceError("The model returned an empty response.")

    return text


def _generate_text_only(prompt: str) -> str:
    """Text-only generation for corrective retries (no image)."""
    import torch  # type: ignore

    model, processor = _get_model_and_processor()

    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]

    try:
        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
    except Exception:
        try:
            text_part = processor.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False
            )
            inputs = processor(text=[text_part], return_tensors="pt", padding=True)
        except Exception as exc2:
            raise InferenceError(f"Could not encode repair prompt: {exc2}") from exc2

    try:
        if _device != "cpu":
            dev = next(model.parameters()).device
            inputs = {k: v.to(dev) for k, v in inputs.items() if hasattr(v, "to")}
    except Exception:
        pass

    input_len = inputs.get("input_ids", torch.tensor([])).shape[-1] if "input_ids" in inputs else 0

    try:
        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                temperature=None,
                top_p=None,
            )
    except RuntimeError as exc:
        raise InferenceError(f"Text repair generation failed: {str(exc)[:200]}") from exc

    try:
        new_tokens = generated[:, input_len:]
        text = processor.batch_decode(new_tokens, skip_special_tokens=True)[0].strip()
    except Exception as exc:
        raise InferenceError(f"Token decoding failed during repair: {exc}") from exc

    if not text:
        raise InferenceError("The model returned an empty response during repair.")
    return text


# ---------------------------------------------------------------------------
# Diagram repair
# ---------------------------------------------------------------------------


def _repair_diagram(
    raw_source: str, error: str, expected_type: str = "flowchart"
) -> tuple[str | None, str | None]:
    """One corrective retry for Mermaid diagrams."""
    try:
        repair_content = diagram_mod.build_repair_prompt(raw_source, error, expected_type=expected_type)
        fixed = _generate_text_only(repair_content)
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


def analyze(
    image: validation.ValidatedImage, mode: str = "auto"
) -> tuple[validation.Extraction, dict]:
    """Run a full extraction using local Gemma inference.

    Returns (extraction, telemetry).
    """
    mid = model_id()
    telemetry: dict = {
        "model": mid,
        "model_chain": [mid],
        "model_used": mid,
        "model_fallbacks": [],
        "mode": mode,
        "endpoint": "local",
        "attempts": 0,
        "diagram_attempts": 0,
        "elapsed_ms": 0,
    }
    started = time.perf_counter()

    sys_prompt = prompts.build_system_prompt(mode)
    usr_prompt = prompts.build_user_prompt(mode)
    pil_image = _pil_from_validated(image)

    # Primary generation attempt
    try:
        text = _generate(pil_image, sys_prompt, usr_prompt)
        telemetry["attempts"] = 1
    except InferenceError:
        telemetry["elapsed_ms"] = int((time.perf_counter() - started) * 1000)
        raise

    # Parse JSON
    payload = validation.extract_json_object(text)
    if payload is None:
        # One corrective JSON-repair attempt (text-only)
        telemetry["attempts"] = 2
        repair = (
            "Your previous reply could not be parsed as JSON. Reply with ONLY the valid JSON "
            "object matching the schema, with no markdown fences, no notes, and no commentary. "
            f"Here is what you previously returned:\n\n{text[:2000]}"
        )
        try:
            text = _generate_text_only(repair)
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

    # Validate diagram, with one corrective retry
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
            fixed, repair_error = _repair_diagram(raw_source, str(exc), expected_type=raw_type)
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
