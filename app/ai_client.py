"""Local Hugging Face transformers inference client for Scribble2Structure.

Handles:
  1. Load the configured VLM once via a module-level singleton (cached after
     first call; never reloads on subsequent requests).
  2. Build a multimodal chat message from the uploaded image + mode prompt.
  3. Run local generate() and decode the reply text.
  4. Parse the JSON object out of the reply (with at most 1 corrective retry).
  5. Validate the Mermaid diagram (with at most 1 corrective retry).
  6. Normalize everything into an Extraction.

GPU: Uses CUDA float16 — the model is placed on the GPU directly when it fits
     in VRAM, otherwise device_map="auto" with a 3 GiB ceiling offloads the
     overflow to CPU RAM.
CPU: Falls back to float32 on CPU when no CUDA device is available.
"""

from __future__ import annotations

import io
import json
import logging
import os
import time
from contextlib import nullcontext
from threading import Lock

from dotenv import load_dotenv

from . import diagram as diagram_mod
from . import prompts, validation

load_dotenv(override=False)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_MODEL = "HuggingFaceTB/SmolVLM-500M-Instruct"
MAX_NEW_TOKENS = 1536

# How much of the model's own reply to keep for diagnosis when a run yields
# nothing. Enough to see what went wrong, short enough to return in JSON.
RAW_PREVIEW_CHARS = 1200

# Model choices the loader is expected to handle, smallest first. Documented in
# readme.md and .env.example; keep the default in sync with both.
KNOWN_MODELS = (
    "HuggingFaceTB/SmolVLM-500M-Instruct",
    "Qwen/Qwen2-VL-2B-Instruct",
    "Qwen/Qwen2.5-VL-3B-Instruct",
    "google/gemma-3-4b-it",
)

# VRAM ceiling for device_map="auto" — keeps a buffer for activations/KV cache
_GPU_MEM_CAP = "3GiB"
# CPU RAM to allow as overflow (adjust if your machine has less RAM)
_CPU_MEM_CAP = "6GiB"


def _env(name: str) -> str:
    return (os.getenv(name) or "").strip()


def hf_token() -> str:
    return _env("HF_TOKEN")


def api_key() -> str:
    return hf_token()


def model_id() -> str:
    """The configured model, falling back to the shipped default."""
    return _env("HF_MODEL_ID") or DEFAULT_MODEL


def weights_cached() -> bool:
    """True when the configured model's config is already in the HF cache."""
    try:
        from huggingface_hub import try_to_load_from_cache  # type: ignore

        return try_to_load_from_cache(model_id(), "config.json") is not None
    except Exception:
        return False


def endpoint_url() -> str | None:
    return None


def key_present() -> bool:
    """True when inference can run: either a token is set or weights are cached."""
    return bool(hf_token()) or weights_cached()


# ---------------------------------------------------------------------------
# Public error types
# ---------------------------------------------------------------------------


class InferenceError(RuntimeError):
    """A user-facing inference failure."""


class ModelUnavailable(InferenceError):
    """Model saturated or unavailable."""


# ---------------------------------------------------------------------------
# Singleton loader
# ---------------------------------------------------------------------------

_model = None
_processor = None
_model_device: str = "cpu"
_load_lock = Lock()
_load_error: str | None = None


def _is_retryable(exc: BaseException) -> bool:
    """True when a load failure is environmental and worth retrying later.

    Out-of-memory and broken shared-library loads are transient (another process
    freed VRAM, the user closed a memory-hungry app). Memoizing those would keep
    the server permanently broken until a restart.
    """
    if isinstance(exc, MemoryError):
        return True
    if isinstance(exc, OSError):
        return True
    text = str(exc).lower()
    return "out of memory" in text or "cuda" in text


def _get_model_and_processor():
    """Return (model, processor), loading from disk on first call (thread-safe)."""
    global _model, _processor, _model_device, _load_error

    if _load_error is not None:
        raise InferenceError(f"Model failed to load previously: {_load_error}")
    if _model is not None and _processor is not None:
        return _model, _processor

    with _load_lock:
        if _model is not None and _processor is not None:
            return _model, _processor
        if _load_error is not None:
            raise InferenceError(f"Model failed to load previously: {_load_error}")

        try:
            import torch  # type: ignore
            from transformers import AutoProcessor  # type: ignore

            mid = model_id()
            token = hf_token() or None

            cuda_ok = torch.cuda.is_available()
            if cuda_ok:
                dtype = torch.float16          # fp16 — efficient on RTX 3050
                _model_device = "cuda"
                logger.info(
                    "CUDA detected — loading '%s' in float16 (GPU cap: %s, "
                    "CPU overflow: %s)", mid, _GPU_MEM_CAP, _CPU_MEM_CAP,
                )
            else:
                dtype = torch.float32
                dev_map = None
                max_mem = None
                _model_device = "cpu"
                logger.info("No CUDA — loading '%s' on CPU in float32", mid)

            load_kwargs: dict = dict(
                torch_dtype=dtype,
                low_cpu_mem_usage=True,
                token=token,
            )

            model_obj = None
            _last_load_error = "no candidate model class available"

            # Placement attempts: whole model on the GPU first (fastest), then
            # device_map="auto" with a VRAM ceiling so accelerate offloads the
            # overflow to CPU instead of raising CUDA OOM.
            placements: list[tuple] = (
                [("cuda", None), ("auto", {0: _GPU_MEM_CAP, "cpu": _CPU_MEM_CAP})]
                if cuda_ok
                else [(None, None)]
            )

            # Candidate model classes: the architecture the checkpoint actually
            # declares comes first, then generic Auto classes. Guessing a fixed
            # order loads a wrong-architecture model silently, which fails later
            # during generation instead of here.
            candidates: list[str] = []
            try:
                import transformers  # type: ignore
                from transformers import AutoConfig  # type: ignore

                cfg = AutoConfig.from_pretrained(mid, token=token)
                for arch in getattr(cfg, "architectures", None) or []:
                    if arch not in candidates:
                        candidates.append(arch)
            except Exception as exc:
                logger.debug("Could not read model architecture: %s", exc)

            for fallback in ("AutoModelForImageTextToText", "AutoModelForVision2Seq"):
                if fallback not in candidates:
                    candidates.append(fallback)

            import transformers as _tf  # type: ignore

            for dev_map, max_mem in placements:
                kwargs = dict(load_kwargs)
                if dev_map:
                    kwargs["device_map"] = dev_map
                if max_mem:
                    kwargs["max_memory"] = max_mem

                for cls_name in candidates:
                    model_cls = getattr(_tf, cls_name, None)
                    if model_cls is None or not hasattr(model_cls, "from_pretrained"):
                        logger.debug("transformers has no class %s", cls_name)
                        continue
                    try:
                        model_obj = model_cls.from_pretrained(mid, **kwargs)
                        logger.info(
                            "Loaded model with class %s (device_map=%s)",
                            cls_name, dev_map,
                        )
                        break
                    except (ImportError, OSError, ValueError, TypeError, RuntimeError) as exc:
                        model_obj = None
                        _last_load_error = f"{cls_name}: {exc}"
                        logger.debug("Class %s not usable: %s", cls_name, exc)
                        continue

                if model_obj is not None:
                    break

            if model_obj is None:
                raise InferenceError(
                    f"Could not load '{mid}' with any supported model class. "
                    f"Last error — {_last_load_error}"
                )

            proc = AutoProcessor.from_pretrained(mid, token=token)

            if not cuda_ok:
                model_obj = model_obj.to("cpu")

            model_obj.eval()
            _model = model_obj
            _processor = proc

            if cuda_ok:
                allocated = torch.cuda.memory_allocated(0) / 1024 ** 3
                total = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
                logger.info(
                    "Model ready on GPU. VRAM used: %.2f / %.2f GiB", allocated, total
                )
            else:
                logger.info("Model ready on CPU.")

            return _model, _processor

        except InferenceError:
            raise
        except MemoryError as exc:
            _load_error = "Out of memory loading model weights."
            raise InferenceError(_load_error) from exc
        except Exception as exc:
            if _is_retryable(exc):
                _load_error = None
            else:
                _load_error = str(exc)[:400]
            raise InferenceError(
                f"Failed to load local model '{model_id()}': {str(exc)[:400]}"
            ) from exc


# ---------------------------------------------------------------------------
# Image helper
# ---------------------------------------------------------------------------


def _pil_from_validated(image: validation.ValidatedImage):
    try:
        from PIL import Image  # type: ignore
        return Image.open(io.BytesIO(image.data)).convert("RGB")
    except Exception as exc:
        raise InferenceError(f"Could not decode image for model input: {exc}") from exc


# ---------------------------------------------------------------------------
# Core generation helpers
# ---------------------------------------------------------------------------


def _inputs_to_device(inputs: dict, model):
    """Move all tensor values in inputs to the same device as the model."""
    torch = _torch()
    if torch is None or _model_device != "cuda":
        return inputs
    try:
        if not torch.cuda.is_available():
            return inputs
        dev = next(model.parameters()).device
        return {k: v.to(dev) if hasattr(v, "to") else v for k, v in inputs.items()}
    except Exception:
        return inputs


def _build_inputs(processor, messages: list[dict], pil_image=None):
    """
    Build processor inputs compatible with Qwen2-VL, Gemma3, and other VLMs.

    Strategy:
      1. apply_chat_template(tokenize=False) → text string
      2. processor(text, images) → tokenized tensor dict
    Falls back to the single-call tokenize=True API if step 1 fails.
    """
    # --- Two-step (recommended for Qwen2-VL, SmolVLM and most modern VLMs) ---
    # `except ... as name` unbinds the name at block exit, so both failures are
    # copied into names that outlive their handler.
    two_step_error: Exception | None = None
    try:
        text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        kwargs: dict = dict(text=[text], padding=True, return_tensors="pt")
        if pil_image is not None:
            kwargs["images"] = [pil_image]
        return processor(**kwargs)
    except Exception as exc:
        two_step_error = exc
        logger.debug("Two-step processor call failed (%s), trying single-call", exc)

    # --- Single-call fallback (Gemma3 / older VLMs) -------------------------
    # The image must be forwarded here too: dropping it would make the model
    # describe a prompt with no visual input at all.
    try:
        template_kwargs: dict = {}
        if pil_image is not None:
            template_kwargs["images"] = [pil_image]
        return processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            **template_kwargs,
        )
    except Exception as exc:
        raise InferenceError(
            f"Processor could not encode the prompt/image. "
            f"Two-step error: {two_step_error}. Single-call error: {exc}."
        ) from exc


_torch_cache: list = []  # [torch] on success, [None] once an import has failed


def _torch():
    """Import torch if it is usable here, else None.

    The failure is remembered: a broken install is not fixed between requests,
    and re-importing a DLL that cannot load is slow and can destabilize the
    process.
    """
    if _torch_cache:
        return _torch_cache[0]
    try:
        import torch  # type: ignore

        _torch_cache.append(torch)
        return torch
    except Exception as exc:  # broken or missing CUDA/cuDNN install, low memory
        logger.warning("torch is unavailable: %s", exc)
        _torch_cache.append(None)
        return None


def _run_generate(model, inputs: dict):
    """Call model.generate(), handling OOM with a clear error."""
    torch = _torch()
    ctx = torch.inference_mode() if torch is not None else nullcontext()
    try:
        with ctx:
            return model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
            )
    except InferenceError:
        raise
    except RuntimeError as exc:
        msg = str(exc)
        if "out of memory" in msg.lower():
            # Free the cache and surface a clean user error
            if torch is not None and torch.cuda.is_available():
                torch.cuda.empty_cache()
            raise InferenceError(
                "GPU ran out of memory during generation. "
                "Try a smaller image, or reduce MAX_NEW_TOKENS in ai_client.py."
            ) from exc
        raise InferenceError(f"Generation failed: {msg[:300]}") from exc
    except Exception as exc:
        raise InferenceError(f"Generation failed unexpectedly: {str(exc)[:300]}") from exc


def _decode(processor, generated, input_len: int) -> str:
    """Decode only the newly generated tokens (strip the prompt prefix)."""
    try:
        new_tokens = generated[:, input_len:]
        text = processor.batch_decode(new_tokens, skip_special_tokens=True)[0].strip()
    except Exception as exc:
        raise InferenceError(f"Token decoding failed: {exc}") from exc
    if not text:
        raise InferenceError("The model returned an empty response.")
    return text


# ---------------------------------------------------------------------------
# Public generation functions
# ---------------------------------------------------------------------------


def _generate(pil_image, system_prompt: str, user_prompt: str) -> str:
    """Run one multimodal generation round. Returns raw text."""
    model, processor = _get_model_and_processor()

    messages = [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": pil_image},
                {"type": "text", "text": user_prompt},
            ],
        },
    ]

    inputs = _build_inputs(processor, messages, pil_image=pil_image)
    input_len = inputs["input_ids"].shape[-1] if "input_ids" in inputs else 0
    inputs = _inputs_to_device(inputs, model)

    generated = _run_generate(model, inputs)
    return _decode(processor, generated, input_len)


def _generate_text_only(prompt: str) -> str:
    """Text-only generation for corrective retries (no image)."""
    model, processor = _get_model_and_processor()

    messages = [
        {"role": "user", "content": [{"type": "text", "text": prompt}]}
    ]

    inputs = _build_inputs(processor, messages, pil_image=None)
    input_len = inputs["input_ids"].shape[-1] if "input_ids" in inputs else 0
    inputs = _inputs_to_device(inputs, model)

    generated = _run_generate(model, inputs)
    return _decode(processor, generated, input_len)


# ---------------------------------------------------------------------------
# Diagram repair
# ---------------------------------------------------------------------------


def _repair_diagram(
    raw_source: str, error: str, expected_type: str = "flowchart"
) -> tuple[str | None, str | None]:
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


def analyze(
    image: validation.ValidatedImage, mode: str = "auto"
) -> tuple[validation.Extraction, dict]:
    """Run a full extraction using local VLM inference.

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

    try:
        text = _generate(pil_image, sys_prompt, usr_prompt)
        telemetry["attempts"] = 1
    except InferenceError:
        telemetry["elapsed_ms"] = int((time.perf_counter() - started) * 1000)
        raise

    payload = validation.extract_json_object(text)
    if payload is None:
        telemetry["attempts"] = 2
        repair = (
            "Your previous reply could not be parsed as JSON. Reply with ONLY the valid JSON "
            "object matching the schema — no markdown fences, no notes, no commentary. "
            f"Here is what you previously returned:\n\n{text[:2000]}"
        )
        try:
            text = _generate_text_only(repair)
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
            retry_text = _generate(pil_image, sys_prompt, f"{usr_prompt}\n\n{strict}")
            telemetry["attempts"] = 2
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
