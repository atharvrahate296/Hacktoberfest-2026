"""Input validation and model-response normalization.

Two jobs, kept separate because they fail for unrelated reasons:

* ``validate_upload`` runs before inference and decides whether we are willing
  to spend a request. Handles file size, extensions, content sniffing, integrity,
  and EXIF orientation transposition.
* ``normalize_extraction`` runs after inference and ensures the model's
  JSON is validated and safe to render.
"""

from __future__ import annotations

import io
import json
import os
import re
from dataclasses import dataclass, field
from typing import Any

# Magic-byte prefixes. A declared Content-Type from a browser is a hint, not
# evidence, so we look at the actual bytes before trusting the extension.
_SIGNATURES: list[tuple[bytes, str]] = [
    (b"\x89PNG\r\n\x1a\n", "image/png"),
    (b"\xff\xd8\xff", "image/jpeg"),
    (b"GIF87a", "image/gif"),
    (b"GIF89a", "image/gif"),
]

_WEBP_PREFIX = b"RIFF"
_WEBP_MARK = b"WEBP"

ALLOWED_MIME = {"image/png", "image/jpeg", "image/webp"}
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}

def _get_max_bytes() -> int:
    try:
        return int(os.getenv("MAX_UPLOAD_MB", "10")) * 1024 * 1024
    except (ValueError, TypeError):
        return 10 * 1024 * 1024

MAX_BYTES = _get_max_bytes()
MAX_PIXELS = 40_000_000  # ~40 MP; guards decompression-bomb style inputs

_MIME_BY_EXT = {
    "png": "image/png",
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
    "webp": "image/webp",
}


class ValidationError(ValueError):
    """Raised when an upload is unusable. Message is safe to show a user."""


@dataclass
class ValidatedImage:
    data: bytes
    mime: str
    filename: str
    width: int | None = None
    height: int | None = None
    extension: str = ""


def sniff_mime(head: bytes) -> str | None:
    """Return the MIME type implied by the file's leading bytes, or None."""
    for sig, mime in _SIGNATURES:
        if head.startswith(sig):
            return mime
    if head.startswith(_WEBP_PREFIX) and _WEBP_MARK in head[:16]:
        return "image/webp"
    return None


def validate_upload(
    data: bytes, declared_mime: str | None, filename: str
) -> ValidatedImage:
    """Validate an upload on both declared type and real content.

    Applies EXIF orientation correction and dimension checks.
    Raises ValidationError with a user-facing message on any failure.
    """
    name = (filename or "").strip()
    if not name:
        raise ValidationError("No filename provided.")

    ext = name.rsplit(".", 1)[-1].lower() if "." in name else ""
    if ext not in ALLOWED_EXTENSIONS:
        raise ValidationError(
            f"'{name}' is not a supported file type. Use PNG, JPG, JPEG, or WEBP."
        )

    if not data:
        raise ValidationError("The uploaded file is empty.")

    max_b = _get_max_bytes()
    if len(data) > max_b:
        raise ValidationError(
            f"Image is {len(data) // (1024 * 1024)} MB; the limit is "
            f"{max_b // (1024 * 1024)} MB."
        )

    actual = sniff_mime(data[:32])
    if actual is None:
        raise ValidationError(
            "That file does not appear to be a real image. The bytes do not match "
            "PNG, JPEG, GIF, or WEBP."
        )
    if actual == "image/gif":
        raise ValidationError("GIF is not supported. Please upload a PNG, JPG, or WEBP.")

    # The declared type and the extension must agree with the real bytes.
    if declared_mime and declared_mime.startswith("image/") and declared_mime != actual:
        raise ValidationError(
            f"File content is {actual} but the upload declared {declared_mime}. "
            "Please re-export the image and try again."
        )
    if _MIME_BY_EXT[ext] != actual:
        raise ValidationError(
            f"File is named '.{ext}' but its content is {actual}. Rename it to match "
            "or re-export it."
        )

    width = height = None
    try:
        from PIL import Image, ImageOps

        with Image.open(io.BytesIO(data)) as im:
            im.verify()

        with Image.open(io.BytesIO(data)) as im:
            # Correct EXIF orientation where applicable
            transposed = ImageOps.exif_transpose(im)
            if transposed is not None and transposed is not im:
                im = transposed
                buf = io.BytesIO()
                fmt = "PNG" if actual == "image/png" else ("WEBP" if actual == "image/webp" else "JPEG")
                im.save(buf, format=fmt)
                data = buf.getvalue()

            width, height = im.size

        if width * height > MAX_PIXELS:
            raise ValidationError(
                f"Image is {width}x{height} pixels, which exceeds the "
                f"{MAX_PIXELS // 1_000_000} MP limit."
            )
    except ValidationError:
        raise
    except Exception:
        # Truncated or otherwise corrupt files fail decode. Content sniffing
        # already passed, so this is a genuine corruption signal.
        raise ValidationError("That image file is corrupt or incomplete.")

    return ValidatedImage(
        data=data, mime=actual, filename=name, width=width, height=height, extension=ext
    )


class _BytesReader:
    """Minimal file-like wrapper so Pillow can open from memory."""

    def __init__(self, data: bytes) -> None:
        self._data = data
        self._pos = 0

    def read(self, n: int = -1) -> bytes:
        if n is None or n < 0:
            out = self._data[self._pos :]
            self._pos = len(self._data)
            return out
        out = self._data[self._pos : self._pos + n]
        self._pos += len(out)
        return out

    def seek(self, pos: int, whence: int = 0) -> int:
        if whence == 0:
            self._pos = pos
        elif whence == 1:
            self._pos += pos
        else:
            self._pos = len(self._data) + pos
        return self._pos

    def tell(self) -> int:
        return self._pos


# --------------------------------------------------------------------------
# Model response normalization
# --------------------------------------------------------------------------

_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)

VALID_CONFIDENCE = {"high", "medium", "low"}


@dataclass
class Extraction:
    """A validated, renderable extraction. Every field is safe to display."""

    title: str
    mode: str = "auto"
    detected_mode: str | None = None
    summary: str = ""
    notes: list[dict[str, Any]] = field(default_factory=list)
    takeaways: list[str] = field(default_factory=list)
    action_items: list[dict[str, Any]] = field(default_factory=list)
    uncertain_text: list[dict[str, str]] = field(default_factory=list)
    diagram: dict[str, str] | None = None
    diagram_explanation: str = ""
    sql_schema: str | None = None
    suggestions: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def extract_json_object(text: str) -> dict[str, Any] | None:
    """Pull a JSON object out of a model response.

    Handles bare JSON, fenced JSON, and JSON surrounded by preamble prose.
    Returns None if no parseable object is found.
    """
    if not text:
        return None

    candidates: list[str] = []
    fenced = _FENCE.findall(text)
    candidates.extend(fenced)

    stripped = text.strip()
    candidates.append(stripped)

    # Scan for the first balanced {...} block, respecting strings and escapes.
    start = stripped.find("{")
    while start != -1:
        depth = 0
        in_str = False
        esc = False
        for i in range(start, len(stripped)):
            ch = stripped[i]
            if esc:
                esc = False
                continue
            if ch == "\\":
                esc = True
                continue
            if ch == '"':
                in_str = not in_str
                continue
            if in_str:
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidates.append(stripped[start : i + 1])
                    start = -1
                    break
        if start == -1:
            break
        start = stripped.find("{", start + 1)

    for cand in candidates:
        cand = cand.strip()
        if not cand:
            continue
        try:
            parsed = json.loads(cand)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def _as_str_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    out = []
    for item in value:
        if isinstance(item, str) and item.strip():
            out.append(item.strip())
        elif isinstance(item, dict):
            for key in ("text", "content", "item", "takeaway", "suggestion"):
                if isinstance(item.get(key), str) and item[key].strip():
                    out.append(item[key].strip())
                    break
    return out


def _as_notes(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    out = []
    for item in value:
        if isinstance(item, str):
            if item.strip():
                out.append({"heading": "Notes", "content": [item.strip()], "confidence": "medium"})
            continue
        if not isinstance(item, dict):
            continue
        content = _as_str_list(item.get("content")) or _as_str_list(item.get("lines"))
        if not content:
            continue
        heading = item.get("heading") or item.get("title") or "Notes"
        conf = str(item.get("confidence", "medium")).strip().lower()
        if conf not in VALID_CONFIDENCE:
            conf = "medium"
        out.append(
            {
                "heading": str(heading).strip()[:200],
                "content": content,
                "confidence": conf,
            }
        )
    return out


def _as_action_items(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    out = []
    for item in value:
        if isinstance(item, str):
            if item.strip():
                out.append({"task": item.strip(), "owner": None, "deadline": None})
            continue
        if not isinstance(item, dict):
            continue
        task = item.get("task") or item.get("action") or item.get("text")
        if not isinstance(task, str) or not task.strip():
            continue
        owner = item.get("owner")
        deadline = item.get("deadline")
        out.append(
            {
                "task": task.strip(),
                # Never let a model invent an owner or a date. Empty strings and
                # the literal "null"/"none" all collapse to None.
                "owner": (
                    owner.strip()
                    if isinstance(owner, str)
                    and owner.strip().lower() not in {"null", "none", "n/a", "unknown", "", "unassigned"}
                    else None
                ),
                "deadline": (
                    deadline.strip()
                    if isinstance(deadline, str)
                    and deadline.strip().lower() not in {"null", "none", "n/a", "unknown", "", "tbd"}
                    else None
                ),
            }
        )
    return out


def _as_uncertain(value: Any) -> list[dict[str, str]]:
    if not isinstance(value, list):
        return []
    out = []
    for item in value:
        if isinstance(item, str) and item.strip():
            out.append({"text_or_location": item.strip(), "reason": "Flagged as uncertain"})
            continue
        if not isinstance(item, dict):
            continue
        loc = item.get("text_or_location") or item.get("location") or item.get("text")
        reason = item.get("reason") or item.get("why") or "Illegible or ambiguous"
        if isinstance(loc, str) and loc.strip():
            out.append({"text_or_location": loc.strip(), "reason": str(reason).strip()})
    return out


def normalize_extraction(
    payload: dict[str, Any] | None,
    *,
    requested_mode: str = "auto",
    diagram_validated: dict | None = None,
) -> Extraction:
    """Coerce arbitrary model JSON into a renderable Extraction.

    Never raises. Missing or wrong-typed fields degrade to empty rather than
    crashing, because notes and text must survive even when diagram fails.
    """
    warnings: list[str] = []
    if payload is None:
        return Extraction(
            title="Extraction unavailable",
            mode=requested_mode,
            warnings=["The model did not return a parseable JSON object."],
        )

    title = payload.get("title")
    title = str(title).strip()[:160] if isinstance(title, str) and title.strip() else "Untitled Extraction"

    raw_detected = payload.get("detected_mode")
    detected_mode = (
        str(raw_detected).strip().lower()
        if isinstance(raw_detected, str) and raw_detected.strip()
        else (requested_mode if requested_mode != "auto" else None)
    )

    summary = payload.get("summary")
    summary = str(summary).strip() if isinstance(summary, str) and summary.strip() else ""

    notes = _as_notes(payload.get("notes"))
    takeaways = _as_str_list(payload.get("takeaways"))
    actions = _as_action_items(payload.get("action_items"))
    uncertain = _as_uncertain(payload.get("uncertain_text"))
    suggestions = _as_str_list(payload.get("suggestions"))

    raw_sql = payload.get("sql_schema")
    sql_schema = (
        str(raw_sql).strip()
        if isinstance(raw_sql, str) and raw_sql.strip()
        else None
    )

    raw_diagram = payload.get("diagram")
    diagram: dict[str, str] | None = None
    explanation = payload.get("diagram_explanation")
    explanation = (
        str(explanation).strip()
        if isinstance(explanation, str) and explanation.strip()
        else ""
    )

    if isinstance(raw_diagram, dict):
        validated = diagram_validated if diagram_validated is not None else None
        if validated and validated.get("mermaid"):
            diagram = {
                "type": validated.get("type", "flowchart"),
                "mermaid": validated["mermaid"],
            }
            if not explanation:
                explanation = validated.get("explanation", "")
        else:
            warnings.append(
                "A diagram was generated but did not pass Mermaid validation, so it "
                "was omitted. The structured notes and takeaways are unaffected."
            )
    elif isinstance(raw_diagram, str) and raw_diagram.strip():
        validated = diagram_validated if diagram_validated is not None else None
        if validated and validated.get("mermaid"):
            diagram = {"type": validated.get("type", "flowchart"), "mermaid": validated["mermaid"]}
        else:
            warnings.append(
                "Diagram source failed validation and was omitted; notes are unaffected."
            )

    if not notes and not takeaways and not actions:
        warnings.append(
            "No readable structured text was found in this image. Try a sharper photo with "
            "better lighting or tighter cropping."
        )

    return Extraction(
        title=title,
        mode=requested_mode,
        detected_mode=detected_mode,
        summary=summary,
        notes=notes,
        takeaways=takeaways,
        action_items=actions,
        uncertain_text=uncertain,
        diagram=diagram,
        diagram_explanation=explanation,
        sql_schema=sql_schema,
        suggestions=suggestions,
        warnings=warnings,
    )
