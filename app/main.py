"""Scribble2Structure -- from messy thinking to structured knowledge.

FastAPI app: validation, one inference route, and a results payload the UI can
render without doing any interpretation of its own.
"""

from __future__ import annotations

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from . import ai_client, prompts, validation

app = FastAPI(
    title="Scribble2Structure",
    description="Turn handwritten notes, whiteboards, and sketches into structured knowledge.",
)


@app.get("/api/health")
def health() -> JSONResponse:
    ready = ai_client.key_present()
    return JSONResponse(
        {
            "status": "ok" if ready else "needs_key",
            "api_key": "present" if ready else "missing",
            "model": ai_client.model_id(),
            "supported_types": sorted(validation.ALLOWED_EXTENSIONS),
            "max_mb": validation.MAX_BYTES // (1024 * 1024),
        },
        status_code=200 if ready else 503,
    )


@app.get("/api/config")
def config() -> dict:
    """Model + capability info the UI shows before an upload."""
    return {
        "model": ai_client.model_id(),
        "api_key_present": ai_client.key_present(),
        "max_mb": validation.MAX_BYTES // (1024 * 1024),
        "supported": sorted(validation.ALLOWED_EXTENSIONS),
    }


@app.post("/api/analyze")
async def analyze(image: UploadFile = File(...)) -> JSONResponse:
    raw = await image.read()

    # Validation happens before any token is spent, and before the model id is
    # consulted, so a bad upload fails fast and free.
    try:
        checked = validation.validate_upload(raw, image.content_type, image.filename or "")
    except validation.ValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    try:
        extraction, telemetry = ai_client.analyze(checked)
    except ai_client.InferenceError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    return JSONResponse(
        {
            "title": extraction.title,
            "notes": extraction.notes,
            "takeaways": extraction.takeaways,
            "action_items": extraction.action_items,
            "uncertain_text": extraction.uncertain_text,
            "diagram": extraction.diagram,
            "diagram_explanation": extraction.diagram_explanation,
            "warnings": extraction.warnings,
            "source": {
                "filename": checked.filename,
                "width": checked.width,
                "height": checked.height,
                "bytes": len(checked.data),
            },
            "telemetry": telemetry,
            "system_prompt": prompts.SYSTEM_PROMPT,
        }
    )


@app.get("/")
def index() -> FileResponse:
    return FileResponse("static/index.html")


app.mount("/static", StaticFiles(directory="static"), name="static")
