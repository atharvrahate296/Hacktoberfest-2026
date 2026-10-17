"""Scribble2Structure -- from messy thinking to structured knowledge.

FastAPI app: validation, Hugging Face Gemma inference route, transformation mode
routing, sample gallery endpoint, and structured knowledge responses.
"""

from __future__ import annotations

import os
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from . import ai_client, prompts, validation

app = FastAPI(
    title="Scribble2Structure",
    description="Turn handwritten notes, whiteboards, sketches, and diagrams into structured knowledge.",
)

SAMPLES = [
    {
        "id": "01_ml_notes",
        "title": "Machine Learning Notes",
        "description": "Handwritten gradient descent formulas, learning rate, and L2 regularization.",
        "mode": "notes",
        "filename": "01_ml_notes.png",
        "url": "/samples/01_ml_notes.png",
    },
    {
        "id": "02_os_process",
        "title": "OS Process State Model",
        "description": "Operating system process transitions: New, Ready, Running, Waiting, Terminated.",
        "mode": "flowchart",
        "filename": "02_os_process.png",
        "url": "/samples/02_os_process.png",
    },
    {
        "id": "03_software_architecture",
        "title": "Software Architecture Sketch",
        "description": "Cloud e-commerce microservices, API Gateway, DB replica, Redis, and Kafka.",
        "mode": "architecture",
        "filename": "03_software_architecture.png",
        "url": "/samples/03_software_architecture.png",
    },
    {
        "id": "04_student_productivity",
        "title": "Productivity Brainstorm",
        "description": "Mind map of study habits, task systems, health goals, and tech tools.",
        "mode": "mindmap",
        "filename": "04_student_productivity.png",
        "url": "/samples/04_student_productivity.png",
    },
    {
        "id": "05_meeting_notes",
        "title": "Meeting Sync & Action Items",
        "description": "Sprint planning notes with decisions, confirmed assignees, deadlines, and questions.",
        "mode": "actions",
        "filename": "05_meeting_notes.png",
        "url": "/samples/05_meeting_notes.png",
    },
    {
        "id": "06_database_er",
        "title": "Database ER Diagram",
        "description": "Relational entities (Customer, Order, Item, Product) with PK, FK, and cardinalities.",
        "mode": "er",
        "filename": "06_database_er.png",
        "url": "/samples/06_database_er.png",
    },
]


@app.get("/api/health")
def health() -> JSONResponse:
    ready = ai_client.key_present()
    return JSONResponse(
        {
            "status": "ok" if ready else "needs_key",
            "api_key": "present" if ready else "missing",
            "model": ai_client.model_id(),
            "endpoint": ai_client.endpoint_url() or "local",
            "modes": prompts.MODES,
            "supported_types": sorted(validation.ALLOWED_EXTENSIONS),
            "max_mb": validation.MAX_BYTES // (1024 * 1024),
        },
        status_code=200 if ready else 503,
    )


@app.get("/api/config")
def config() -> dict:
    """Model, capability, and transformation mode info for the UI."""
    return {
        "model": ai_client.model_id(),
        "api_key_present": ai_client.key_present(),
        "endpoint": ai_client.endpoint_url() or "local",
        "max_mb": validation.MAX_BYTES // (1024 * 1024),
        "supported": sorted(validation.ALLOWED_EXTENSIONS),
        "modes": prompts.MODES,
    }


@app.get("/api/samples")
def get_samples() -> list[dict]:
    """Return available sample inputs for the sample gallery."""
    return SAMPLES


@app.post("/api/analyze")
async def analyze(
    image: UploadFile = File(...),
    mode: str = Form("auto"),
) -> JSONResponse:
    raw = await image.read()

    # Pre-inference validation (format, size, dimensions, EXIF)
    try:
        checked = validation.validate_upload(raw, image.content_type, image.filename or "")
    except validation.ValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    try:
        extraction, telemetry = ai_client.analyze(checked, mode=mode)
    except ai_client.InferenceError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    return JSONResponse(
        {
            "title": extraction.title,
            "mode": extraction.mode,
            "detected_mode": extraction.detected_mode,
            "summary": extraction.summary,
            "notes": extraction.notes,
            "takeaways": extraction.takeaways,
            "action_items": extraction.action_items,
            "uncertain_text": extraction.uncertain_text,
            "diagram": extraction.diagram,
            "diagram_explanation": extraction.diagram_explanation,
            "sql_schema": extraction.sql_schema,
            "suggestions": extraction.suggestions,
            "warnings": extraction.warnings,
            "source": {
                "filename": checked.filename,
                "width": checked.width,
                "height": checked.height,
                "bytes": len(checked.data),
            },
            "telemetry": telemetry,
            "system_prompt": prompts.build_system_prompt(extraction.mode),
        }
    )


@app.get("/")
def index() -> FileResponse:
    return FileResponse("static/index.html")


# Mount static assets and sample images
os.makedirs("samples", exist_ok=True)
os.makedirs("static", exist_ok=True)

app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/samples", StaticFiles(directory="samples"), name="samples")
