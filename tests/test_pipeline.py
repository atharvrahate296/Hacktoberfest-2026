"""End-to-end pipeline test with a scripted cloud model.

Proves the whole path -- validate -> infer -> normalize -> JSON payload the UI
consumes -- without a network call or a real API key.
"""

from __future__ import annotations

import io
import json

from fastapi.testclient import TestClient

from app import ai_client, main

from conftest import FakeAPIError, reply_json

client = TestClient(main.app)

REALISTIC_RESPONSE = reply_json(
    {
        "title": "Cache Invalidation Strategy",
        "detected_mode": "flowchart",
        "summary": "TTL combined with write-through cache invalidation to eliminate stale replica reads.",
        "notes": [
            {
                "heading": "Problem",
                "content": [
                    "Read replicas serve stale data after a write",
                    "Observed on the product listing page",
                ],
                "confidence": "high",
            },
            {
                "heading": "Options considered",
                "content": ["TTL only - simple but serves stale data", "Write-through invalidation"],
                "confidence": "medium",
            },
        ],
        "takeaways": [
            "TTL alone is not enough; pair it with invalidation on write",
            "The listing page is the only confirmed stale surface",
        ],
        "action_items": [
            {"task": "Add TTL to the cache layer", "owner": None, "deadline": None},
            {"task": "Prototype invalidation on write", "owner": None, "deadline": None},
        ],
        "uncertain_text": [
            {"text_or_location": "bottom-right annotation", "reason": "handwriting is smudged"}
        ],
        "diagram": {
            "type": "flowchart",
            "mermaid": "flowchart TD\n    W[Write] --> I[Invalidate cache]\n    I --> R[Replica catches up]",
        },
        "diagram_explanation": "A write triggers invalidation, then replicas catch up.",
        "suggestions": [
            "Benchmark replica replication delay under peak load"
        ],
    }
)


def _png_bytes():
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (60, 40), "#336699").save(buf, format="PNG")
    return buf.getvalue()


class TestMetadataEndpoints:
    def test_health_endpoint(self):
        resp = client.get("/api/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["api_key"] == "present"
        assert data["model"] == ai_client.DEFAULT_MODEL
        assert data["endpoint"] == ai_client.endpoint_url()
        assert data["model_chain"] == [ai_client.DEFAULT_MODEL]

    def test_config_endpoint_reports_the_configured_model(self):
        resp = client.get("/api/config")
        assert resp.status_code == 200
        data = resp.json()
        assert data["api_key_present"] is True
        assert data["model"] == ai_client.DEFAULT_MODEL
        assert "modes" in data
        assert "auto" in data["modes"]

    def test_health_is_503_and_actionable_without_a_key(self, monkeypatch):
        """The first thing a new user hits is a missing key, not a 500."""
        for name in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "GEMINI_KEY"):
            monkeypatch.delenv(name, raising=False)

        resp = client.get("/api/health")
        assert resp.status_code == 503
        data = resp.json()
        assert data["status"] == "needs_key"
        assert data["api_key"] == "missing"
        # The model is still reported so the UI can show what it would use.
        assert data["model"] == ai_client.DEFAULT_MODEL

    def test_config_still_serves_metadata_without_a_key(self, monkeypatch):
        for name in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "GEMINI_KEY"):
            monkeypatch.delenv(name, raising=False)

        resp = client.get("/api/config")
        assert resp.status_code == 200
        assert resp.json()["api_key_present"] is False

    def test_a_missing_key_is_502_not_an_unhandled_500(self, monkeypatch):
        """ConfigurationError must stay inside the InferenceError handler."""
        for name in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "GEMINI_KEY"):
            monkeypatch.delenv(name, raising=False)
        monkeypatch.setattr(ai_client, "_client", None, raising=False)
        monkeypatch.setattr(ai_client, "_client_error", None, raising=False)

        files = {"image": ("board.png", _png_bytes(), "image/png")}
        resp = client.post("/api/analyze", files=files, data={"mode": "auto"})
        assert resp.status_code == 502
        detail = resp.json()["detail"]
        assert "No Gemini API key" in detail
        assert "aistudio.google.com/apikey" in detail

    def test_samples_endpoint(self):
        resp = client.get("/api/samples")
        assert resp.status_code == 200
        samples = resp.json()
        assert len(samples) >= 6
        modes = [s["mode"] for s in samples]
        for mode in ("notes", "flowchart", "architecture", "mindmap", "actions", "er"):
            assert mode in modes


class TestEndToEndPipeline:
    def test_analyze_end_to_end(self, client_env):
        genai = client_env(replies=[REALISTIC_RESPONSE])

        files = {"image": ("board.png", _png_bytes(), "image/png")}
        resp = client.post("/api/analyze", files=files, data={"mode": "auto"})
        assert resp.status_code == 200
        payload = resp.json()

        assert payload["title"] == "Cache Invalidation Strategy"
        assert payload["detected_mode"] == "flowchart"
        assert payload["summary"].startswith("TTL combined")
        assert len(payload["notes"]) == 2
        assert len(payload["takeaways"]) == 2
        assert len(payload["action_items"]) == 2
        assert payload["diagram"]["type"] == "flowchart"
        assert "flowchart TD" in payload["diagram"]["mermaid"]
        assert len(payload["uncertain_text"]) == 1
        assert len(payload["suggestions"]) == 1
        assert payload["source"]["filename"] == "board.png"
        assert payload["telemetry"]["model"] == ai_client.DEFAULT_MODEL
        assert payload["telemetry"]["endpoint"] == ai_client.endpoint_url()
        assert payload["telemetry"]["elapsed_ms"] >= 0
        assert len(genai.calls) == 1

    def test_inference_error_becomes_502(self, client_env):
        client_env(error=FakeAPIError("Service unavailable", 503))

        files = {"image": ("board.png", _png_bytes(), "image/png")}
        resp = client.post("/api/analyze", files=files, data={"mode": "auto"})
        assert resp.status_code == 502
        assert "Service unavailable" in resp.json()["detail"]

    def test_bad_upload_is_rejected_before_inference(self, client_env):
        genai = client_env(replies=[REALISTIC_RESPONSE])

        files = {"image": ("board.png", b"not an image at all", "image/png")}
        resp = client.post("/api/analyze", files=files, data={"mode": "auto"})
        assert resp.status_code == 400
        assert genai.calls == []


class TestDiagramValidationIsServerSide:
    def test_invalid_mermaid_is_omitted_but_notes_survive(self, client_env):
        payload = {
            "title": "Bad Diagram",
            "notes": [{"heading": "H", "content": ["C"]}],
            "diagram": {"type": "flowchart", "mermaid": "flowchart TD\n    A[Unclosed --> B"},
        }
        client_env(replies=[json.dumps(payload), "flowchart TD\n    still ] broken"])

        files = {"image": ("board.png", _png_bytes(), "image/png")}
        resp = client.post("/api/analyze", files=files, data={"mode": "flowchart"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["diagram"] is None
        assert len(data["notes"]) == 1
        assert data["telemetry"]["diagram_attempts"] == 2
        assert data["warnings"]