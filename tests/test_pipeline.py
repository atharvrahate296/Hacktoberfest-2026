"""End-to-end pipeline test with mocked Hugging Face inference.

Proves the whole path -- validate -> infer -> normalize -> JSON payload the UI
consumes -- without a real network call.
"""

import io
import json

import pytest
import requests
from fastapi.testclient import TestClient

from app import main

client = TestClient(main.app)

REALISTIC_RESPONSE = json.dumps(
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


@pytest.fixture(autouse=True)
def env(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_test_token_12345")
    monkeypatch.setenv("HF_MODEL_ID", "google/gemma-3-27b-it")
    monkeypatch.delenv("HF_ENDPOINT_URL", raising=False)


def _png_bytes():
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (60, 40), "#336699").save(buf, format="PNG")
    return buf.getvalue()


class TestEndToEndPipeline:
    def test_health_endpoint(self):
        resp = client.get("/api/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["api_key"] == "present"
        assert "gemma" in data["model"].lower()

    def test_config_endpoint(self):
        resp = client.get("/api/config")
        assert resp.status_code == 200
        data = resp.json()
        assert data["api_key_present"] is True
        assert "modes" in data
        assert "auto" in data["modes"]

    def test_samples_endpoint(self):
        resp = client.get("/api/samples")
        assert resp.status_code == 200
        samples = resp.json()
        assert len(samples) >= 6
        modes = [s["mode"] for s in samples]
        assert "notes" in modes
        assert "flowchart" in modes
        assert "architecture" in modes
        assert "mindmap" in modes
        assert "actions" in modes
        assert "er" in modes

    def test_analyze_end_to_end(self, monkeypatch):
        def fake_post(url, headers=None, data=None, timeout=None):
            r = requests.Response()
            r.status_code = 200
            r._content = json.dumps({"choices": [{"message": {"content": REALISTIC_RESPONSE}}]}).encode()
            return r

        monkeypatch.setattr(requests, "post", fake_post)

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
        assert payload["telemetry"]["model"] == "google/gemma-3-27b-it"
