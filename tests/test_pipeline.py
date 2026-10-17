"""End-to-end pipeline test with a mocked provider.

Proves the whole path -- validate -> infer -> normalize -> JSON payload the UI
consumes -- without a network call. This is a contract test for the frontend:
if the response shape changes, this fails.

This is NOT live provider verification. See README for that distinction.
"""

import io
import json

import pytest
import requests

from app import ai_client, validation

REALISTIC_RESPONSE = json.dumps(
    {
        "title": "Cache Invalidation Strategy",
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
    }
)


@pytest.fixture(autouse=True)
def env(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-test")
    monkeypatch.setenv("OPENROUTER_MODEL", "google/gemma-4-26b-a4b-it:free")


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(
        requests,
        "post",
        lambda *a, **k: _resp(200, {"choices": [{"message": {"content": REALISTIC_RESPONSE}}]}),
    )
    from fastapi.testclient import TestClient

    from app.main import app

    return TestClient(app)


def _resp(status, body):
    r = requests.Response()
    r.status_code = status
    r._content = json.dumps(body).encode()
    return r


@pytest.fixture
def png():
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (200, 120), "#ffffff").save(buf, format="PNG")
    return buf.getvalue()


class TestFullPipeline:
    def test_response_has_every_field_the_ui_reads(self, client, png):
        r = client.post("/api/analyze", files={"image": ("board.png", png, "image/png")})
        assert r.status_code == 200
        d = r.json()

        for key in (
            "title",
            "notes",
            "takeaways",
            "action_items",
            "uncertain_text",
            "diagram",
            "diagram_explanation",
            "warnings",
            "source",
            "telemetry",
        ):
            assert key in d, f"missing {key}"

        assert d["title"] == "Cache Invalidation Strategy"
        assert len(d["notes"]) == 2
        assert d["notes"][0]["confidence"] == "high"
        assert len(d["takeaways"]) == 2
        assert len(d["action_items"]) == 2
        assert d["action_items"][0]["owner"] is None
        assert len(d["uncertain_text"]) == 1
        assert d["source"]["width"] == 200
        assert d["telemetry"]["model"] == "google/gemma-4-26b-a4b-it:free"
        assert d["telemetry"]["elapsed_ms"] >= 0

    def test_diagram_survives_the_whole_pipeline(self, client, png):
        d = client.post(
            "/api/analyze", files={"image": ("board.png", png, "image/png")}
        ).json()
        assert d["diagram"]["mermaid"].startswith("flowchart TD")
        assert d["diagram_explanation"]

    def test_no_fabricated_content_when_model_returns_nothing(self, client, png, monkeypatch):
        """An empty-but-valid response must produce empty sections, not filler."""
        monkeypatch.setattr(
            requests,
            "post",
            lambda *a, **k: _resp(
                200,
                {
                    "choices": [
                        {
                            "message": {
                                "content": json.dumps(
                                    {
                                        "title": "Blank",
                                        "notes": [],
                                        "takeaways": [],
                                        "action_items": [],
                                        "diagram": None,
                                    }
                                )
                            }
                        }
                    ]
                },
            ),
        )
        d = client.post("/api/analyze", files={"image": ("b.png", png, "image/png")}).json()
        assert d["notes"] == []
        assert d["takeaways"] == []
        assert d["action_items"] == []
        assert d["diagram"] is None
        assert any("No readable text" in w for w in d["warnings"])

    def test_bad_diagram_loses_only_the_diagram(self, client, png, monkeypatch):
        monkeypatch.setattr(
            requests,
            "post",
            lambda *a, **k: _resp(
                200,
                {
                    "choices": [
                        {
                            "message": {
                                "content": json.dumps(
                                    {
                                        "title": "Keeps notes",
                                        "notes": [
                                            {"heading": "H", "content": ["critical note"]}
                                        ],
                                        "takeaways": ["critical takeaway"],
                                        "diagram": {"mermaid": "flowchart TD\n A[x --> B[y]"},
                                    }
                                )
                            }
                        }
                    ]
                },
            ),
        )
        d = client.post("/api/analyze", files={"image": ("b.png", png, "image/png")}).json()
        assert d["diagram"] is None
        assert d["notes"][0]["content"] == ["critical note"]
        assert d["takeaways"] == ["critical takeaway"]
        assert d["warnings"]


class TestConfigEndpoint:
    def test_config_reports_missing_key_honestly(self, monkeypatch):
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        from fastapi.testclient import TestClient

        from app.main import app

        d = TestClient(app).get("/api/config").json()
        assert d["api_key_present"] is False
        assert d["max_mb"] == 8
