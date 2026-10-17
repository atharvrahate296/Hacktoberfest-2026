"""Hugging Face Gemma inference-layer tests using mocked HTTP. No API key or network required."""

import io
import json

import pytest
import requests

from app import ai_client, validation


@pytest.fixture
def image():
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (30, 20), "#ffffff").save(buf, format="PNG")
    return validation.validate_upload(buf.getvalue(), "image/png", "board.png")


def _reply(text):
    return {"choices": [{"message": {"content": text}}]}


@pytest.fixture(autouse=True)
def env(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_test_token_12345")
    monkeypatch.setenv("HF_MODEL_ID", "google/gemma-3-27b-it")
    monkeypatch.setenv("HF_MODEL_FALLBACKS", "")
    monkeypatch.delenv("HF_ENDPOINT_URL", raising=False)


def _patch(monkeypatch, side_effect):
    calls = []

    def fake_post(url, headers=None, data=None, timeout=None):
        payload = json.loads(data) if data else {}
        calls.append({"url": url, "headers": headers, "data": payload})
        result = side_effect[len(calls) - 1] if isinstance(side_effect, list) else side_effect
        if isinstance(result, Exception):
            raise result
        status, body = result
        r = requests.Response()
        r.status_code = status
        r._content = json.dumps(body).encode()
        return r

    monkeypatch.setattr(requests, "post", fake_post)
    return calls


class TestMissingKey:
    def test_missing_key_raises_friendly_error(self, image, monkeypatch):
        monkeypatch.delenv("HF_TOKEN", raising=False)
        with pytest.raises(ai_client.InferenceError, match="HF_TOKEN is not set"):
            ai_client.analyze(image)

    def test_key_present_reflects_env(self, monkeypatch):
        monkeypatch.delenv("HF_TOKEN", raising=False)
        assert ai_client.key_present() is False
        monkeypatch.setenv("HF_TOKEN", "hf_xyz")
        assert ai_client.key_present() is True


class TestProviderErrors:
    @pytest.mark.parametrize(
        "status,match",
        [
            (401, "rejected the access token"),
            (402, "insufficient credits"),
            (403, "rejected the access token"),
            (404, "was not found on Hugging Face"),
            (500, "returned status 500"),
        ],
    )
    def test_http_status_translated(self, image, monkeypatch, status, match):
        _patch(monkeypatch, (status, {"error": {"message": "x"}}))
        with pytest.raises(ai_client.InferenceError, match=match):
            ai_client.analyze(image)

    def test_503_model_loading_translated(self, image, monkeypatch):
        _patch(monkeypatch, (503, {"error": "Model is loading", "estimated_time": 20.0}))
        with pytest.raises(ai_client.InferenceError, match="currently loading"):
            ai_client.analyze(image)

    def test_all_models_rate_limited_is_friendly(self, image, monkeypatch):
        _patch(monkeypatch, (429, {"error": {"message": "saturated"}}))
        with pytest.raises(ai_client.InferenceError, match="rate limited"):
            ai_client.analyze(image)

    def test_timeout_translated(self, image, monkeypatch):
        _patch(monkeypatch, requests.Timeout("slow"))
        with pytest.raises(ai_client.InferenceError, match="timed out after"):
            ai_client.analyze(image)

    def test_network_error_translated(self, image, monkeypatch):
        _patch(monkeypatch, requests.ConnectionError("no route"))
        with pytest.raises(ai_client.InferenceError, match="Could not reach"):
            ai_client.analyze(image)

    def test_empty_response_rejected(self, image, monkeypatch):
        _patch(monkeypatch, (200, {"choices": [{"message": {"content": "   "}}]}))
        with pytest.raises(ai_client.InferenceError, match="empty response"):
            ai_client.analyze(image)

    def test_bad_shape_rejected(self, image, monkeypatch):
        _patch(monkeypatch, (200, {"unexpected": "payload"}))
        with pytest.raises(ai_client.InferenceError, match="unexpected response shape"):
            ai_client.analyze(image)


class TestRoutingAndConfig:
    def test_default_url_uses_router(self, image, monkeypatch):
        calls = _patch(
            monkeypatch,
            (200, _reply(json.dumps({"title": "Test", "notes": [{"heading": "H", "content": ["C"]}]})))
        )
        ai_client.analyze(image)
        assert len(calls) == 1
        assert "router.huggingface.co/v1/chat/completions" in calls[0]["url"]
        assert calls[0]["data"]["model"] == "google/gemma-3-27b-it"

    def test_custom_endpoint_url_used(self, image, monkeypatch):
        monkeypatch.setenv("HF_ENDPOINT_URL", "https://custom-gemma.endpoints.huggingface.cloud")
        calls = _patch(
            monkeypatch,
            (200, _reply(json.dumps({"title": "Custom", "notes": [{"heading": "H", "content": ["C"]}]})))
        )
        ai_client.analyze(image)
        assert len(calls) == 1
        assert "custom-gemma.endpoints.huggingface.cloud/v1/chat/completions" in calls[0]["url"]

    def test_mode_passed_to_prompts(self, image, monkeypatch):
        calls = _patch(
            monkeypatch,
            (200, _reply(json.dumps({"title": "ER", "detected_mode": "er", "notes": []})))
        )
        ai_client.analyze(image, mode="er")
        assert len(calls) == 1
        messages = calls[0]["data"]["messages"]
        system_msg = next(m["content"] for m in messages if m["role"] == "system")
        assert "Database ER Diagram" in system_msg


class TestCorrectiveRetries:
    def test_json_repair_on_malformed_initial_response(self, image, monkeypatch):
        calls = _patch(
            monkeypatch,
            [
                (200, _reply("Here is your output but it is definitely not valid JSON {{")),
                (200, _reply(json.dumps({"title": "Recovered", "notes": [{"heading": "H", "content": ["C"]}]}))),
            ],
        )
        ext, tele = ai_client.analyze(image)
        assert tele["attempts"] == 2
        assert ext.title == "Recovered"
        assert len(calls) == 2

    def test_diagram_repair_on_invalid_mermaid(self, image, monkeypatch):
        bad_diagram = {
            "title": "Bad Diagram Test",
            "notes": [{"heading": "H", "content": ["C"]}],
            "diagram": {"type": "flowchart", "mermaid": "flowchart TD\n    A[Unclosed bracket --> B"},
        }
        good_diagram = "flowchart TD\n    A[Fixed bracket] --> B[Next]"

        calls = _patch(
            monkeypatch,
            [
                (200, _reply(json.dumps(bad_diagram))),
                (200, _reply(good_diagram)),
            ],
        )
        ext, tele = ai_client.analyze(image)
        assert tele["diagram_attempts"] == 2
        assert ext.diagram is not None
        assert "Fixed bracket" in ext.diagram["mermaid"]
