"""Local Hugging Face inference-client tests.

No weights are downloaded: the model/processor pair is scripted (see conftest)
while prompt assembly, decoding, JSON repair, Mermaid validation, and telemetry
run for real.
"""

from __future__ import annotations

import json

import pytest

from app import ai_client

from conftest import reply_json


@pytest.fixture(autouse=True)
def env(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_test_token_12345")
    monkeypatch.setenv("HF_MODEL_ID", "HuggingFaceTB/SmolVLM-500M-Instruct")


class TestModelNaming:
    def test_default_model_is_a_known_model(self):
        assert ai_client.DEFAULT_MODEL in ai_client.KNOWN_MODELS

    def test_model_id_follows_env(self, monkeypatch):
        monkeypatch.setenv("HF_MODEL_ID", "Qwen/Qwen2-VL-2B-Instruct")
        assert ai_client.model_id() == "Qwen/Qwen2-VL-2B-Instruct"
        monkeypatch.delenv("HF_MODEL_ID")
        assert ai_client.model_id() == ai_client.DEFAULT_MODEL

    def test_endpoint_is_always_local(self):
        assert ai_client.endpoint_url() is None

    def test_telemetry_reports_configured_model(self, image, client_env):
        client_env(replies=[reply_json({"title": "T", "notes": []})])
        _, telemetry = ai_client.analyze(image)
        assert telemetry["model"] == "HuggingFaceTB/SmolVLM-500M-Instruct"
        assert telemetry["model_used"] == "HuggingFaceTB/SmolVLM-500M-Instruct"
        assert telemetry["endpoint"] == "local"
        assert telemetry["model_fallbacks"] == []

    def test_readme_and_env_example_match_default(self):
        import io
        import pathlib

        root = pathlib.Path(__file__).resolve().parents[1]
        readme = io.open(root / "readme.md", encoding="utf-8").read()
        env_example = io.open(root / ".env.example", encoding="utf-8").read()
        assert f"HF_MODEL_ID={ai_client.DEFAULT_MODEL}" in readme
        assert f"HF_MODEL_ID={ai_client.DEFAULT_MODEL}" in env_example


class TestReadiness:
    def test_key_present_with_token(self, monkeypatch):
        monkeypatch.setenv("HF_TOKEN", "hf_xyz")
        assert ai_client.key_present() is True

    def test_key_present_when_weights_cached(self, monkeypatch):
        monkeypatch.delenv("HF_TOKEN", raising=False)
        monkeypatch.setattr(ai_client, "weights_cached", lambda: True)
        assert ai_client.key_present() is True

    def test_not_ready_without_token_or_cache(self, monkeypatch):
        monkeypatch.delenv("HF_TOKEN", raising=False)
        monkeypatch.setattr(ai_client, "weights_cached", lambda: False)
        assert ai_client.key_present() is False


class TestPromptAssembly:
    def test_two_step_call_forwards_image(self, image, client_env):
        model, processor = client_env(
            replies=[reply_json({"title": "T", "notes": [{"heading": "H", "content": ["C"]}]})]
        )
        ai_client.analyze(image, mode="notes")

        assert processor.processor_calls, "processor(text=...) path was not used"
        kwargs = processor.processor_calls[0]
        assert "images" in kwargs and len(kwargs["images"]) == 1
        # The image placeholder must survive into the templated prompt text.
        assert "[image]" in kwargs["text"][0]

        user_msg = [
            m for m in processor.template_calls[0]["messages"] if m["role"] == "user"
        ][0]
        assert any(p.get("type") == "image" for p in user_msg["content"])

    def test_fallback_path_still_forwards_image(self, image, client_env):
        model, processor = client_env(
            replies=[reply_json({"title": "T", "notes": []})], two_step_ok=False
        )
        ext, _ = ai_client.analyze(image)

        assert ext.title == "T"
        template_calls = processor.template_calls
        assert template_calls[-1]["tokenize"] is True
        assert template_calls[-1]["images"], "fallback path dropped the image"

    def test_mode_selects_system_prompt(self, image, client_env):
        model, processor = client_env(replies=[reply_json({"title": "ER", "notes": []})])
        ai_client.analyze(image, mode="er")
        system_msg = [
            m for m in processor.template_calls[0]["messages"] if m["role"] == "system"
        ][0]
        text = " ".join(p["text"] for p in system_msg["content"])
        assert "Database ER Diagram" in text

    def test_decode_strips_the_prompt(self):
        # A generated row carries the prompt tokens before the reply; decoding
        # must return only the reply.
        from conftest import FakeIds, FakeProcessor

        generated = FakeIds([["<pad>", "<pad>", "the reply"]])
        processor = FakeProcessor()
        assert ai_client._decode(processor, generated, 2) == "the reply"
        assert ai_client._decode(processor, generated, 0) == "<pad><pad>the reply"


class TestGenerationErrors:
    def test_empty_response_rejected(self, image, client_env):
        client_env(replies=["   "])
        with pytest.raises(ai_client.InferenceError, match="empty response"):
            ai_client.analyze(image)

    def test_oom_is_a_friendly_error(self, image, client_env):
        client_env(error=RuntimeError("CUDA out of memory. Tried to allocate 2 GiB"))
        with pytest.raises(ai_client.InferenceError, match="out of memory during generation"):
            ai_client.analyze(image)

    def test_generation_failure_wrapped(self, image, client_env):
        client_env(error=RuntimeError("shape mismatch in forward pass"))
        with pytest.raises(ai_client.InferenceError, match="Generation failed"):
            ai_client.analyze(image)

    def test_encode_failure_names_both_paths(self, image, client_env):
        client_env(replies=["x"], two_step_ok=False, template_works=False)
        with pytest.raises(ai_client.InferenceError, match="could not encode the prompt"):
            ai_client.analyze(image)

    def test_unparseable_output_degrades_gracefully(self, image, client_env):
        # No JSON on either attempt: notes survive, a warning explains why.
        client_env(replies=["not json at all", "still not json"])
        ext, telemetry = ai_client.analyze(image)
        assert ext.notes == []
        assert any("could not parse" in w for w in ext.warnings)
        assert telemetry["raw_preview"]


class TestEmptyContentRetry:
    """A reply that parses but carries nothing means the model never read
    the image; one image-aware retry is worth its latency."""

    EMPTY = {"title": "Untitled Extraction", "notes": [], "takeaways": [], "diagram": None}
    REAL = {
        "title": "OS Process State Model",
        "notes": [{"heading": "States", "content": ["New", "Ready"], "confidence": "high"}],
        "diagram": {"type": "flowchart", "mermaid": "flowchart TD\n    N[New] --> R[Ready]"},
    }

    def test_retry_with_image_recovers_content(self, image, client_env):
        model, _ = client_env(replies=[json.dumps(self.EMPTY), json.dumps(self.REAL)])
        ext, telemetry = ai_client.analyze(image, mode="flowchart")

        assert telemetry["content_attempts"] == 1
        assert telemetry["recovered"] is True
        assert ext.title == "OS Process State Model"
        assert len(ext.notes) == 1
        assert ext.diagram is not None

    def test_retry_still_image_aware(self, image, client_env):
        model, processor = client_env(
            replies=[json.dumps(self.EMPTY), json.dumps(self.REAL)]
        )
        ai_client.analyze(image, mode="flowchart")
        # Two image generations: the image must be attached to the retry too.
        assert sum(1 for c in processor.processor_calls if c.get("images")) == 2
        assert "previous answer contained no usable content" in processor.processor_calls[-1]["text"][0]

    def test_degrades_with_a_raw_preview_when_retry_fails(self, image, client_env):
        client_env(replies=[json.dumps(self.EMPTY), json.dumps(self.EMPTY)])
        ext, telemetry = ai_client.analyze(image)

        assert telemetry["degraded"] is True
        assert "recovered" not in telemetry
        assert telemetry["raw_preview"]
        assert ext.notes == []

    def test_no_retry_when_content_was_extracted(self, image, client_env):
        model, _ = client_env(replies=[json.dumps(self.REAL)])
        ext, telemetry = ai_client.analyze(image)
        assert "content_attempts" not in telemetry
        assert len(model.generate_calls) == 1

    def test_retry_failure_is_not_fatal(self, image, client_env):
        # The retry raises; the empty first result is still returned, annotated.
        model, _ = client_env(replies=[json.dumps(self.EMPTY)], error=RuntimeError("CUDA out of memory"))
        model.error_on = 2
        ext, telemetry = ai_client.analyze(image)
        assert telemetry["content_retry_error"]
        assert telemetry["degraded"] is True
        assert ext.notes == []


class TestCorrectiveRetries:
    def test_json_repair_on_malformed_initial_response(self, image, client_env):
        model, _ = client_env(
            replies=[
                "Here is your output but it is not valid JSON {{",
                reply_json({"title": "Recovered", "notes": [{"heading": "H", "content": ["C"]}]}),
            ]
        )
        ext, telemetry = ai_client.analyze(image)
        assert telemetry["attempts"] == 2
        assert ext.title == "Recovered"
        assert len(model.generate_calls) == 2

    def test_diagram_repair_on_invalid_mermaid(self, image, client_env):
        bad = {
            "title": "Bad Diagram",
            "notes": [{"heading": "H", "content": ["C"]}],
            "diagram": {"type": "flowchart", "mermaid": "flowchart TD\n    A[Unclosed --> B"},
        }
        model, _ = client_env(
            replies=[json.dumps(bad), "flowchart TD\n    A[Fixed bracket] --> B[Next]"]
        )
        ext, telemetry = ai_client.analyze(image)
        assert telemetry["diagram_attempts"] == 2
        assert ext.diagram is not None
        assert "Fixed bracket" in ext.diagram["mermaid"]
        assert len(model.generate_calls) == 2

    def test_valid_diagram_needs_no_retry(self, image, client_env):
        good = {
            "title": "Good Diagram",
            "notes": [{"heading": "H", "content": ["C"]}],
            "diagram": {"type": "flowchart", "mermaid": "flowchart TD\n    A[Write] --> B[Read]"},
        }
        model, _ = client_env(replies=[json.dumps(good)])
        ext, telemetry = ai_client.analyze(image)
        assert telemetry["diagram_attempts"] == 1
        assert len(model.generate_calls) == 1
        assert ext.diagram["mermaid"].startswith("flowchart TD")

    def test_unrepairable_diagram_does_not_lose_notes(self, image, client_env):
        bad = {
            "title": "Broken Diagram",
            "notes": [{"heading": "H", "content": ["C"]}],
            "diagram": {"type": "flowchart", "mermaid": "flowchart TD\n    A[Unclosed --> B"},
        }
        client_env(replies=[json.dumps(bad), "flowchart TD\n    still ] broken"])
        ext, telemetry = ai_client.analyze(image)
        assert ext.diagram is None
        assert telemetry["diagram_attempts"] == 2
        assert telemetry["diagram_error"]
        assert len(ext.notes) == 1
        assert any("Mermaid validation" in w for w in ext.warnings)