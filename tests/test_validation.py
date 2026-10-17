"""Validation tests. No network, no API key required."""

import io

import pytest

from app import validation


def _png_bytes(width=40, height=20):
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (width, height), "#123456").save(buf, format="PNG")
    return buf.getvalue()


def _jpeg_bytes():
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (30, 30), "#abcdef").save(buf, format="JPEG")
    return buf.getvalue()


def _webp_bytes():
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (30, 30), "#abcdef").save(buf, format="WEBP")
    return buf.getvalue()


class TestUploadValidation:
    def test_valid_png_accepted(self):
        img = validation.validate_upload(_png_bytes(), "image/png", "board.png")
        assert img.mime == "image/png"
        assert img.extension == "png"
        assert img.width == 40 and img.height == 20

    def test_valid_jpeg_accepted(self):
        img = validation.validate_upload(_jpeg_bytes(), "image/jpeg", "note.jpg")
        assert img.mime == "image/jpeg"

    def test_valid_webp_accepted(self):
        img = validation.validate_upload(_webp_bytes(), "image/webp", "note.webp")
        assert img.mime == "image/webp"

    def test_unsupported_extension_rejected(self):
        with pytest.raises(validation.ValidationError, match="not a supported file type"):
            validation.validate_upload(_png_bytes(), "image/png", "notes.txt")

    def test_missing_extension_rejected(self):
        with pytest.raises(validation.ValidationError):
            validation.validate_upload(_png_bytes(), "image/png", "noextension")

    def test_empty_file_rejected(self):
        with pytest.raises(validation.ValidationError, match="empty"):
            validation.validate_upload(b"", "image/png", "a.png")

    def test_oversized_file_rejected(self, monkeypatch):
        monkeypatch.setenv("MAX_UPLOAD_MB", "10")
        big = b"\x89PNG\r\n\x1a\n" + b"0" * (10 * 1024 * 1024 + 10)
        with pytest.raises(validation.ValidationError, match="limit is"):
            validation.validate_upload(big, "image/png", "big.png")

    def test_non_image_bytes_rejected(self):
        with pytest.raises(validation.ValidationError, match="does not appear to be a real image"):
            validation.validate_upload(b"just some text here", "image/png", "fake.png")

    def test_mime_mismatch_rejected(self):
        with pytest.raises(validation.ValidationError, match="declared"):
            validation.validate_upload(_jpeg_bytes(), "image/png", "mismatch.jpg")

    def test_extension_mismatch_rejected(self):
        with pytest.raises(validation.ValidationError, match="Rename it to match"):
            validation.validate_upload(_jpeg_bytes(), "image/jpeg", "mismatch.png")


class TestNormalizeExtraction:
    def test_none_payload_returns_safe_extraction(self):
        ext = validation.normalize_extraction(None, requested_mode="flowchart")
        assert ext.title == "Extraction unavailable"
        assert ext.mode == "flowchart"
        assert len(ext.warnings) > 0

    def test_complete_payload_normalized(self):
        payload = {
            "title": "Architecture Overview",
            "detected_mode": "architecture",
            "summary": "This document describes the high-level cloud architecture.",
            "notes": [
                {
                    "heading": "API Tier",
                    "content": ["Handles TLS termination", "Applies rate limiting"],
                    "confidence": "high",
                }
            ],
            "takeaways": ["Microservices communicate via gRPC internally"],
            "action_items": [
                {"task": "Configure Redis cluster", "owner": "Marcus", "deadline": "Friday"},
                {"task": "Review PR", "owner": "null", "deadline": "unknown"},
            ],
            "uncertain_text": [
                {"text_or_location": "margin note", "reason": "smudged ink"}
            ],
            "sql_schema": "CREATE TABLE users (id INT PRIMARY KEY);",
            "suggestions": ["Consider adding circuit breakers"],
        }
        ext = validation.normalize_extraction(payload, requested_mode="auto")
        assert ext.title == "Architecture Overview"
        assert ext.detected_mode == "architecture"
        assert ext.summary == "This document describes the high-level cloud architecture."
        assert len(ext.notes) == 1
        assert ext.notes[0]["confidence"] == "high"
        assert len(ext.takeaways) == 1
        assert len(ext.action_items) == 2
        # Verify null strings collapse to None
        assert ext.action_items[0]["owner"] == "Marcus"
        assert ext.action_items[1]["owner"] is None
        assert ext.action_items[1]["deadline"] is None
        assert ext.sql_schema.startswith("CREATE TABLE")
        assert len(ext.suggestions) == 1
        assert len(ext.uncertain_text) == 1

    def test_extract_json_object_from_fenced_and_prose(self):
        text = """Here is the result:
        ```json
        {
          "title": "Clean Result",
          "notes": [{"heading": "H", "content": ["Line 1"]}]
        }
        ```
        Hope this helps!"""
        obj = validation.extract_json_object(text)
        assert obj is not None
        assert obj["title"] == "Clean Result"
