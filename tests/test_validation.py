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


class TestUploadValidation:
    def test_valid_png_accepted(self):
        img = validation.validate_upload(_png_bytes(), "image/png", "board.png")
        assert img.mime == "image/png"
        assert img.extension == "png"
        assert img.width == 40 and img.height == 20

    def test_valid_jpeg_accepted(self):
        img = validation.validate_upload(_jpeg_bytes(), "image/jpeg", "note.jpg")
        assert img.mime == "image/jpeg"

    def test_unsupported_extension_rejected(self):
        with pytest.raises(validation.ValidationError, match="not a supported file type"):
            validation.validate_upload(_png_bytes(), "image/png", "notes.txt")

    def test_missing_extension_rejected(self):
        with pytest.raises(validation.ValidationError):
            validation.validate_upload(_png_bytes(), "image/png", "noextension")

    def test_empty_file_rejected(self):
        with pytest.raises(validation.ValidationError, match="empty"):
            validation.validate_upload(b"", "image/png", "a.png")

    def test_oversized_file_rejected(self):
        big = b"\x89PNG\r\n\x1a\n" + b"0" * (validation.MAX_BYTES + 1)
        with pytest.raises(validation.ValidationError, match="limit is"):
            validation.validate_upload(big, "image/png", "big.png")

    def test_non_image_bytes_rejected(self):
        with pytest.raises(validation.ValidationError, match="does not appear to be a real image"):
            validation.validate_upload(b"just some text here", "image/png", "fake.png")

    def test_mime_mismatch_rejected(self):
        # Real JPEG bytes declared as PNG.
        with pytest.raises(validation.ValidationError, match="declared"):
            validation.validate_upload(_jpeg_bytes(), "image/png", "a.png")

    def test_extension_content_mismatch_rejected(self):
        with pytest.raises(validation.ValidationError, match="named"):
            validation.validate_upload(_png_bytes(), "image/png", "board.jpg")

    def test_gif_extension_rejected(self):
        from PIL import Image

        buf = io.BytesIO()
        Image.new("RGB", (10, 10)).save(buf, format="GIF")
        with pytest.raises(validation.ValidationError, match="not a supported file type"):
            validation.validate_upload(buf.getvalue(), "image/gif", "a.gif")

    def test_gif_content_disguised_as_png_rejected(self):
        """Extension says PNG, bytes say GIF -- the bytes must win."""
        from PIL import Image

        buf = io.BytesIO()
        Image.new("RGB", (10, 10)).save(buf, format="GIF")
        # Content sniffing runs first, so the GIF-specific message is correct here.
        with pytest.raises(validation.ValidationError, match="GIF is not supported"):
            validation.validate_upload(buf.getvalue(), "image/png", "a.png")

    def test_corrupt_png_rejected(self):
        corrupt = _png_bytes()[:60]  # valid header, truncated body
        with pytest.raises(validation.ValidationError):
            validation.validate_upload(corrupt, "image/png", "broken.png")

    def test_oversized_dimensions_rejected(self, monkeypatch):
        # Threshold lowered rather than allocating a genuinely huge image.
        monkeypatch.setattr(validation, "MAX_PIXELS", 100)
        with pytest.raises(validation.ValidationError, match="MP limit"):
            validation.validate_upload(_png_bytes(600, 600), "image/png", "huge.png")

    def test_dimensions_within_limit_accepted(self):
        img = validation.validate_upload(_png_bytes(600, 600), "image/png", "big.png")
        assert (img.width, img.height) == (600, 600)


class TestSniffMime:
    @pytest.mark.parametrize(
        "data,expected",
        [
            (b"\x89PNG\r\n\x1a\n", "image/png"),
            (b"\xff\xd8\xff\xe0", "image/jpeg"),
            (b"GIF89a...", "image/gif"),
            (b"RIFF\x00\x00\x00\x00WEBPVP8 ", "image/webp"),
            (b"<html>", None),
        ],
    )
    def test_sniff(self, data, expected):
        assert validation.sniff_mime(data) == expected


class TestExtractJsonObject:
    def test_bare_json(self):
        assert validation.extract_json_object('{"title":"x"}') == {"title": "x"}

    def test_fenced_json(self):
        text = 'Here you go:\n```json\n{"title":"x","notes":[]}\n```\nDone.'
        assert validation.extract_json_object(text)["title"] == "x"

    def test_json_with_preamble(self):
        text = 'Sure!\n{"title":"y","takeaways":["a"]}\nHope that helps.'
        assert validation.extract_json_object(text)["title"] == "y"

    def test_nested_braces(self):
        text = '{"title":"z","diagram":{"mermaid":"flowchart TD\\n A-->B"}}'
        assert validation.extract_json_object(text)["diagram"]["mermaid"].startswith("flowchart")

    def test_braces_inside_strings_do_not_break_scan(self):
        text = '{"title":"a {brace} b","notes":[]}'
        assert validation.extract_json_object(text)["title"] == "a {brace} b"

    def test_garbage_returns_none(self):
        assert validation.extract_json_object("no json at all") is None
        assert validation.extract_json_object("") is None


class TestNormalizeExtraction:
    def test_full_payload(self):
        payload = {
            "title": "Cache Invalidation",
            "notes": [
                {"heading": "Problem", "content": ["Stale reads"], "confidence": "high"},
                {"heading": "Fix", "content": ["TTL", "Invalidate on write"], "confidence": "low"},
            ],
            "takeaways": ["Choose one strategy"],
            "action_items": [{"task": "Write the invalidation hook", "owner": None, "deadline": None}],
            "uncertain_text": [{"text_or_location": "bottom right", "reason": "blurred"}],
            "diagram": None,
        }
        e = validation.normalize_extraction(payload)
        assert e.title == "Cache Invalidation"
        assert len(e.notes) == 2
        assert e.notes[1]["confidence"] == "low"
        assert e.takeaways == ["Choose one strategy"]
        assert e.action_items[0]["task"] == "Write the invalidation hook"
        assert e.uncertain_text[0]["reason"] == "blurred"
        assert e.diagram is None

    def test_empty_sections_preserved(self):
        e = validation.normalize_extraction(
            {"title": "T", "notes": [], "takeaways": [], "action_items": []}
        )
        assert e.notes == [] and e.takeaways == [] and e.action_items == []

    def test_fabricated_owner_collapsed_to_null(self):
        payload = {
            "title": "T",
            "action_items": [
                {"task": "Ship it", "owner": "unknown", "deadline": "null"},
                {"task": "Write docs", "owner": "Priya", "deadline": "Friday"},
            ],
        }
        e = validation.normalize_extraction(payload)
        assert e.action_items[0]["owner"] is None
        assert e.action_items[0]["deadline"] is None
        assert e.action_items[1]["owner"] == "Priya"
        assert e.action_items[1]["deadline"] == "Friday"

    def test_bad_confidence_defaults_medium(self):
        e = validation.normalize_extraction(
            {"title": "T", "notes": [{"heading": "H", "content": ["c"], "confidence": "VERY HIGH"}]}
        )
        assert e.notes[0]["confidence"] == "medium"

    def test_none_payload_does_not_raise(self):
        e = validation.normalize_extraction(None)
        assert e.notes == []
        assert e.warnings

    def test_notes_as_plain_strings(self):
        e = validation.normalize_extraction(
            {"title": "T", "notes": ["a standalone line"]}
        )
        assert e.notes[0]["content"] == ["a standalone line"]

    def test_unvalidated_diagram_is_dropped_with_warning(self):
        payload = {"title": "T", "notes": [{"heading": "H", "content": ["c"]}],
                   "diagram": {"type": "flowchart", "mermaid": "flowchart TD\nA-->"}}
        e = validation.normalize_extraction(payload, diagram_validated=None)
        assert e.diagram is None
        assert any("validation" in w.lower() for w in e.warnings)
        assert len(e.notes) == 1  # notes survive a bad diagram

    def test_validated_diagram_is_kept(self):
        payload = {"title": "T", "diagram": {"type": "flowchart", "mermaid": "flowchart TD\nA-->B"}}
        e = validation.normalize_extraction(
            payload, diagram_validated={"type": "flowchart", "mermaid": "flowchart TD\nA-->B\n"}
        )
        assert e.diagram["mermaid"].startswith("flowchart TD")
