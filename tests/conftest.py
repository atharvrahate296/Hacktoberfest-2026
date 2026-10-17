"""Shared test doubles for the local Hugging Face inference client.

The real model is never downloaded in tests. What is faked is the tensor layer
only: a model whose ``generate`` returns a scripted reply, and a processor whose
chat-template/tokenizer/decode surface matches the API ``app/ai_client.py`` uses.
Everything above that -- prompt assembly, JSON repair, Mermaid validation,
normalization, HTTP routing -- runs for real.
"""

from __future__ import annotations

import io
import json

import pytest

from app import ai_client, validation

PROMPT_LEN = 2


class FakeIds:
    """Minimal stand-in for a torch tensor of token ids."""

    def __init__(self, rows: list[list]):
        self._rows = [list(r) for r in rows]

    @property
    def shape(self):
        return (len(self._rows), len(self._rows[0]) if self._rows else 0)

    def __len__(self):
        return len(self._rows)

    def __getitem__(self, key):
        if isinstance(key, tuple):
            _, cols = key
            return FakeIds([row[cols] for row in self._rows])
        return FakeIds([row[key] for row in self._rows])

    def to(self, device):
        return self

    def rows(self):
        return self._rows


class FakeModel:
    def __init__(self, replies: list[str], error: Exception | None = None):
        self.replies = list(replies)
        self.error = error
        # 1-based index of the generate() call that should raise; None = always.
        self.error_on: int | None = None
        self.generate_calls: list[dict] = []
        self.eval_called = False

    def generate(self, **inputs):
        self.generate_calls.append(inputs)
        if self.error is not None and (
            self.error_on is None or self.error_on == len(self.generate_calls)
        ):
            raise self.error
        reply = self.replies.pop(0) if self.replies else ""
        return FakeIds([["<pad>", "<pad>", reply]])

    def eval(self):
        self.eval_called = True
        return self

    def to(self, device):
        return self

    def parameters(self):
        yield type("Param", (), {"device": "cpu"})()


class FakeProcessor:
    """Mimics the processor call shapes ``_build_inputs`` tries in order."""

    def __init__(self, two_step_ok: bool = True, template_works: bool = True):
        self.two_step_ok = two_step_ok
        self.template_works = template_works
        self.template_calls: list[dict] = []
        self.processor_calls: list[dict] = []

    def apply_chat_template(
        self, messages, tokenize=False, add_generation_prompt=True, **kwargs
    ):
        self.template_calls.append(
            {"messages": messages, "tokenize": tokenize, "images": kwargs.get("images")}
        )
        if not self.template_works:
            raise ValueError("chat template unsupported")
        if tokenize:
            if not kwargs.get("images") and any(
                part.get("type") == "image"
                for msg in messages
                for part in msg.get("content", [])
            ):
                # A real processor cannot tokenize an image-less prompt that
                # still declares one; this forces the caller's fallback path.
                raise ValueError("image content present but no images passed")
            return {"input_ids": FakeIds([[0] * PROMPT_LEN]), "pixel_values": object()}
        parts = []
        for msg in messages:
            for part in msg.get("content", []):
                if part.get("type") == "text":
                    parts.append(f"{msg['role']}:{part['text']}")
                else:
                    parts.append(f"{msg['role']}:[image]")
        return "\n".join(parts)

    def __call__(self, **kwargs):
        self.processor_calls.append(kwargs)
        if not self.two_step_ok:
            raise ValueError("processor(text=..., images=...) unsupported")
        return {"input_ids": FakeIds([[0] * PROMPT_LEN]), "pixel_values": object()}

    def batch_decode(self, ids, skip_special_tokens=True):
        return ["".join(str(tok) for tok in row) for row in ids.rows()]


@pytest.fixture
def image():
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (30, 20), "#ffffff").save(buf, format="PNG")
    return validation.validate_upload(buf.getvalue(), "image/png", "board.png")


@pytest.fixture
def client_env(monkeypatch):
    """Install a scripted model/processor pair and reset loader globals.

    Returns a factory: ``client_env(replies=[...], error=..., two_step_ok=...)
    -> (model, processor)``.
    """

    def install(replies=None, error=None, two_step_ok=True, template_works=True):
        monkeypatch.setattr(ai_client, "_model", None, raising=False)
        monkeypatch.setattr(ai_client, "_processor", None, raising=False)
        monkeypatch.setattr(ai_client, "_load_error", None, raising=False)
        monkeypatch.setattr(ai_client, "_model_device", "cpu", raising=False)

        model = FakeModel(replies or [], error=error)
        processor = FakeProcessor(two_step_ok=two_step_ok, template_works=template_works)
        monkeypatch.setattr(
            ai_client, "_get_model_and_processor", lambda: (model, processor)
        )
        return model, processor

    return install


def reply_json(payload: dict) -> str:
    return json.dumps(payload)