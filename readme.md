# Scribble2Structure

**From messy visual thinking to structured, editable knowledge.**

Scribble2Structure turns handwritten notes, whiteboard photographs, brainstorming
sketches, technical system drawings and meeting notes into structured artifacts with
rendered Mermaid diagrams, verifiable action plans and multi-format exports.

Inference runs **entirely on your own machine** through Hugging Face Transformers. There is
no inference API, no per-request cost, and no image ever leaves the device.

---

## Choosing a model that fits your laptop

Extraction quality scales with the model, and the model you can run is bounded by the
memory your machine actually has. **This project is designed around that constraint:** the
same codebase runs a 1 GB model on an entry-level laptop and a 9 GB model on a workstation,
with no code changes — only the `HF_MODEL_ID` value in `.env`.

| `HF_MODEL_ID` | Download | Fits comfortably on | Quality |
| --- | --- | --- | --- |
| `HuggingFaceTB/SmolVLM-500M-Instruct` *(default)* | ~1 GB | 4 GB VRAM, or CPU-only | Runs anywhere; weakest extraction. Useful for wiring and layout checks. |
| `Qwen/Qwen2-VL-2B-Instruct` | ~4 GB | 8 GB VRAM, or 4 GB VRAM with CPU offload | **Recommended.** Best quality-to-footprint ratio; the practical default for real use. |
| `Qwen/Qwen2.5-VL-3B-Instruct` | ~6 GB | 8–12 GB VRAM | Better on dense handwriting and multi-box diagrams. |
| `google/gemma-3-4b-it` | ~9 GB | 12–16 GB VRAM | Highest accuracy; gated on Hugging Face, requires accepting the licence. |

### How memory is handled at runtime

`app/ai_client.py` sizes the model to the hardware instead of assuming one target:

- **CUDA available** → the model is placed on the GPU in `float16` for speed. If it does not
  fit in VRAM, it retries with `device_map="auto"` under a **3 GiB VRAM ceiling**, letting
  `accelerate` offload the overflow to system RAM instead of raising an out-of-memory error.
- **No CUDA** → `float32` on CPU. Slower, but no special hardware required.
- **Model choice is yours, at runtime** — the id is read from the environment on every
  request, so switching models needs only an edit to `.env` and a restart. Weights are cached
  in `~/.cache/huggingface` after the first run.

---

## Features

- **Local, model-flexible inference** — any of the vision-language models above, selected by
  your available memory; automatic CPU/GPU placement with graceful offload.
- **Reconstruction as the deliverable** — diagrams, notes and action plans are rebuilt as
  editable artifacts rather than described in prose.
- **8 transformation modes**:
  1. **Auto Detect** — picks the best mode from the image's layout and content.
  2. **Handwritten Notes → Structured Notes** — headings, bullets, concepts, formulas.
  3. **Diagram → Flowchart** — nodes, labels, arrows and decision paths as Mermaid.
  4. **Brainstorming → Mind Map** — themes, categories and radiating ideas as Mermaid.
  5. **Technical Diagram → Architecture** — tiers, services, protocols, storage, data flow.
  6. **Meeting Notes → Action Plan** — decisions, tasks, owners and deadlines, with no invented facts.
  7. **Scientific Concept Map** — equations, symbols, units and relationships preserved.
  8. **Database ER Diagram** — entities, attributes, PK/FK, cardinalities, plus SQL `CREATE TABLE`.
- **Rendered Mermaid with a live editor** — `flowchart`, `graph`, `mindmap`,
  `sequenceDiagram` and `erDiagram`; edit the source and re-render in place; export `.mmd` or `.svg`.
- **Side-by-side comparison** — the uploaded image next to the reconstructed diagram.
- **Editable output with state management** — live editing of titles, summaries, notes,
  action items and diagrams, with reset and unsaved-change protection.
- **Multi-format exports** — Markdown, plain text, JSON, Mermaid and SVG.
- **Sample gallery** — six generated inputs, one per diagram-oriented mode, for zero-effort evaluation.

---

## Implementation

### Pipeline

```
upload → validate → prompt → generate → repair → normalize → render
         │                       │         │          │
    magic bytes,           image +     JSON retry,   schema-echo
    size, EXIF,            mode prompt Mermaid retry rejection
    pixel ceiling
```

1. **Validation** (`app/validation.py`) — the file's leading bytes are checked against real
   image signatures, not the browser-supplied content type; extension, declared MIME and
   actual format must agree. Size, megapixel ceiling and EXIF orientation are handled before
   any inference cost is incurred.
2. **Prompting** (`app/prompts.py`) — a mode-specific system prompt plus the full JSON schema.
   Instructions state that the image is data, never instructions.
3. **Inference** (`app/ai_client.py`) — the model is loaded once into a thread-safe singleton.
   Placement is attempted GPU-first, then via `device_map="auto"` with memory ceilings; the
   architecture declared by the checkpoint decides which model class is used.
4. **Corrective retries** — one retry when the reply is not parseable JSON, and one retry when
   the generated Mermaid fails validation. Both are text-only follow-ups.
5. **Normalization** (`app/validation.py`) — arbitrary model JSON is coerced into a
   renderable `Extraction`. Never raises: missing fields degrade to empty, so notes survive
   even when the diagram does not. Values that merely echo the prompt's own template are
   discarded rather than displayed as extracted content.
6. **Diagram validation** (`app/diagram.py`) — fences and prose are stripped, script-like
   markup is rejected, then supported type, size, bracket/quote balance and node-label
   bracketing are checked before Mermaid ever reaches the browser.
7. **Rendering** (`static/index.html`) — Mermaid.js draws the diagram; every model string is
   HTML-escaped before insertion, and each render uses a unique id so concurrent draws cannot
   collide.

### Fail-safe by design

A hostile, broken or simply weak model response degrades instead of breaking the product:
notes and takeaways survive a diagram failure, a rejected diagram falls back to showing its
source in a code block, and a load failure is translated into an actionable message rather
than a stack trace.

---

## Setup

### 1. Prerequisites

- Python 3.10+
- Disk space for the chosen model (1–9 GB, see the table above)
- A Hugging Face access token from [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens),
  needed only to download weights the first time. Gated models additionally require accepting
  their licence.

### 2. Install

```bash
git clone https://github.com/rohan1252030019-netizen/scribble2structure.git
cd scribble2structure

python -m venv .venv
# Windows:      .\.venv\Scripts\activate
# Linux/macOS:  source .venv/bin/activate

pip install -r requirements.txt
```

### 3. Configure

```bash
cp .env.example .env      # Windows: copy .env.example .env
```

```ini
# Hugging Face token - only needed for the first weight download
HF_TOKEN=hf_your_actual_token_here

# Pick the model that matches your available memory (default shown)
HF_MODEL_ID=HuggingFaceTB/SmolVLM-500M-Instruct

# Optional image upload limit in MB (default: 10)
MAX_UPLOAD_MB=10
```

### 4. Run

```bash
uvicorn app.main:app --host 127.0.0.1 --port 8090 --reload
```

Open **http://127.0.0.1:8090**.

---

## API

| Method | Endpoint | Purpose |
| --- | --- | --- |
| `GET` | `/api/health` | Readiness and the active model. `200` when runnable, `503` when weights are missing. |
| `GET` | `/api/config` | Model id, upload limit, supported formats, available modes. |
| `GET` | `/api/samples` | Sample gallery entries. |
| `POST` | `/api/analyze` | `multipart/form-data` with `image` and `mode`. `400` on a bad upload, `502` on an inference failure. |

---

## Samples

Six generated inputs live in `samples/`, covering notes, flowcharts, architecture, mind maps,
meeting notes and ER diagrams. Regenerate them with:

```bash
python scripts/create_samples.py
```

---

## Tests

```bash
pytest
```

The suite runs with no model download and no network. Only the tensor layer is faked
(`tests/conftest.py`); prompt assembly, decoding, corrective retries, Mermaid validation,
normalization and HTTP routing all execute for real.

- `tests/test_ai_client.py` — model selection, prompt assembly, decode, error translation, retries.
- `tests/test_diagram.py` — Mermaid sanitization and validation.
- `tests/test_validation.py` — upload integrity and response normalization, including template-echo rejection.
- `tests/test_frontend.py` — frontend contracts: render targets exist, ids are unique, no hardcoded model.
- `tests/test_pipeline.py` — end-to-end `/api/analyze`.

---

## Project structure

```
app/
  ai_client.py    Local Hugging Face VLM client: load, place, generate, decode, repair
  validation.py   Upload integrity, EXIF transpose, response normalization
  diagram.py      Mermaid sanitization, validation, repair prompts
  prompts.py      Mode-specific system prompts and the JSON schema
  main.py         FastAPI routes
static/index.html Frontend (Mermaid rendering, editors, exports)
samples/          Sample input images
scripts/          Sample generator
tests/            Test suite and shared test doubles
```

---

## Security & reliability

- **Secrets stay server-side.** `HF_TOKEN` is read only by the inference layer and is never
  sent to the browser or committed; `.env` is gitignored.
- **Uploads are inspected, not trusted.** Magic-byte sniffing, format agreement, size and
  megapixel ceilings, and EXIF orientation correction.
- **Model output is treated as data.** Generated SQL, Mermaid and prose are never executed.
  Script-like markup in a diagram is rejected, and all rendered text is HTML-escaped.
- **No mocked responses in the product.** The real model runs in every mode; only the test
  suite substitutes fakes.