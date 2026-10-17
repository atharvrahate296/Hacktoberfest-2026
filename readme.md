# Scribble2Structure

**From messy visual thinking to structured, editable knowledge.**

Scribble2Structure transforms handwritten notes, whiteboard photographs, brainstorming sketches, technical system drawings, and meeting notes into structured, editable knowledge artifacts with rendered Mermaid diagrams, verifiable action plans, and multi-format exports.

Powered by **local multimodal AI inference** using **Qwen2-VL** (or Google Gemma) models via **Hugging Face Transformers** — no cloud API costs, runs fully on your machine.

---

## Features

- **Local Multimodal AI Inference**: Runs vision-language models entirely on your machine using HuggingFace Transformers. Default model: `Qwen/Qwen2-VL-2B-Instruct` (~4 GB). Compatible with `Qwen2.5-VL-3B`, `SmolVLM`, and `google/gemma-3-4b-it`.
- **8 Transformation Modes**:
  1. **Auto Detect** — Adaptive layout & content classifier that selects the optimal mode automatically.
  2. **Handwritten Notes to Structured Notes** — Transcribes handwriting, outlines headings, key concepts, formulas, and definitions.
  3. **Diagram to Flowchart** — Detects steps, arrows, decision points, and generates editable Mermaid flowcharts.
  4. **Brainstorming to Mind Map** — Extracts core themes, categories, radiating ideas, and generates Mermaid mindmaps.
  5. **Technical Diagram to Architecture Document** — Identifies tiers, services, protocols, databases, caches, and generates system architecture diagrams.
  6. **Meeting Notes to Action Plan** — Extracts decisions, confirmed tasks, owners, and deadlines without hallucinating missing data.
  7. **Scientific Concept Map** — Preserves mathematical equations, scientific laws, symbols, units, and concept hierarchies.
  8. **Database ER Diagram** — Extracts relational entities, attributes, PK/FK flags, cardinalities, Mermaid ER diagrams, and SQL table definitions (`CREATE TABLE`).
- **Interactive Mermaid Diagrams**: Visual rendering using Mermaid.js (`flowchart`, `graph`, `mindmap`, `sequenceDiagram`, `erDiagram`), live diagram editor, and `.mmd` download.
- **Side-by-Side Comparison**: Compare the original uploaded whiteboard/notes image against the extracted structured knowledge and rendered diagram.
- **Editable Knowledge & State Management**: Live in-browser editing of titles, executive summaries, notes, action item checkboxes, and diagrams with clean reset and unsaved-changes protection.
- **Multi-Format Exports**: One-click download as **Markdown (`.md`)**, **Plain Text (`.txt`)**, **Structured JSON (`.json`)**, and **Mermaid (`.mmd`)**.
- **Interactive Sample Inputs Gallery**: Includes 6 realistic visual assets covering every transformation mode for zero-effort evaluation.
- **Robust Security**: API tokens remain strictly server-side; input validation rejects oversized or corrupt files, corrects EXIF orientation, and treats model outputs safely without arbitrary code execution.

---

## Installation & Setup

### Prerequisites

- Python 3.10+
- **~4 GB free disk space** (for `Qwen/Qwen2-VL-2B-Instruct` weights, downloaded once on first run)
- Hugging Face Access Token with read permissions (from [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)) — needed only for the initial model download
- *(Optional)* NVIDIA GPU with 4+ GB VRAM for fast inference; CPU works but is slower

### 1. Clone & Environment Setup

```bash
git clone https://github.com/atharvrahate296/Hacktoberfest-2026.git
cd Hacktoberfest-2026

# Create virtual environment
python -m venv .venv
# On Windows:
.\.venv\Scripts\activate
# On Linux / macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment Variables

Copy `.env.example` to `.env`:

```bash
cp .env.example .env
```

Edit `.env` and provide your Hugging Face credentials:

```ini
# Hugging Face access token — https://huggingface.co/settings/tokens
# Only needed for the initial model weight download (gated models)
HF_TOKEN=hf_your_actual_token_here

# Local model to load (weights cached in ~/.cache/huggingface after first run)
# ~4 GB:  Qwen/Qwen2-VL-2B-Instruct      (recommended, fast)
# ~6 GB:  Qwen/Qwen2.5-VL-3B-Instruct    (better quality)
# ~2 GB:  HuggingFaceTB/SmolVLM-Instruct  (smallest)
# ~9 GB:  google/gemma-3-4b-it            (original plan)
HF_MODEL_ID=Qwen/Qwen2-VL-2B-Instruct

# Optional image upload limit in MB (default: 10)
MAX_UPLOAD_MB=10
```

---

## Running the Application

Start the local server using `uvicorn`:

```bash
uvicorn app.main:app --host 127.0.0.1 --port 8090 --reload
```

Open your browser to: **[http://127.0.0.1:8090](http://127.0.0.1:8090)**

---

## Sample Inputs Gallery

The application includes 6 realistic image inputs in `samples/` for immediate testing:

1. **`01_ml_notes.png`** — Handwritten machine learning notes (gradient descent, cost function, L2 regularization).
2. **`02_os_process.png`** — Operating system process state machine (New, Ready, Running, Waiting, Terminated).
3. **`03_software_architecture.png`** — Microservices e-commerce system architecture with API gateway, Redis, and Kafka.
4. **`04_student_productivity.png`** — Mind map of student productivity, study habits, and task management.
5. **`05_meeting_notes.png`** — Sprint sync meeting notes with action items, assignees, and deadlines.
6. **`06_database_er.png`** — Relational database schema with customer, orders, items, and products.

To re-generate or modify sample assets:

```bash
python scripts/create_samples.py
```

---

## Running Automated Tests

Run the complete test suite:

```bash
pytest
```

The tests cover:

- Hugging Face Gemma inference client routing, error translation, rate-limit retries, and corrective repair routines (`tests/test_ai_client.py`).
- Mermaid diagram parsing, sanitization, syntax validation, and repair for flowcharts, mindmaps, and ER diagrams (`tests/test_diagram.py`).
- Upload validation (formats, EXIF orientation, size ceilings, corruption checks) and response normalization (`tests/test_validation.py`).
- End-to-end FastAPI endpoint integration (`tests/test_pipeline.py`).

---

## Architecture

```
.
├── app/
│   ├── __init__.py
│   ├── ai_client.py       # Hugging Face Gemma multimodal client with retries
│   ├── diagram.py         # Mermaid validation, sanitization, & repair (Flowchart, Mindmap, ER)
│   ├── main.py            # FastAPI server routes, health check, config, and sample endpoints
│   ├── prompts.py         # Mode-specific Gemma prompts & system instructions
│   └── validation.py      # Upload integrity, EXIF transpose, & schema normalization
├── samples/               # Usable sample input images
├── scripts/
│   └── create_samples.py  # Sample asset generator
├── static/
│   └── index.html         # Modern web application frontend
├── tests/
│   ├── test_ai_client.py  # Inference client unit tests
│   ├── test_diagram.py    # Diagram validator tests
│   ├── test_pipeline.py   # Full pipeline integration tests
│   └── test_validation.py # Upload & normalization tests
├── .env.example           # Environment template
└── requirements.txt       # Project dependencies
```

---

## Security & Reliability Considerations

- **Server-Side Secret Isolation**: `HF_TOKEN` is read exclusively on the server side and never sent to client code or exposed in public repositories.
- **No Mock Responses**: Real Hugging Face Gemma inference is executed in all modes.
- **Fail-Safe Diagrams**: Broken or untrusted diagrams never crash the application; unparseable source is displayed safely in fenced code blocks while preserving extracted notes and takeaways.
- **No Arbitrary Code Execution**: Generated SQL, Python, or shell code is treated as text data and never executed.
