"""Frontend contract tests for static/index.html.

The browser is not exercised here, but the failures these guard against are all
static: an element id referenced by the render code must exist, the model name
must come from /api/config rather than being hardcoded, and the view must not
render prose explanation in place of a reconstruction.
"""

from __future__ import annotations

import io
import pathlib
import re
from collections import Counter

import pytest

from app import ai_client

ROOT = pathlib.Path(__file__).resolve().parents[1]
HTML_PATH = ROOT / "static" / "index.html"


@pytest.fixture(scope="module")
def html() -> str:
    return io.open(HTML_PATH, encoding="utf-8").read()


def test_model_is_not_hardcoded(html: str):
    # The badge is filled from /api/config; no model may ship as placeholder copy.
    assert "Qwen/Qwen2-VL-2B-Instruct" not in html
    assert "Gemma" not in html
    assert '<span id="modelTag">loading…</span>' in html


def test_model_copy_is_bound_to_config(html: str):
    assert 'id="modelFootnote"' in html
    assert '$("modelTag").textContent = c.model;' in html


def test_no_stale_model_as_a_configured_default():
    for name in ("readme.md", ".env.example"):
        text = io.open(ROOT / name, encoding="utf-8").read()
        assert f"HF_MODEL_ID=google/gemma-3-27b-it" not in text
        assert f"HF_MODEL_ID=Qwen/Qwen2-VL-2B-Instruct" not in text
    assert 'HF_MODEL_ID=HuggingFaceTB/SmolVLM-500M-Instruct' in io.open(
        ROOT / ".env.example", encoding="utf-8"
    ).read()


@pytest.mark.parametrize(
    "target",
    ["mmdMain", "sideMmd"],
    ids=["diagram-panel", "comparison-panel"],
)
def test_render_targets_exist_in_the_dom(html: str, target: str):
    """drawMermaid() returns silently when its target id is absent.

    A mismatch here is what left the side-by-side diagram blank.
    """
    assert f'id="{target}"' in html, f'drawMermaid target {target} has no element'
    assert f'drawMermaid("{target}"' in html, f"{target} is never rendered"


def test_mermaid_render_ids_are_unique(html: str):
    """mermaid.render() injects a temporary node under the id it is given."""
    assert 'mermaid.render(renderId, code)' in html
    assert "mermaid.render(id + \"Svg\"" not in html
    assert "++mermaidRenderCount" in html


def test_render_guards_against_a_detached_panel(html: str):
    assert "el.isConnected" in html


def test_svg_export_reads_the_rendered_node(html: str):
    assert 'id="dlSvgBtn"' in html
    assert 'document.querySelector("#mmdMain svg")' in html
    assert '"image/svg+xml"' in html


def test_element_ids_are_unique(html: str):
    """Duplicate ids break the getElementById wiring between panels."""
    ids = re.findall(r'\sid="([^"]+)"', html)
    dupes = [i for i, n in Counter(ids).items() if n > 1]
    assert not dupes, f"duplicate element ids: {sorted(dupes)}"


def test_explanation_prose_is_not_rendered(html: str):
    """The reconstruction is the deliverable; prose may be exported only."""
    assert "${d.diagram_explanation" not in html
    assert "${esc(d.diagram_explanation)}" not in html


def test_sample_gallery_is_wired(html: str):
    assert "/api/samples" in html
    assert "/api/config" in html
    assert 'data-mode="${esc(s.mode)}"' in html


def test_failed_runs_show_diagnostics(html: str):
    """An empty result must explain itself rather than render a blank page."""
    assert "function diagnosticsHtml" in html
    assert "${diagnosticsHtml(d, t)}" in html
    assert "t.raw_preview" in html
    assert "t.diagram_error" in html
    assert 'class="diag"' in html


def test_diagnostics_escape_model_output(html: str):
    assert "<pre class=\"src\">${esc(t.raw_preview)}</pre>" in html


def test_rendered_panels_escape_model_text(html: str):
    """Everything interpolated into innerHTML must go through esc()."""
    for field in ("d.title", "n.heading", "n.confidence", "a.task", "u.reason"):
        assert f"esc({field})" in html, f"{field} reaches innerHTML unescaped"
    assert "(n.content || []).map(c => `<li>${esc(c)}</li>`)" in html


def test_default_model_is_documented(html: str):
    assert ai_client.DEFAULT_MODEL in ai_client.KNOWN_MODELS