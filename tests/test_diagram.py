"""Mermaid pipeline tests. Pure functions, no network."""

import pytest

from app import diagram as dgm

GOOD = "flowchart TD\n    A[Start] --> B[Next]"
GOOD_ER = """erDiagram
    USER ||--o{ ORDER : places
    USER {
        int id PK
        string name
    }"""
GOOD_MINDMAP = """mindmap
  root((Ideas))
    Alpha
    Beta"""


class TestStripFences:
    def test_plain_source_unchanged(self):
        assert dgm.strip_fences(GOOD) == GOOD

    def test_mermaid_fence_removed(self):
        assert dgm.strip_fences("```mermaid\n" + GOOD + "\n```") == GOOD

    def test_bare_fence_removed(self):
        assert dgm.strip_fences("```\n" + GOOD + "\n```") == GOOD

    def test_leading_prose_discarded(self):
        text = "Sure! Here is the diagram:\n\n" + GOOD
        assert dgm.strip_fences(text).startswith("flowchart TD")

    def test_empty_input(self):
        assert dgm.strip_fences("") == ""


class TestValidateMermaid:
    def test_valid_flowchart(self):
        out = dgm.validate_mermaid(GOOD)
        assert out["type"] == "flowchart"
        assert out["mermaid"].endswith("\n")

    def test_graph_alias_normalized_to_flowchart(self):
        out = dgm.validate_mermaid("graph LR\n    A --> B")
        assert out["type"] == "flowchart"

    def test_mindmap_supported(self):
        out = dgm.validate_mermaid(GOOD_MINDMAP)
        assert out["type"] == "mindmap"

    def test_er_diagram_supported(self):
        out = dgm.validate_mermaid(GOOD_ER)
        assert out["type"] == "erDiagram"

    def test_fenced_valid_source_accepted(self):
        assert dgm.validate_mermaid("```mermaid\n" + GOOD + "\n```")["type"] == "flowchart"

    def test_empty_rejected(self):
        with pytest.raises(dgm.DiagramError, match="empty"):
            dgm.validate_mermaid("   ")

    def test_none_rejected(self):
        with pytest.raises(dgm.DiagramError, match="No diagram"):
            dgm.validate_mermaid(None)

    def test_oversized_rejected(self):
        big = "flowchart TD\n" + "\n".join(f"    N{i} --> N{i+1}" for i in range(500))
        with pytest.raises(dgm.DiagramError, match="limit is"):
            dgm.validate_mermaid(big)

    def test_too_many_lines_rejected(self):
        many = "flowchart TD\n" + "\n".join(f"    N{i}[n{i}] --> M{i}[m{i}]" for i in range(260))
        with pytest.raises(dgm.DiagramError, match="limit is"):
            dgm.validate_mermaid(many)

    def test_unsupported_type_rejected(self):
        with pytest.raises(dgm.DiagramError, match="Unsupported diagram type"):
            dgm.validate_mermaid("pie title Pets\n    \"Dogs\" : 386")

    def test_script_tag_rejected(self):
        bad = "flowchart TD\n    A[Start] --> B[<script>alert(1)</script>]"
        with pytest.raises(dgm.DiagramError, match="rejected for safety"):
            dgm.validate_mermaid(bad)

    def test_unbalanced_brackets_rejected(self):
        bad = "flowchart TD\n    A[Start --> B[Next]"
        with pytest.raises(dgm.DiagramError, match="Unbalanced"):
            dgm.validate_mermaid(bad)

    def test_unbracketed_node_label_with_spaces_rejected(self):
        bad = "flowchart TD\n    Write --> Invalidate cache"
        with pytest.raises(dgm.DiagramError, match="bracketed when it contains spaces"):
            dgm.validate_mermaid(bad)


class TestRepairPrompt:
    def test_repair_prompt_contains_error_and_type(self):
        prompt = dgm.build_repair_prompt("flowchart TD\nA --> B", "Node label error", expected_type="flowchart")
        assert "Node label error" in prompt
        assert "flowchart TD" in prompt

    def test_repair_prompt_for_er(self):
        prompt = dgm.build_repair_prompt("erDiagram\nA ||--o{ B", "Unbalanced quotes", expected_type="erDiagram")
        assert "erDiagram" in prompt
