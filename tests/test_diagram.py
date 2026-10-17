"""Mermaid pipeline tests. Pure functions, no network."""

import pytest

from app import diagram as dgm

GOOD = "flowchart TD\n    A[Start] --> B[Next]"


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
        out = dgm.validate_mermaid("mindmap\n  root((Ideas))\n    Alpha\n    Beta")
        assert out["type"] == "mindmap"

    def test_fenced_valid_source_accepted(self):
        assert dgm.validate_mermaid("```mermaid\n" + GOOD + "\n```")["type"] == "flowchart"

    def test_empty_rejected(self):
        with pytest.raises(dgm.DiagramError, match="empty"):
            dgm.validate_mermaid("   ")

    def test_none_rejected(self):
        with pytest.raises(dgm.DiagramError, match="No diagram"):
            dgm.validate_mermaid(None)

    def test_oversized_rejected(self):
        big = "flowchart TD\n" + "\n".join(f"    N{i} --> N{i+1}" for i in range(400))
        with pytest.raises(dgm.DiagramError, match="limit is"):
            dgm.validate_mermaid(big)

    def test_too_many_lines_rejected(self):
        many = "flowchart TD\n" + "\n".join(f"    N{i}[n{i}] --> M{i}[m{i}]" for i in range(150))
        with pytest.raises(dgm.DiagramError):
            dgm.validate_mermaid(many)

    @pytest.mark.parametrize(
        "bad",
        [
            "<script>alert(1)</script>",
            'flowchart TD\n A["x"] --> B\n click A "javascript:alert(1)"',
            "<iframe src=evil></iframe>",
            "flowchart TD\n A[<img src=x onerror=alert(1)>]",
        ],
    )
    def test_dangerous_content_rejected(self, bad):
        with pytest.raises(dgm.DiagramError, match="script-like|markup"):
            dgm.validate_mermaid(bad)

    def test_unsupported_type_rejected(self):
        with pytest.raises(dgm.DiagramError, match="Unsupported diagram type"):
            dgm.validate_mermaid("pie title X\n    \"a\" : 10")

    def test_unbalanced_brackets_rejected(self):
        with pytest.raises(dgm.DiagramError, match="Unbalanced"):
            dgm.validate_mermaid("flowchart TD\n    A[Start --> B")

    def test_unbalanced_quotes_rejected(self):
        with pytest.raises(dgm.DiagramError, match="Unbalanced double quotes"):
            dgm.validate_mermaid('flowchart TD\n    A["unclosed] --> B[x]')

    def test_declaration_without_nodes_rejected(self):
        with pytest.raises(dgm.DiagramError, match="no nodes"):
            dgm.validate_mermaid("flowchart TD")

    def test_no_relationships_rejected(self):
        with pytest.raises(dgm.DiagramError, match="no relationships"):
            dgm.validate_mermaid("flowchart TD\n    Alpha\n    Beta")

    def test_tabs_normalized(self):
        out = dgm.validate_mermaid("flowchart TD\n\tA[x]\t-->\tB[y]")
        assert "\t" not in out["mermaid"]

    @pytest.mark.parametrize(
        "bad",
        [
            # Bare multi-word node ids: balances fine, will not parse.
            "flowchart TD\n    Write --> Invalidate cache",
            "flowchart TD\n    A[x] --> catch up now",
            "flowchart TD\n    A --> B --> Replica lag",
        ],
    )
    def test_unbracketed_multword_labels_rejected(self, bad):
        """The most common real model output that balances but cannot parse."""
        with pytest.raises(dgm.DiagramError, match="bracketed"):
            dgm.validate_mermaid(bad)

    @pytest.mark.parametrize(
        "good",
        [
            "flowchart TD\n    W[Write] --> IC[Invalidate cache]",
            'flowchart TD\n    W["Invalidate cache"] --> R[Replicas]',
            "flowchart TD\n    W(Write) --> R(Replicas)",
            "flowchart TD\n    W{Write} --> R{Replicas}",
            "flowchart TD\n    W[Write] --> R[Replicas] ~~~ yes",
        ],
    )
    def test_bracketed_labels_accepted(self, good):
        assert dgm.validate_mermaid(good)["type"] == "flowchart"

    def test_repair_prompt_calls_out_the_bracketing_rule(self):
        p = dgm.build_repair_prompt("flowchart TD\n Write --> Invalidate cache", "err")
        assert "square brackets" in p.lower()
        assert "Write[Write] --> IC[Invalidate cache]" in p

    def test_explanation_preserved(self):
        out = dgm.validate_mermaid(GOOD, explanation="shows the two steps")
        assert out["explanation"] == "shows the two steps"


class TestRepairPrompt:
    def test_contains_error_and_source(self):
        p = dgm.build_repair_prompt(GOOD, "Unbalanced '[]'")
        assert "Unbalanced '[]'" in p
        assert "flowchart TD" in p
        assert GOOD in p

    def test_instructs_bare_output(self):
        p = dgm.build_repair_prompt(GOOD, "err")
        assert "ONLY a corrected Mermaid diagram" in p


class TestFailureNeverBreaksNotes:
    """The product invariant: a bad diagram must not cost the user their notes."""

    def test_notes_survive_a_rejected_diagram(self):
        from app import validation

        payload = {
            "title": "Board",
            "notes": [{"heading": "Ideas", "content": ["Ship it"], "confidence": "high"}],
            "takeaways": ["Prioritise shipping"],
            "diagram": {"type": "flowchart", "mermaid": "<script>bad()</script>"},
        }
        e = validation.normalize_extraction(payload, diagram_validated=None)
        assert e.diagram is None
        assert e.notes[0]["content"] == ["Ship it"]
        assert e.takeaways == ["Prioritise shipping"]
