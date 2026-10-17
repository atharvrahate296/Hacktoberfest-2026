"""Mermaid sanitization and validation.

The diagram is an enhancement. Structured notes are the product. Everything
here is written so that a hostile or broken diagram can never take down the
notes -- the worst case is that we show the source in a code block.
"""

from __future__ import annotations

import re

# Diagram types we render. Anything else is rejected rather than trusted.
SUPPORTED_TYPES = {"flowchart", "graph", "mindmap", "sequenceDiagram"}

# Conservative ceiling. A hand-drawn whiteboard should never need more.
MAX_MERMAID_CHARS = 4000
MAX_MERMAID_LINES = 200

# Anchored form, used to confirm the source really is a diagram.
_TYPE_RE = re.compile(r"^\s*(flowchart|graph|mindmap|sequenceDiagram)\b", re.IGNORECASE)
# Unanchored form, used to find where the diagram starts when prose precedes it.
_TYPE_ANYWHERE_RE = re.compile(
    r"(?:^|\n)\s*(flowchart|graph|mindmap|sequenceDiagram)\b", re.IGNORECASE
)
_FENCE_RE = re.compile(r"```(?:mermaid)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)

# Anything that could execute or exfiltrate. Mermaid has no legitimate need for
# these in a diagram body, so their presence means we discard the input.
_DANGEROUS = (
    "<script",
    "</script",
    "javascript:",
    "<iframe",
    "<object",
    "<embed",
    "<img",
    "<svg",
    "onerror=",
    "onload=",
    "onclick=",
    "data:text/html",
)


class DiagramError(ValueError):
    """Raised when Mermaid source cannot be accepted. Message is user-facing."""


def strip_fences(source: str) -> str:
    """Remove an optional outer ```mermaid fence and surrounding prose."""
    if not source:
        return ""
    text = source.strip()

    fenced = _FENCE_RE.findall(text)
    if fenced:
        # Take the longest fenced block; a short one is usually stray prose.
        text = max(fenced, key=len).strip()

    # If prose surrounds a diagram declaration, keep from the declaration on.
    match = _TYPE_ANYWHERE_RE.search(text)
    if match:
        text = text[match.start() :].strip()

    return text


def _reject_unsafe(source: str) -> None:
    lowered = source.lower()
    for token in _DANGEROUS:
        if token in lowered:
            raise DiagramError(
                "The diagram contained markup or script-like content and was rejected "
                "for safety."
            )


def _check_shape(source: str) -> None:
    if not source:
        raise DiagramError("The diagram was empty.")
    if len(source) > MAX_MERMAID_CHARS:
        raise DiagramError(
            f"The diagram was {len(source)} characters; the limit is "
            f"{MAX_MERMAID_CHARS}. It was omitted rather than truncated."
        )
    lines = [ln for ln in source.splitlines() if ln.strip()]
    if len(lines) > MAX_MERMAID_LINES:
        raise DiagramError(
            f"The diagram had {len(lines)} lines; the limit is {MAX_MERMAID_LINES}."
        )


def _check_types(source: str) -> None:
    match = _TYPE_RE.match(source)
    if not match:
        head = source.splitlines()[0][:60] if source.splitlines() else source[:60]
        raise DiagramError(
            f"Unsupported diagram type (starts with '{head}'). "
            f"Supported types: {', '.join(sorted(SUPPORTED_TYPES))}."
        )
    declared = match.group(1).lower()
    # 'graph' is a legacy alias for flowchart; normalize so the renderer and the
    # UI agree on one name.
    return "flowchart" if declared == "graph" else declared


def _check_syntax_heuristics(source: str) -> None:
    """Cheap structural checks that catch the most common generation failures."""
    # Bracket/quote balance. Unbalanced delimiters are the single most common
    # defect in model-generated Mermaid and render as a hard failure.
    for opener, closer in (("[", "]"), ("(", ")"), ("{", "}")):
        if source.count(opener) != source.count(closer):
            raise DiagramError(
                f"Unbalanced '{opener}{closer}' in the diagram "
                f"({source.count(opener)} vs {source.count(closer)})."
            )
    if source.count('"') % 2 != 0:
        raise DiagramError("Unbalanced double quotes in the diagram.")

    first = source.splitlines()[0].strip()
    body = "\n".join(source.splitlines()[1:]).strip()
    if first.lower().startswith(("flowchart", "graph")) and not body:
        raise DiagramError("The diagram declared no nodes.")

    if "-->" not in source and "---" not in source and "-" not in source:
        # mindmap and sequenceDiagram legitimately have no arrows.
        if not first.lower().startswith(("mindmap", "sequencediagram")):
            raise DiagramError("The diagram contains no relationships or steps.")

    _check_node_labels(source)


# A bare node id containing a space is the single most common way a model
# produces Mermaid that balances fine and still will not parse:
#   Write --> Invalidate cache      <- rejected by mermaid, needs Write[Write]
# Bracketed, parenthesised, braced and quoted labels are all legal.
_LINK_SIDE = re.compile(r"(?:-->|---|-\.->|==>)\s*(.+)$")
_LABELLED = re.compile(r"\[[^\]]*\]|\([^)]*\)|\{[^}]*\}|\"[^\"]*\"|'[^']*'")


def _check_node_labels(source: str) -> None:
    """Reject unbracketed multi-word node references on link lines."""
    for line in source.splitlines()[1:]:
        stripped = line.strip()
        if not stripped or stripped.startswith("%%"):
            continue
        match = _LINK_SIDE.search(stripped)
        if not match:
            continue
        tail = _LABELLED.sub("", match.group(1)).strip()
        # Remove edge decorations the model may append, e.g. "--> B ~~~ yes".
        tail = re.sub(r"[~|].*$", "", tail).strip()
        if not tail:
            continue
        if " " in tail:
            raise DiagramError(
                f"Node label must be bracketed when it contains spaces: "
                f"'{tail}' (found on line: {stripped[:60]})."
            )


def validate_mermaid(source: str, explanation: str = "") -> dict:
    """Validate Mermaid source. Raises DiagramError if unusable.

    Returns {"type", "mermaid", "explanation"} on success.
    """
    if source is None:
        raise DiagramError("No diagram was provided.")

    cleaned = strip_fences(source)
    _check_shape(cleaned)
    _reject_unsafe(cleaned)
    diagram_type = _check_types(cleaned)
    _check_syntax_heuristics(cleaned)

    # Normalize: single trailing newline, no tabs.
    cleaned = cleaned.replace("\t", "    ").rstrip() + "\n"

    return {
        "type": diagram_type,
        "mermaid": cleaned,
        "explanation": explanation.strip() if isinstance(explanation, str) else "",
    }


def build_repair_prompt(source: str, error: str) -> str:
    """Message sent back to the model for the single corrective retry."""
    return (
        "Your Mermaid diagram was rejected by the validator with this error:\n"
        f"{error}\n\n"
        "Return ONLY a corrected Mermaid diagram, with no prose and no code fence.\n"
        "Requirements:\n"
        "  - First line must be exactly: flowchart TD\n"
        "  - EVERY node label containing a space MUST be wrapped in square "
        "brackets. This is the most common failure:\n"
        "      WRONG:  Write --> Invalidate cache\n"
        "      RIGHT:  Write[Write] --> IC[Invalidate cache]\n"
        "  - Brackets must balance and stay on one line.\n"
        "  - No HTML, <script>, <style>, or any tag.\n"
        "  - No parentheses containing commas or colons inside labels.\n"
        "  - Keep it under 40 lines.\n\n"
        f"The diagram you attempted:\n{source}\n\n"
        "Return only the corrected diagram."
    )
