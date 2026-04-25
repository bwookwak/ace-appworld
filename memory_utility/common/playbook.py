"""Playbook parsing + rendering utilities.

An AppWorld playbook is a plain text file whose content is a sequence of
bulleted insights, each formatted as `[<id>] <content>`, grouped under
markdown-style section headers (`## STRATEGIES AND HARD RULES`, etc.).
This module converts between that text form and the pipeline's
structured `memory.json` representation.
"""
from __future__ import annotations

import re
from typing import Any

_BULLET = re.compile(r"^\s*\[([a-zA-Z0-9_\-]+)\]\s*(.*)$")
_HEADER = re.compile(r"^##\s+(.+?)\s*:?\s*$")


def parse_playbook_text(text: str) -> list[dict[str, Any]]:
    """Parse a playbook .txt into a list of insight dicts.

    Returns a list of {id, text, section} preserving the order in the file.
    Multi-line continuations of a bullet are joined with a space.
    """
    current_section = ""
    insights: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    for raw in text.splitlines():
        line = raw.rstrip()
        if not line.strip():
            if current is not None:
                insights.append(current)
                current = None
            continue
        h = _HEADER.match(line)
        if h:
            if current is not None:
                insights.append(current)
                current = None
            current_section = h.group(1).strip()
            continue
        m = _BULLET.match(line)
        if m:
            if current is not None:
                insights.append(current)
            current = {
                "id": m.group(1),
                "text": m.group(2).strip(),
                "section": current_section or "OTHERS",
            }
        else:
            # continuation of previous bullet
            if current is not None:
                current["text"] = (current["text"] + " " + line.strip()).strip()
    if current is not None:
        insights.append(current)
    return insights


def render_playbook(insights: list[dict[str, Any]], sections_order: list[str] | None = None) -> str:
    """Render a list of insights back into the `[id] content` text format.

    Insights are grouped by `section`. Section order follows `sections_order`
    if provided; otherwise sections appear in first-seen order.
    """
    by_section: dict[str, list[dict[str, Any]]] = {}
    section_seen: list[str] = []
    for ins in insights:
        sec = ins.get("section") or "OTHERS"
        if sec not in by_section:
            by_section[sec] = []
            section_seen.append(sec)
        by_section[sec].append(ins)
    if sections_order is None:
        order = section_seen
    else:
        order = list(sections_order) + [s for s in section_seen if s not in sections_order]
    parts: list[str] = []
    for sec in order:
        if sec not in by_section:
            continue
        parts.append(f"## {sec}")
        for ins in by_section[sec]:
            parts.append(f"[{ins['id']}] {ins['text']}")
        parts.append("")
    return "\n".join(parts).rstrip() + "\n"


def token_length_heuristic(text: str) -> int:
    """Cheap token count estimate. Uses tiktoken if available, else len/4."""
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        return max(1, len(text) // 4)


_CITATION_LINE = re.compile(r"\[cited_insights:\s*([^\]]*)\]")
_INLINE_ID = re.compile(r"\[([a-z]+-\d{5})\]")


def parse_citations(text: str) -> dict[str, list[str]]:
    """Extract cited insight IDs from an agent turn.

    Returns a dict with two lists:
      - `structured`: IDs from `[cited_insights: id1, id2]` lines
      - `inline`: IDs matching `[a-z]+-\\d{5}` anywhere else in the text

    The caller decides which to use (or to take the union).
    """
    structured: list[str] = []
    for m in _CITATION_LINE.finditer(text):
        payload = m.group(1).strip()
        if not payload or payload.lower() == "none":
            continue
        for part in payload.split(","):
            ident = part.strip()
            if ident:
                structured.append(ident)
    # For inline, strip anything inside a cited_insights line first so we
    # don't double-count it.
    cleaned = _CITATION_LINE.sub("", text)
    inline = list({m.group(1) for m in _INLINE_ID.finditer(cleaned)})
    return {"structured": structured, "inline": inline}
