from __future__ import annotations

"""Graph2Text serialization for KG neighbor subgraphs.
"""

from typing import Dict

from .kg_neighbors import KGGraphIndex, _clean_rel


def serialize_graph_to_text_desc(g: Dict, graph: KGGraphIndex) -> str:
    center = g.get("center", "")
    edges = g.get("edges", []) or []
    center_desc = graph.attrs.get(center, {}).get("desc", "") or center

    lines = [f"[ENTITY] {center_desc}"]
    for (h, r, t) in edges:
        t_desc = graph.attrs.get(t, {}).get("desc", "") or t
        r_txt = _clean_rel(r)
        lines.append(f"[RELATION] {r_txt} -> {t_desc}")
    return "\n".join(lines).strip()
