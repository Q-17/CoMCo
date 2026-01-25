from __future__ import annotations

"""Structured serialization tool for KG entities.

This tool combines:
1) neighbor subgraph sampling (BFS)
2) Graph2Text serialization

So agents can request a deterministic structured text r_s for an entity.
"""

from dataclasses import dataclass
from typing import Tuple

from .base import Tool, ToolContext
from .kg_neighbors import KGGraphIndex, KGSampleParams
from .kg_serialize import serialize_graph_to_text_desc


@dataclass
class KGStructuredTextConfig:
    params: KGSampleParams


class KGStructuredTextTool(Tool[str, str]):
    """entity_id -> structured serialization r_s (multiline string)."""

    def __init__(self, graph: KGGraphIndex, cfg: KGStructuredTextConfig):
        self.graph = graph
        self.cfg = cfg

    def run(self, ctx: ToolContext, entity_id: str) -> str:
        g = self.graph.bfs_sample(entity_id, self.cfg.params)
        return serialize_graph_to_text_desc(g, self.graph)
