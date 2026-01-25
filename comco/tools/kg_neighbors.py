from __future__ import annotations

"""KG neighbor subgraph sampling tool.

This module ports your standalone `GraphIndex + bfs_sample` logic into the repo
so that it can be invoked as a Tool from agents, with ToolRegistry caching.

Design goals:
1) Minimal behavior drift from your scripts.
2) Deterministic output for a given (entity_id, params, graph files).
3) Usable for building structured serialization r_s and KG abstracts a_s.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple
from collections import defaultdict, deque


def _clean_rel(rel: str) -> str:
    return rel.replace("_", " ").strip()


@dataclass
class KGSampleParams:
    max_hops: int = 1
    max_edges: int = 5
    max_nodes: int = 5


class KGGraphIndex:
    """A lightweight in-memory KG index.

    Notes
    -----
    - We treat `entity_id` as the *first column* id in entities_full.txt, which
      aligns with your neighbor/abstract generation scripts.
    - The KG triples file is assumed to be TSV with 3 columns: head, relation, tail.
    """

    def __init__(self, kg_train_path: str, entities_full_path: str):
        self.adj_undirected = defaultdict(set)  # entity -> set(neighbors)
        self.out_edges = defaultdict(list)  # entity -> list[(rel, tail)]
        self.attrs: Dict[str, Dict[str, str]] = {}

        self._load_entities_full(entities_full_path)
        self._load_triples(kg_train_path)

    def _load_entities_full(self, path: str) -> None:
        with open(path, "r", encoding="utf-8") as f:
            for ln in f:
                parts = ln.rstrip("\n").split("\t")
                if len(parts) < 2:
                    continue
                ent_id = parts[0].strip()
                # Prefer KG-style 4th column as desc when present.
                desc = parts[3].strip() if len(parts) >= 4 else (parts[-1].strip() if len(parts) >= 2 else "")
                if desc and desc.lower() != "null":
                    self.attrs[ent_id] = {"desc": desc}

    def _load_triples(self, path: str) -> None:
        with open(path, "r", encoding="utf-8") as f:
            for ln in f:
                parts = ln.rstrip("\n").split("\t")
                if len(parts) < 3:
                    continue
                h = parts[0].strip()
                r = _clean_rel(parts[1])
                t = parts[2].strip()
                self.out_edges[h].append((r, t))
                self.adj_undirected[h].add(t)
                self.adj_undirected[t].add(h)

    def bfs_sample(self, center: str, params: KGSampleParams) -> Dict:
        visited = {center}
        q = deque([(center, 0)])
        edges: List[Tuple[str, str, str]] = []

        while q and len(edges) < params.max_edges and len(visited) < params.max_nodes:
            u, d = q.popleft()
            if d >= params.max_hops:
                continue
            for (r, v) in self.out_edges.get(u, []):
                if len(edges) >= params.max_edges:
                    break
                edges.append((u, r, v))
                if v not in visited:
                    visited.add(v)
                    q.append((v, d + 1))

        return {"center": center, "edges": edges, "attrs": self.attrs.get(center, {})}
