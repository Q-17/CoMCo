from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


Edge = Tuple[str, str]  # (image_id, text_id)


@dataclass
class BlackboardState:
    """Shared blackboard state (read/write by all agents).

    The state intentionally separates:
    - tool outputs (embeddings, captions, attributes)
    - candidate lists
    - per-signal scores and fused scores
    - global calibration artifacts (groups, hubness)
    - control directives (weights, hard cases, stop flag)
    """

    # inventories
    image_ids: List[str]
    text_ids: List[str]

    # dataset mode ("table" or "kg")
    dataset_mode: str = "table"
    dataset_name: str = ""

    # resolved inputs
    image_paths: Dict[str, str]
    text_descs: Dict[str, str]
    # Optional: for table datasets where entities_full includes a name field.
    text_names: Dict[str, str] = field(default_factory=dict)

    # tool caches (entity-level)
    anchor_image_emb: Dict[str, np.ndarray] = field(default_factory=dict)
    anchor_text_emb: Dict[str, np.ndarray] = field(default_factory=dict)

    captions: Dict[str, str] = field(default_factory=dict)
    summaries: Dict[str, str] = field(default_factory=dict)
    # KG-only structured serialization cache (e.g., graph neighbors in text form)
    serialized_struct: Dict[str, str] = field(default_factory=dict)
    visual_attrs: Dict[str, List[str]] = field(default_factory=dict)
    struct_attrs: Dict[str, List[str]] = field(default_factory=dict)

    # candidates
    candidates: Dict[str, List[str]] = field(default_factory=dict)  # image_id -> [text_id]

    # signal scores (per edge)
    signals: Dict[Edge, Dict[str, float]] = field(default_factory=dict)  # {anch, sem, attr, mllm}

    # derived scores (per edge)
    scores: Dict[Edge, Dict[str, float]] = field(default_factory=dict)  # {evid, cal, reg}

    # global artifacts
    groups: Dict[str, List[str]] = field(default_factory=dict)  # group_id -> image_ids
    image2group: Dict[str, str] = field(default_factory=dict)
    hubness: Dict[str, float] = field(default_factory=dict)  # text_id -> [0,1]

    # control directives
    weights: Dict[str, float] = field(
        default_factory=lambda: {"anch": 1.0, "sem": 0.0, "attr": 0.0, "mllm": 0.0}
    )
    hard_cases: List[str] = field(default_factory=list)
    stop: bool = False

    # bookkeeping
    round_idx: int = 0

    # diagnostics
    avg_margin: float = 0.0
    avg_group_coherence: float = 0.0

    def edge(self, image_id: str, text_id: str) -> Edge:
        return (image_id, text_id)

    def ensure_edge(self, image_id: str, text_id: str) -> None:
        e = (image_id, text_id)
        if e not in self.signals:
            self.signals[e] = {}
        if e not in self.scores:
            self.scores[e] = {}

    def topk(self, image_id: str, key: str = "reg", k: int = 10) -> List[Tuple[str, float]]:
        """Return top-k candidates for an image by a given score key."""
        cand = self.candidates.get(image_id, [])
        pairs: List[Tuple[str, float]] = []
        for tid in cand:
            v = self.scores.get((image_id, tid), {}).get(key, None)
            if v is None:
                continue
            pairs.append((tid, float(v)))
        pairs.sort(key=lambda x: -x[1])
        return pairs[:k]
