from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import numpy as np
from sentence_transformers import SentenceTransformer

from .base import Tool, ToolContext


@dataclass
class SentenceEncoderConfig:
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    device: str = "cpu"
    batch_size: int = 64
    normalize: bool = True


class SentenceEncoderTool(Tool[Sequence[str], np.ndarray]):
    """Encode a list of strings into sentence embeddings [N,D]."""

    def __init__(self, cfg: SentenceEncoderConfig):
        self.cfg = cfg
        self._model: SentenceTransformer | None = None

    def _ensure(self) -> SentenceTransformer:
        if self._model is None:
            self._model = SentenceTransformer(self.cfg.model_name, device=self.cfg.device)
        return self._model

    def run(self, ctx: ToolContext, inp: Sequence[str]) -> np.ndarray:
        model = self._ensure()
        embs = model.encode(
            list(inp),
            batch_size=self.cfg.batch_size,
            normalize_embeddings=self.cfg.normalize,
            show_progress_bar=False,
        )
        return np.asarray(embs, dtype=np.float32)
