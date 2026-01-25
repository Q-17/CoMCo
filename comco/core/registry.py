from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np


def sha1_text(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def sha1_file(path: str, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


@dataclass
class ToolRegistry:
    """Simple get-or-run registry with disk caching.

    Tools can persist outputs as:
    - .npy for numpy arrays
    - .json for dict/list primitives

    Agents should call tools via the registry to enable reuse across rounds.
    """

    cache_dir: str
    dataset_fingerprint: str

    def _dir(self, namespace: str) -> Path:
        p = Path(self.cache_dir) / self.dataset_fingerprint / namespace
        p.mkdir(parents=True, exist_ok=True)
        return p

    def get_or_run_numpy(
        self,
        namespace: str,
        key: str,
        fn: Callable[[], np.ndarray],
    ) -> np.ndarray:
        out = self._dir(namespace) / f"{key}.npy"
        if out.exists():
            return np.load(out)
        arr = fn()
        np.save(out, arr.astype(np.float32))
        return arr

    def get_or_run_json(
        self,
        namespace: str,
        key: str,
        fn: Callable[[], Any],
    ) -> Any:
        out = self._dir(namespace) / f"{key}.json"
        if out.exists():
            with open(out, "r", encoding="utf-8") as f:
                return json.load(f)
        obj = fn()
        with open(out, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False)
        return obj
