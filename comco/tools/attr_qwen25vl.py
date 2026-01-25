from __future__ import annotations

import json
from dataclasses import dataclass
from typing import List, Optional, Tuple

from .base import Tool, ToolContext
from .qwen25_vl_client import OllamaQwenVLConfig, invoke_ollama_chat


def _parse_json_list(text: str) -> List[str]:
    """Parse a JSON list from model output; tolerate surrounding text."""
    if not text:
        return []
    text = text.strip()
    # try direct
    try:
        obj = json.loads(text)
        if isinstance(obj, list):
            return [str(x).strip() for x in obj if str(x).strip()]
        if isinstance(obj, dict):
            # common schema: {"attrs":[...]}
            for k in ["attrs", "attributes", "phrases"]:
                if k in obj and isinstance(obj[k], list):
                    return [str(x).strip() for x in obj[k] if str(x).strip()]
    except Exception:
        pass
    # try extract bracketed list
    l = text.find("[")
    r = text.rfind("]")
    if l != -1 and r != -1 and r > l:
        snippet = text[l:r+1]
        try:
            obj = json.loads(snippet)
            if isinstance(obj, list):
                return [str(x).strip() for x in obj if str(x).strip()]
        except Exception:
            pass
    return []


@dataclass
class QwenVisualAttrConfig:
    qwen: OllamaQwenVLConfig
    prompt_template: str = (
        "Extract the most salient visual attributes from this image as short noun phrases. "
        "Return ONLY a JSON array of strings. Example: [\"red backpack\", \"wooden table\"]"
    )
    max_attrs: int = 20


class Qwen2_5VLVisualAttrTool(Tool[Tuple[str, str], List[str]]):
    """(image_id, image_path) -> list of attribute phrases"""

    def __init__(self, cfg: QwenVisualAttrConfig):
        self.cfg = cfg

    def run(self, ctx: ToolContext, inp: Tuple[str, str]) -> List[str]:
        _iid, image_path = inp
        text = invoke_ollama_chat(self.cfg.qwen, self.cfg.prompt_template, image_path=image_path)
        attrs = _parse_json_list(text)
        return attrs[: self.cfg.max_attrs]


@dataclass
class QwenStructAttrConfig:
    qwen: OllamaQwenVLConfig
    prompt_prefix: str = (
        "Extract the most salient entity attributes as short noun phrases from the description. "
        "Return ONLY a JSON array of strings."
    )
    max_attrs: int = 30


class Qwen2_5VLStructAttrTool(Tool[Tuple[str, str], List[str]]):
    """(text_id, description) -> list of attribute phrases"""

    def __init__(self, cfg: QwenStructAttrConfig):
        self.cfg = cfg

    def run(self, ctx: ToolContext, inp: Tuple[str, str]) -> List[str]:
        tid, desc = inp
        prompt = f"{self.cfg.prompt_prefix}\n\n[Entity ID]\n{tid}\n\n[Description]\n{desc}\n\n[Output]\n"
        text = invoke_ollama_chat(self.cfg.qwen, prompt, image_path=None)
        attrs = _parse_json_list(text)
        return attrs[: self.cfg.max_attrs]
