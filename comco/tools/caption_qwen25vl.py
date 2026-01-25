from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from .base import Tool, ToolContext
from .qwen25_vl_client import OllamaQwenVLConfig, invoke_ollama_chat


@dataclass
class QwenCaptionConfig:
    qwen: OllamaQwenVLConfig
    prompt_template: str = (
        "You are a precise image captioner. "
        "Describe the image in one concise sentence focusing on identity, key attributes, and context. "
        "Do NOT add any extra commentary."
    )


class Qwen2_5VLCaptionTool(Tool[Tuple[str, str], str]):
    """(image_id, image_path) -> caption text"""

    def __init__(self, cfg: QwenCaptionConfig):
        self.cfg = cfg

    def run(self, ctx: ToolContext, inp: Tuple[str, str]) -> str:
        _iid, image_path = inp
        return invoke_ollama_chat(self.cfg.qwen, self.cfg.prompt_template, image_path=image_path)
