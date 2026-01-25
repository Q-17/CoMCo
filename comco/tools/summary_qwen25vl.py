from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from .base import Tool, ToolContext
from .qwen25_vl_client import OllamaQwenVLConfig, invoke_ollama_chat


@dataclass
class QwenSummaryConfig:
    qwen: OllamaQwenVLConfig
    prompt_prefix: str = (
        "You are an expert entity summarizer. Given an entity description, "
        "write a short summary (1-2 sentences) that captures the distinctive information for matching."
    )


class Qwen2_5VLSummaryTool(Tool[Tuple[str, str], str]):
    """(text_id, description) -> summary"""

    def __init__(self, cfg: QwenSummaryConfig):
        self.cfg = cfg

    def run(self, ctx: ToolContext, inp: Tuple[str, str]) -> str:
        tid, desc = inp
        prompt = f"{self.cfg.prompt_prefix}\n\n[Entity ID]\n{tid}\n\n[Description]\n{desc}\n\n[Output]\nSummary:"
        return invoke_ollama_chat(self.cfg.qwen, prompt, image_path=None)
