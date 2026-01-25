from __future__ import annotations

"""Qwen2.5-based abstract generation tools.

We implement two concrete tools aligned with your scripts:

1) KGAbstractTool: takes the structured serialization r_s (first line entity,
   remaining lines relations) and produces a ONE-sentence abstract.

2) TableAbstractTool: takes (name, description) from a table row and produces a ONE-sentence abstract.

Both tools call the same Ollama qwen2.5 / qwen2.5-vl endpoint (text-only request).
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from .base import Tool, ToolContext
from .qwen25_vl_client import OllamaQwenVLConfig, invoke_ollama_chat


@dataclass
class AbstractConfig:
    qwen: OllamaQwenVLConfig
    # if provided, overrides the default prompt templates
    kg_prompt_template: Optional[str] = None
    table_prompt_template: Optional[str] = None


DEFAULT_KG_PROMPT = """You are a knowledge distillation assistant.

The input is a structured description of an entity:
- The first line describes the entity itself.
- The remaining lines describe relations to neighboring entities.

Your task:
1) Summarize the core identity of the entity in ONE concise sentence.
2) Extract only information explicitly present in the input.
3) Do NOT add any new facts or make assumptions.
4) Avoid emotional, stylistic, or evaluative language.
5) Output only the final clean sentence. No explanation.

{text}
"""


DEFAULT_TABLE_PROMPT = """You are a knowledge distillation assistant.

The input is the information about a single {entity_type} from a table.

- name: {name}
- description: {description}

Your task:
1) Summarize the core identity of this {entity_type} in ONE concise sentence.
2) Use only information explicitly present above.
3) Do NOT add any new facts or make assumptions.
4) Avoid emotional, stylistic, or evaluative language.
5) Output only the final clean sentence. No explanation.
"""


class Qwen2_5VLKGAbstractTool(Tool[Tuple[str, str], str]):
    """(entity_id, structured_text) -> abstract sentence."""

    def __init__(self, cfg: AbstractConfig):
        self.cfg = cfg

    def run(self, ctx: ToolContext, inp: Tuple[str, str]) -> str:
        _eid, structured_text = inp
        tmpl = self.cfg.kg_prompt_template or DEFAULT_KG_PROMPT
        prompt = tmpl.format(text=str(structured_text))
        return invoke_ollama_chat(self.cfg.qwen, prompt, image_path=None)


class Qwen2_5VLTableAbstractTool(Tool[Tuple[str, str, str, str], str]):
    """(entity_id, dataset_name, name, description) -> abstract sentence."""

    def __init__(self, cfg: AbstractConfig, entity_type_map: Optional[Dict[str, str]] = None):
        self.cfg = cfg
        self.entity_type_map = entity_type_map or {"imdb": "movie", "walmart": "product"}

    def run(self, ctx: ToolContext, inp: Tuple[str, str, str, str]) -> str:
        _eid, dataset_name, name, description = inp
        ds = str(dataset_name).lower().strip()
        entity_type = self.entity_type_map.get(ds, "entity")
        tmpl = self.cfg.table_prompt_template or DEFAULT_TABLE_PROMPT
        prompt = tmpl.format(entity_type=entity_type, name=str(name), description=str(description))
        return invoke_ollama_chat(self.cfg.qwen, prompt, image_path=None)
