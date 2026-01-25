from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import requests

from .base import Tool, ToolContext


@dataclass
class OllamaMLLMConfig:
    model: str = "qwen2.5vl:7b"
    host: str = "127.0.0.1"
    temperature: float = 0.01


def invoke_ollama_chat(model: str, host: str, prompt: str, image_path: Optional[str] = None, temperature: float = 0.01) -> str:
    url = f"http://{host}:11434/api/chat"
    encoded_image = None
    if image_path:
        try:
            with open(image_path, "rb") as f:
                encoded_image = base64.b64encode(f.read()).decode("utf-8")
        except Exception:
            encoded_image = None

    data = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": prompt,
                "images": [encoded_image] if encoded_image else [],
                "temperature": temperature,
            }
        ],
        "stream": False,
    }
    r = requests.post(url, json=data, timeout=120)
    r.raise_for_status()
    return r.json()["message"]["content"].replace("\n", " ").strip()


def parse_json_ranking(text: str) -> List[str]:
    """Parse a JSON ranking list.

    Expected output:
      {"ranking": ["id1", "id2", ...]}

    For robustness, we also accept a fallback {"best": "..."} and convert it
    into a length-1 ranking.
    """
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        # attempt extract first JSON object
        s = text.find("{")
        e = text.rfind("}")
        if s == -1 or e == -1 or e <= s:
            return []
        try:
            obj = json.loads(text[s : e + 1])
        except json.JSONDecodeError:
            return []

    if not isinstance(obj, dict):
        return []

    if "ranking" in obj and isinstance(obj["ranking"], list):
        return [str(x) for x in obj["ranking"]]
    if "best" in obj and obj["best"]:
        return [str(obj["best"])]
    return []


def build_prompt(image_id: str, candidates: List[str], text_descs: Dict[str, str]) -> str:
    lines: List[str] = []
    lines.append("You are an expert judge for image-to-entity matching.")
    lines.append("Your task is to RANK all candidates from best to worst for the given image.")
    lines.append("Return JSON ONLY in the following format:")
    lines.append('  {"ranking": ["ENTITY_ID_1", "ENTITY_ID_2", ...]}')
    lines.append("The ranking MUST contain every candidate ID exactly once.")
    lines.append("")
    lines.append("[Candidates]")
    for i, tid in enumerate(candidates, 1):
        desc = text_descs.get(tid, "")
        if len(desc) > 500:
            desc = desc[:500] + " ..."
        lines.append(f"{i}. ENTITY_ID: {tid}")
        lines.append(f"   Description: {desc if desc else '(empty)'}")
    lines.append("")
    lines.append("Output JSON only.")
    return "\n".join(lines)


class OllamaRerankTool(Tool[Tuple[str, str, List[str], Dict[str, str]], List[str]]):
    """MLLM disambiguation tool.

    Input:
      (image_id, image_path, candidate_ids, text_descs)
    Output:
      ranking list (best-first). Implementations should return a full
      permutation of the shortlist whenever possible.
    """

    name = "ollama_mllm_rerank"

    def __init__(self, cfg: OllamaMLLMConfig):
        self.cfg = cfg

    def run(self, ctx: ToolContext, x: Tuple[str, str, List[str], Dict[str, str]]) -> List[str]:
        image_id, image_path, candidates, text_descs = x
        prompt = build_prompt(image_id, candidates, text_descs)
        try:
            text = invoke_ollama_chat(
                model=self.cfg.model,
                host=self.cfg.host,
                prompt=prompt,
                image_path=image_path,
                temperature=self.cfg.temperature,
            )
        except Exception:
            return []

        ranking = parse_json_ranking(text)
        return ranking
