from __future__ import annotations

import base64
from dataclasses import dataclass
from typing import Optional

import requests


@dataclass
class OllamaQwenVLConfig:
    """Config for Qwen2.5-VL served via Ollama chat endpoint."""
    model: str = "qwen2.5vl:7b"
    host: str = "127.0.0.1"
    temperature: float = 0.01


def invoke_ollama_chat(
    cfg: OllamaQwenVLConfig,
    prompt: str,
    image_path: Optional[str] = None,
) -> str:
    """Invoke Ollama /api/chat. If image_path is provided, send as base64 image."""
    url = f"http://{cfg.host}:11434/api/chat"

    images = []
    if image_path:
        try:
            with open(image_path, "rb") as f:
                images = [base64.b64encode(f.read()).decode("utf-8")]
        except Exception:
            images = []

    payload = {
        "model": cfg.model,
        "messages": [
            {
                "role": "user",
                "content": prompt,
                "images": images,
                "temperature": cfg.temperature,
            }
        ],
        "stream": False,
    }

    resp = requests.post(url, json=payload, timeout=120)
    resp.raise_for_status()
    data = resp.json()
    return (data.get("message", {}).get("content", "") or "").strip()
