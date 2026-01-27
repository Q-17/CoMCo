# -*- coding: utf-8 -*-
"""Generate table abstracts from entities_full.txt using Qwen (Ollama).
"""

from __future__ import annotations

import argparse
import os

from tqdm import tqdm

from comco.tools.qwen25_vl_client import OllamaQwenVLConfig, invoke_ollama_chat
from comco.tools.abstract_qwen25 import DEFAULT_TABLE_PROMPT


def infer_entity_type(dataset_name: str) -> str:
    ds = dataset_name.lower()
    if ds == "imdb":
        return "movie"
    if ds == "walmart":
        return "product"
    return "entity"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_root", type=str, required=True)
    ap.add_argument("--dataset_name", type=str, default="")
    ap.add_argument("--input", type=str, default="entities_full.txt")
    ap.add_argument("--output", type=str, default="")
    ap.add_argument("--model", type=str, default="qwen2.5:7b")
    ap.add_argument("--host", type=str, default="127.0.0.1")
    args = ap.parse_args()

    dataset_name = args.dataset_name or os.path.basename(os.path.normpath(args.dataset_root))
    entity_type = infer_entity_type(dataset_name)

    inp = os.path.join(args.dataset_root, args.input)
    out = args.output or os.path.join(args.dataset_root, f"abstractby{args.model.replace(':','-')}.txt")
    qwen = OllamaQwenVLConfig(model=args.model, host=args.host, temperature=0.01)

    with open(inp, "r", encoding="utf-8") as f:
        total = sum(1 for _ in f)

    with open(inp, "r", encoding="utf-8") as f, open(out, "w", encoding="utf-8") as fw:
        for ln in tqdm(f, total=total, desc=f"Getting abstracts for {dataset_name}", dynamic_ncols=True):
            ln = ln.strip()
            if not ln:
                continue
            parts = ln.split("\t")
            if len(parts) < 3:
                continue
            ent_id = parts[0].strip()
            name = parts[-2].strip()
            desc = parts[-1].strip()
            prompt = DEFAULT_TABLE_PROMPT.format(entity_type=entity_type, name=name, description=desc)
            summ = invoke_ollama_chat(qwen, prompt, image_path=None)
            fw.write(f"{ent_id}\t{summ}\n")

    print(f"Done. Saved -> {out}")


if __name__ == "__main__":
    main()
