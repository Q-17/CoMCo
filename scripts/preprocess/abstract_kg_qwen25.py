# -*- coding: utf-8 -*-
"""Generate KG abstracts from neighbors.txt using Qwen (Ollama).
"""

from __future__ import annotations

import argparse
import os

from tqdm import tqdm

from comco.tools.qwen25_vl_client import OllamaQwenVLConfig, invoke_ollama_chat
from comco.tools.abstract_qwen25 import DEFAULT_KG_PROMPT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_root", type=str, required=True)
    ap.add_argument("--input", type=str, default="neighbors.txt")
    ap.add_argument("--output", type=str, default="")
    ap.add_argument("--model", type=str, default="qwen2.5:7b")
    ap.add_argument("--host", type=str, default="127.0.0.1")
    args = ap.parse_args()

    inp = os.path.join(args.dataset_root, args.input)
    out = args.output or os.path.join(args.dataset_root, f"abstractby{args.model.replace(':','-')}.txt")

    qwen = OllamaQwenVLConfig(model=args.model, host=args.host, temperature=0.01)

    with open(inp, "r", encoding="utf-8") as f:
        total = sum(1 for _ in f)

    with open(inp, "r", encoding="utf-8") as f, open(out, "w", encoding="utf-8") as fw:
        for ln in tqdm(f, total=total, desc="Getting KG abstracts", dynamic_ncols=True):
            ln = ln.strip()
            if not ln:
                continue
            inner_id, neighbors = ln.split("\t", 1)
            neighbors = neighbors.replace("\t", "\n")
            prompt = DEFAULT_KG_PROMPT.format(text=neighbors)
            summ = invoke_ollama_chat(qwen, prompt, image_path=None)
            fw.write(f"{inner_id}\t{summ}\n")

    print(f"Done. Saved -> {out}")


if __name__ == "__main__":
    main()
