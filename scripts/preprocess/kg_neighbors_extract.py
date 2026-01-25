# -*- coding: utf-8 -*-
"""Extract KG neighbor structured strings (neighbors.txt).

This script is the offline/precompute equivalent of the KG tools:
  KGGraphIndex + bfs_sample + Graph2Text serialization.

Output format (per line):
  <entity_id>\t<structured_text... as TAB-separated lines>

We keep TAB separators to match your original workflow, but the tool pipeline
in CoMCo uses multiline strings internally.
"""

from __future__ import annotations

import argparse
import os

from tqdm import tqdm

from comco.tools.kg_neighbors import KGGraphIndex, KGSampleParams
from comco.tools.kg_serialize import serialize_graph_to_text_desc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_root", type=str, required=True)
    ap.add_argument("--kg_train", type=str, required=True)
    ap.add_argument("--out", type=str, default="neighbors.txt")
    ap.add_argument("--max_hops", type=int, default=1)
    ap.add_argument("--max_edges", type=int, default=5)
    ap.add_argument("--max_nodes", type=int, default=5)
    args = ap.parse_args()

    entities_full = os.path.join(args.dataset_root, "entities_full.txt")
    graph = KGGraphIndex(args.kg_train, entities_full)
    params = KGSampleParams(max_hops=args.max_hops, max_edges=args.max_edges, max_nodes=args.max_nodes)

    out_path = os.path.join(args.dataset_root, args.out)
    with open(entities_full, "r", encoding="utf-8") as f:
        entity_ids = [ln.split("\t", 1)[0].strip() for ln in f if ln.strip()]

    with open(out_path, "w", encoding="utf-8") as fw:
        for eid in tqdm(entity_ids, desc="Extracting neighbors"):
            g = graph.bfs_sample(eid, params)
            text = serialize_graph_to_text_desc(g, graph)
            fw.write(f"{eid}\t" + text.replace("\n", "\t") + "\n")

    print(f"Done. Saved -> {out_path}")


if __name__ == "__main__":
    main()
