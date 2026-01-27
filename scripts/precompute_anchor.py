#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import os

from comco.data.catalog import DatasetCatalog
from comco.core.registry import ToolRegistry, sha1_file, sha1_text
from comco.tools.base import ToolContext
from comco.tools.anchor_clip import ClipAnchorConfig, ClipModelProvider, build_anchor_matrices


def build_fingerprint(catalog: DatasetCatalog, clip_model: str) -> str:
    h1 = sha1_file(catalog.imageid_file)
    h2 = sha1_file(catalog.entities_full_file)
    return sha1_text(f"{h1}:{h2}:{clip_model}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_root", type=str, required=True)
    ap.add_argument("--image_root", type=str, required=True)
    ap.add_argument("--clip_model", type=str, default="ViT-L/14")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--cache_dir", type=str, default=".cache/comco")
    args = ap.parse_args()

    catalog = DatasetCatalog.from_args(dataset_root=args.dataset_root, image_root=args.image_root)
    image_ids, image_paths, text_ids, text_descs = catalog.load()

    fp = build_fingerprint(catalog, args.clip_model)
    registry = ToolRegistry(cache_dir=args.cache_dir, dataset_fingerprint=fp)

    cfg = ClipAnchorConfig(model_name=args.clip_model)
    provider = ClipModelProvider(cfg)

    img_mat, txt_mat = build_anchor_matrices(
        provider=provider,
        cfg=cfg,
        image_ids=image_ids,
        image_paths=image_paths,
        text_ids=text_ids,
        text_descs=text_descs,
        device=args.device,
    )

    
    clip_ns = args.clip_model.replace("/", "")
    registry.get_or_run_numpy(namespace=f"{clip_ns}", key="anchor_image_matrix", fn=lambda: img_mat)
    registry.get_or_run_numpy(namespace=f"{clip_ns}", key="anchor_text_matrix", fn=lambda: txt_mat)

    print("[OK] Anchor matrices cached.")
    print("  image matrix:", img_mat.shape)
    print("  text  matrix:", txt_mat.shape)


if __name__ == "__main__":
    main()
