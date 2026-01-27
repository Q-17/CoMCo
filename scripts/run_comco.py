#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

"""Main entrypoint for running CoMCo.
"""

import argparse
import os

import yaml

from comco.data.catalog import DatasetCatalog
from comco.core.registry import ToolRegistry, sha1_file, sha1_text
from comco.core.state import BlackboardState
from comco.tools.base import ToolContext
from comco.tools.anchor_clip import ClipAnchorConfig, ClipModelProvider, ClipAnchorImageEmbedTool, ClipAnchorTextEmbedTool

from comco.tools.qwen25_vl_client import OllamaQwenVLConfig
from comco.tools.caption_qwen25vl import Qwen2_5VLCaptionTool, QwenCaptionConfig
from comco.tools.summary_qwen25vl import Qwen2_5VLSummaryTool, QwenSummaryConfig
from comco.tools.abstract_qwen25 import AbstractConfig, Qwen2_5VLKGAbstractTool, Qwen2_5VLTableAbstractTool
from comco.tools.attr_qwen25vl import (
    Qwen2_5VLVisualAttrTool,
    Qwen2_5VLStructAttrTool,
    QwenVisualAttrConfig,
    QwenStructAttrConfig,
)
from comco.tools.sentence_encoder import SentenceEncoderTool, SentenceEncoderConfig
from comco.tools.kg_neighbors import KGGraphIndex, KGSampleParams
from comco.tools.kg_structured_text import KGStructuredTextTool, KGStructuredTextConfig
from comco.tools.mllm_rerank import OllamaMLLMConfig, OllamaRerankTool

from comco.pipeline.coordinator import Coordinator, CoordinatorConfig
from comco.agents.match_agent import MatchAgent, MatchAgentConfig
from comco.agents.control_agent import ControlAgent, ControlAgentConfig
from comco.agents.rerank_agent import RerankAgent, RerankAgentConfig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_name", type=str, default="", help="Name used for caching (e.g., WN18 / imdb)")
    ap.add_argument("--dataset_root", type=str, required=True, help="Path to dataset folder containing entities_full.txt and imageid.txt")
    ap.add_argument("--image_root", type=str, required=True, help="Path to raw images root folder")
    ap.add_argument("--config", type=str, default=os.path.join(os.path.dirname(__file__), "..", "comco", "configs", "default.yaml"))
    ap.add_argument("--dataset_mode", type=str, choices=["table", "kg"], default="table")
    ap.add_argument("--kg_train", type=str, default="", help="KG triples TSV file for KG mode")
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    cache_dir = cfg.get("cache_dir", ".cache/comco")
    registry = ToolRegistry(cache_dir)

    catalog = DatasetCatalog.from_args(dataset_root=args.dataset_root, image_root=args.image_root)
    image_ids, image_paths, text_ids, text_descs, text_names = catalog.load()

    state = BlackboardState(
        image_ids=image_ids,
        text_ids=text_ids,
        image_paths=image_paths,
        text_descs=text_descs,
        text_names=text_names,
        dataset_mode=str(args.dataset_mode),
        dataset_name=str(args.dataset_name or os.path.basename(os.path.normpath(args.dataset_root))),
    )

    fusion_cfg = cfg.get("fusion", {})
    state.weights = {
        "anch": float(fusion_cfg.get("w_anchor", 1.0)),
        "sem": float(fusion_cfg.get("w_semantic", 0.0)),
        "attr": float(fusion_cfg.get("w_attribute", 0.0)),
        "mllm": float(fusion_cfg.get("w_mllm", 0.0)),
    }

    ctx = ToolContext(run_id=sha1_text(f"{state.dataset_name}:{sha1_file(args.config)}"))

    # CLIP tools (anchor signal)
    clip_cfg = ClipAnchorConfig(model_name=cfg.get("clip_model", "ViT-L/14"))
    provider = ClipModelProvider(clip_cfg)
    anchor_text_tool = ClipAnchorTextEmbedTool(provider)
    anchor_img_tool = ClipAnchorImageEmbedTool(provider)

    # Qwen tools
    qwen_cfg_dict = cfg.get("qwen25vl", {})
    qwen_cfg = OllamaQwenVLConfig(
        model=str(qwen_cfg_dict.get("model", "qwen2.5vl:7b")),
        host=str(qwen_cfg_dict.get("host", "127.0.0.1")),
        temperature=float(qwen_cfg_dict.get("temperature", 0.01)),
    )

    prompts = cfg.get("prompts", {})

    caption_tool = Qwen2_5VLCaptionTool(
        QwenCaptionConfig(qwen=qwen_cfg, prompt_template=str(prompts.get("caption", "")))
    )

    # Abstract tools
    abstract_cfg = AbstractConfig(
        qwen=qwen_cfg,
        kg_prompt_template=str(prompts.get("kg_abstract", "")) or None,
        table_prompt_template=str(prompts.get("table_abstract", "")) or None,
    )
    kg_abs_tool = Qwen2_5VLKGAbstractTool(abstract_cfg)
    table_abs_tool = Qwen2_5VLTableAbstractTool(abstract_cfg)

    # Legacy summary tool (fallback)
    summary_tool = Qwen2_5VLSummaryTool(
        QwenSummaryConfig(qwen=qwen_cfg, prompt_prefix=str(prompts.get("summary_prefix", "")))
    )

    visual_attr_tool = Qwen2_5VLVisualAttrTool(
        QwenVisualAttrConfig(qwen=qwen_cfg, prompt_template=str(prompts.get("visual_attrs", "")))
    )
    struct_attr_tool = Qwen2_5VLStructAttrTool(
        QwenStructAttrConfig(qwen=qwen_cfg, prompt_prefix=str(prompts.get("struct_attrs_prefix", "")))
    )

    # Sentence encoder
    sent_cfg_dict = cfg.get("sentence_encoder", {})
    sent_tool = SentenceEncoderTool(
        SentenceEncoderConfig(
            model_name=str(sent_cfg_dict.get("model_name", "sentence-transformers/all-MiniLM-L6-v2")),
            device=str(sent_cfg_dict.get("device", "cpu")),
            batch_size=int(sent_cfg_dict.get("batch_size", 64)),
            normalize=bool(sent_cfg_dict.get("normalize", True)),
        )
    )

    # KG structured text tool (optional)
    kg_struct_tool = None
    if state.dataset_mode == "kg":
        if not args.kg_train:
            raise ValueError("--kg_train is required when --dataset_mode=kg")
        kg_params = cfg.get("kg", {})
        params = KGSampleParams(
            max_hops=int(kg_params.get("max_hops", 1)),
            max_edges=int(kg_params.get("max_edges", 5)),
            max_nodes=int(kg_params.get("max_nodes", 5)),
        )
        graph = KGGraphIndex(args.kg_train, os.path.join(args.dataset_root, "entities_full.txt"))
        kg_struct_tool = KGStructuredTextTool(graph, KGStructuredTextConfig(params=params))

    # Agents
    match = MatchAgent(
        cfg=MatchAgentConfig(topk_candidates=int(cfg.get("candidates", {}).get("topk", 100))),
        registry=registry,
        ctx=ctx,
        text_tool=anchor_text_tool,
        image_tool=anchor_img_tool,
        clip_namespace=f"clip_{clip_cfg.model_name.replace('/','')}",
        sentence_tool=sent_tool,
        caption_tool=caption_tool,
        summary_tool=summary_tool,
        kg_struct_tool=kg_struct_tool,
        kg_abstract_tool=kg_abs_tool,
        table_abstract_tool=table_abs_tool,
        visual_attr_tool=visual_attr_tool,
        struct_attr_tool=struct_attr_tool,
    )

    control = ControlAgent(ControlAgentConfig(), registry=registry)

    shortlist_k = int(cfg.get("candidates", {}).get("shortlist_mllm", 10))
    rerank_tool = OllamaRerankTool(
        OllamaMLLMConfig(model=qwen_cfg.model, host=qwen_cfg.host, temperature=qwen_cfg.temperature)
    )
    rerank = RerankAgent(RerankAgentConfig(shortlist_k=shortlist_k), registry=registry, ctx=ctx, rerank_tool=rerank_tool)

    rounds = int(cfg.get("rounds", 3))
    coordinator = Coordinator(CoordinatorConfig(rounds=rounds), match=match, control=control, rerank=rerank)
    coordinator.run(state)

    for iid in state.image_ids[:5]:
        top = state.topk(iid, key="reg", k=5)
        print(f"\nImage {iid} top-5:")
        for tid, score in top:
            print(f"  {tid}\t{score:.4f}")


if __name__ == "__main__":
    main()
