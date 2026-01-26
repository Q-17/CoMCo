from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np

from comco.core.registry import ToolRegistry, sha1_text
from comco.core.state import BlackboardState
from comco.tools.base import ToolContext
from comco.tools.anchor_clip import ClipAnchorImageEmbedTool, ClipAnchorTextEmbedTool
from comco.tools.caption_qwen25vl import Qwen2_5VLCaptionTool
from comco.tools.summary_qwen25vl import Qwen2_5VLSummaryTool
from comco.tools.kg_structured_text import KGStructuredTextTool
from comco.tools.abstract_qwen25 import Qwen2_5VLKGAbstractTool, Qwen2_5VLTableAbstractTool
from comco.tools.attr_qwen25vl import Qwen2_5VLVisualAttrTool, Qwen2_5VLStructAttrTool
from comco.tools.sentence_encoder import SentenceEncoderTool


def _l2norm(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-12)


def softmax(x: np.ndarray) -> np.ndarray:
    # stable softmax over a 1D vector
    if x.size == 0:
        return x
    x = x - np.max(x)
    e = np.exp(x)
    return e / (np.sum(e) + 1e-12)


@dataclass
class MatchAgentConfig:
    topk_candidates: int = 100


class MatchAgent:
    """Match agent: build candidate sets and evidence signals, then fuse into S_evid.
    This agent calls tools only via ToolRegistry to enable persistent disk caching.
    """

    def __init__(
        self,
        cfg: MatchAgentConfig,
        registry: ToolRegistry,
        ctx: ToolContext,
        text_tool: ClipAnchorTextEmbedTool,
        image_tool: ClipAnchorImageEmbedTool,
        clip_namespace: str,
        sentence_tool: Optional[SentenceEncoderTool] = None,
        caption_tool: Optional[Qwen2_5VLCaptionTool] = None,
        summary_tool: Optional[Qwen2_5VLSummaryTool] = None,
        kg_struct_tool: Optional[KGStructuredTextTool] = None,
        kg_abstract_tool: Optional[Qwen2_5VLKGAbstractTool] = None,
        table_abstract_tool: Optional[Qwen2_5VLTableAbstractTool] = None,
        visual_attr_tool: Optional[Qwen2_5VLVisualAttrTool] = None,
        struct_attr_tool: Optional[Qwen2_5VLStructAttrTool] = None,
        nlp_namespace: str = "nlp",
        qwen_namespace: str = "qwen25vl",
    ):
        self.cfg = cfg
        self.registry = registry
        self.ctx = ctx

        self.text_tool = text_tool
        self.image_tool = image_tool
        self.clip_namespace = clip_namespace

        self.sentence_tool = sentence_tool
        self.caption_tool = caption_tool
        self.summary_tool = summary_tool
        self.kg_struct_tool = kg_struct_tool
        self.kg_abstract_tool = kg_abstract_tool
        self.table_abstract_tool = table_abstract_tool
        self.visual_attr_tool = visual_attr_tool
        self.struct_attr_tool = struct_attr_tool

        self.nlp_namespace = nlp_namespace
        self.qwen_namespace = qwen_namespace

    # -------------------------
    # Anchor: entity-level emb
    # -------------------------
    def _get_text_matrix(self, state: BlackboardState) -> Tuple[List[str], np.ndarray]:
        """Ensure all text embeddings exist; return ordered ids and matrix [Nt,D]."""
        text_ids = state.text_ids

        def _compute_one(tid: str) -> np.ndarray:
            desc = state.text_descs.get(tid, "")
            return self.text_tool.run(self.ctx, (tid, desc))

        for tid in text_ids:
            if tid not in state.anchor_text_emb:
                key = sha1_text(f"{tid}:{state.text_descs.get(tid,'')}")
                vec = self.registry.get_or_run_numpy(
                    namespace=f"{self.clip_namespace}/anchor_text",
                    key=key,
                    fn=lambda tid=tid: _compute_one(tid),
                )
                state.anchor_text_emb[tid] = vec

        mat = np.stack([state.anchor_text_emb[tid] for tid in text_ids], axis=0)
        return text_ids, _l2norm(mat)

    def _get_image_vectors(self, state: BlackboardState, image_ids: List[str]) -> np.ndarray:
        """Ensure embeddings for given images; return [B,D] matrix."""

        def _compute_one(iid: str) -> np.ndarray:
            return self.image_tool.run(self.ctx, (iid, state.image_paths[iid]))

        vecs: List[np.ndarray] = []
        for iid in image_ids:
            if iid not in state.anchor_image_emb:
                key = sha1_text(f"{iid}:{state.image_paths.get(iid,'')}")
                vec = self.registry.get_or_run_numpy(
                    namespace=f"{self.clip_namespace}/anchor_image",
                    key=key,
                    fn=lambda iid=iid: _compute_one(iid),
                )
                state.anchor_image_emb[iid] = vec
            vecs.append(state.anchor_image_emb[iid])
        return _l2norm(np.stack(vecs, axis=0))

    def _retrieve_candidates(self, state: BlackboardState) -> None:
        """Anchor-only coarse retrieval: top-K for each image."""
        text_ids, T = self._get_text_matrix(state)
        I = self._get_image_vectors(state, state.image_ids)
        sim = I @ T.T

        k = min(self.cfg.topk_candidates, sim.shape[1])
        idx_part = np.argpartition(-sim, k - 1, axis=1)[:, :k]
        row = np.arange(sim.shape[0])[:, None]
        part_scores = sim[row, idx_part]
        order = np.argsort(-part_scores, axis=1)
        top_idx = idx_part[row, order]

        for i, iid in enumerate(state.image_ids):
            cands = [text_ids[j] for j in top_idx[i].tolist()]
            state.candidates[iid] = cands
            scores_i = part_scores[i][order[i]].tolist()
            for tid, s in zip(cands, scores_i):
                state.ensure_edge(iid, tid)
                state.signals[(iid, tid)]["anch"] = float(s)

    # -------------------------
    # Semantic signal
    # -------------------------
    def _ensure_caption(self, state: BlackboardState, iid: str) -> str:
        if iid in state.captions:
            return state.captions[iid]
        if self.caption_tool is None:
            state.captions[iid] = ""
            return ""
        key = sha1_text(f"{iid}:{state.image_paths.get(iid,'')}:caption")
        cap_obj = self.registry.get_or_run_json(
            namespace=f"{self.qwen_namespace}/caption",
            key=key,
            fn=lambda: self.caption_tool.run(self.ctx, (iid, state.image_paths[iid])),
        )
        cap = cap_obj.get("caption") if isinstance(cap_obj, dict) else cap_obj
        cap = str(cap or "")
        state.captions[iid] = cap
        return cap

    def _ensure_summary(self, state: BlackboardState, tid: str) -> str:
        if tid in state.summaries:
            return state.summaries[tid]

        # -----------------------------
        # KG mode: entity abstract a_s is generated from a structured serialization r_s
        # -----------------------------
        if (state.dataset_mode or "table") == "kg" and self.kg_struct_tool is not None and self.kg_abstract_tool is not None:
            # 1) structured text r_s
            if tid not in state.serialized_struct:
                # include params + entity id in key; params already embedded in tool instance
                key_r = sha1_text(f"{state.dataset_name}:kg_struct:{tid}")
                r_s = self.registry.get_or_run_json(
                    namespace=f"{self.qwen_namespace}/kg_struct",
                    key=key_r,
                    fn=lambda: self.kg_struct_tool.run(self.ctx, tid),
                )
                state.serialized_struct[tid] = str(r_s or "")
            r_s = state.serialized_struct.get(tid, "")

            # 2) abstract a_s
            key_a = sha1_text(f"{state.dataset_name}:kg_abs:{tid}:{r_s}")
            abs_text = self.registry.get_or_run_json(
                namespace=f"{self.qwen_namespace}/kg_abstract",
                key=key_a,
                fn=lambda: self.kg_abstract_tool.run(self.ctx, (tid, r_s)),
            )
            summ = str(abs_text or "")
            state.summaries[tid] = summ
            return summ

        # -----------------------------
        # Table mode: abstract from (name, desc)
        # -----------------------------
        if self.table_abstract_tool is not None:
            name = state.text_names.get(tid, "")
            desc = state.text_descs.get(tid, "")
            key = sha1_text(f"{state.dataset_name}:tbl_abs:{tid}:{name}:{desc}")
            abs_text = self.registry.get_or_run_json(
                namespace=f"{self.qwen_namespace}/table_abstract",
                key=key,
                fn=lambda: self.table_abstract_tool.run(self.ctx, (tid, state.dataset_name, name, desc)),
            )
            summ = str(abs_text or "")
            state.summaries[tid] = summ
            return summ

        # Fallback: previous desc-only summary tool
        desc = state.text_descs.get(tid, "")
        if self.summary_tool is None:
            state.summaries[tid] = desc
            return desc
        key = sha1_text(f"{tid}:{desc}:summary")
        summ_obj = self.registry.get_or_run_json(
            namespace=f"{self.qwen_namespace}/summary",
            key=key,
            fn=lambda: self.summary_tool.run(self.ctx, (tid, desc)),
        )
        summ = summ_obj.get("summary") if isinstance(summ_obj, dict) else summ_obj
        summ = str(summ or "")
        state.summaries[tid] = summ
        return summ

    def _semantic_score(self, state: BlackboardState, iid: str, tid: str) -> float:
        """Compute semantic score cos(Emb(caption), Emb(summary))."""
        if self.sentence_tool is None:
            return 0.0
        cap = self._ensure_caption(state, iid)
        summ = self._ensure_summary(state, tid)
        if not cap or not summ:
            return 0.0

        cap_key = sha1_text(f"cap:{cap}")
        summ_key = sha1_text(f"summ:{summ}")

        cap_vec = self.registry.get_or_run_numpy(
            namespace=f"{self.nlp_namespace}/sent",
            key=cap_key,
            fn=lambda: self.sentence_tool.run(self.ctx, [cap])[0],
        ).astype(np.float32).reshape(-1)

        summ_vec = self.registry.get_or_run_numpy(
            namespace=f"{self.nlp_namespace}/sent",
            key=summ_key,
            fn=lambda: self.sentence_tool.run(self.ctx, [summ])[0],
        ).astype(np.float32).reshape(-1)

        num = float(np.dot(cap_vec, summ_vec))
        den = float(np.linalg.norm(cap_vec) * np.linalg.norm(summ_vec) + 1e-12)
        return num / den

    # -------------------------
    # Attribute signal
    # -------------------------
    def _ensure_visual_attrs(self, state: BlackboardState, iid: str) -> List[str]:
        if iid in state.visual_attrs:
            return state.visual_attrs[iid]
        if self.visual_attr_tool is None:
            state.visual_attrs[iid] = []
            return []
        key = sha1_text(f"{iid}:{state.image_paths.get(iid,'')}:vis_attrs")
        attrs_obj = self.registry.get_or_run_json(
            namespace=f"{self.qwen_namespace}/vis_attrs",
            key=key,
            fn=lambda: self.visual_attr_tool.run(self.ctx, (iid, state.image_paths[iid])),
        )
        attrs = attrs_obj.get("attrs") if isinstance(attrs_obj, dict) else attrs_obj
        attrs = [str(x).strip() for x in (attrs or []) if str(x).strip()]
        state.visual_attrs[iid] = attrs
        return attrs

    def _ensure_struct_attrs(self, state: BlackboardState, tid: str) -> List[str]:
        if tid in state.struct_attrs:
            return state.struct_attrs[tid]
        if self.struct_attr_tool is None:
            state.struct_attrs[tid] = []
            return []
        desc = state.text_descs.get(tid, "")
        key = sha1_text(f"{tid}:{desc}:struct_attrs")
        attrs_obj = self.registry.get_or_run_json(
            namespace=f"{self.qwen_namespace}/struct_attrs",
            key=key,
            fn=lambda: self.struct_attr_tool.run(self.ctx, (tid, desc)),
        )
        attrs = attrs_obj.get("attrs") if isinstance(attrs_obj, dict) else attrs_obj
        attrs = [str(x).strip() for x in (attrs or []) if str(x).strip()]
        state.struct_attrs[tid] = attrs
        return attrs

    def _attribute_score(self, state: BlackboardState, iid: str, tid: str) -> float:
        """Compute attribute score cos(mean(emb(A(v))), mean(emb(A(s))))."""
        if self.sentence_tool is None:
            return 0.0
        a_v = self._ensure_visual_attrs(state, iid)
        a_s = self._ensure_struct_attrs(state, tid)
        if not a_v or not a_s:
            return 0.0

        key_v = sha1_text("vis:" + "|".join(a_v))
        key_s = sha1_text("str:" + "|".join(a_s))

        gv = self.registry.get_or_run_numpy(
            namespace=f"{self.nlp_namespace}/attr_pool",
            key=key_v,
            fn=lambda: self.sentence_tool.run(self.ctx, a_v).mean(axis=0),
        ).astype(np.float32).reshape(-1)

        gs = self.registry.get_or_run_numpy(
            namespace=f"{self.nlp_namespace}/attr_pool",
            key=key_s,
            fn=lambda: self.sentence_tool.run(self.ctx, a_s).mean(axis=0),
        ).astype(np.float32).reshape(-1)

        num = float(np.dot(gv, gs))
        den = float(np.linalg.norm(gv) * np.linalg.norm(gs) + 1e-12)
        return num / den

    
    # -------------------------
    # Fusion
    # -------------------------
    def _fuse_evidence(self, state: BlackboardState) -> None:
        w = state.weights

        for iid in state.image_ids:
            cands = state.candidates.get(iid, [])
            if not cands:
                continue

            # Ensure sem/attr signals exist when needed.
            for tid in cands:
                state.ensure_edge(iid, tid)
                if w.get("sem", 0.0) > 0.0 and "sem" not in state.signals[(iid, tid)]:
                    state.signals[(iid, tid)]["sem"] = float(self._semantic_score(state, iid, tid))
                if w.get("attr", 0.0) > 0.0 and "attr" not in state.signals[(iid, tid)]:
                    state.signals[(iid, tid)]["attr"] = float(self._attribute_score(state, iid, tid))

            anch_scores = np.array([state.signals[(iid, tid)].get("anch", 0.0) for tid in cands], dtype=np.float32)
            sem_scores = np.array([state.signals[(iid, tid)].get("sem", 0.0) for tid in cands], dtype=np.float32)
            attr_scores = np.array([state.signals[(iid, tid)].get("attr", 0.0) for tid in cands], dtype=np.float32)
            mllm_scores = np.array([state.signals[(iid, tid)].get("mllm", 0.0) for tid in cands], dtype=np.float32)

            anch_norm = softmax(anch_scores)
            sem_norm = softmax(sem_scores)
            attr_norm = softmax(attr_scores)
            mllm_norm = softmax(mllm_scores)

            fused = (
                w.get("anch", 0.0) * anch_norm
                + w.get("sem", 0.0) * sem_norm
                + w.get("attr", 0.0) * attr_norm
                + w.get("mllm", 0.0) * mllm_norm
            )

            for tid, fv in zip(cands, fused.tolist()):
                state.scores[(iid, tid)]["evid"] = float(fv)

    def run(self, state: BlackboardState) -> None:
        if not state.candidates:
            self._retrieve_candidates(state)
        self._fuse_evidence(state)
