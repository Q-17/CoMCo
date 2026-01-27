from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn.functional as F

try:
    import clip  # type: ignore
    from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer  # type: ignore
except Exception as e:  # pragma: no cover
    clip = None
    _Tokenizer = None

from .base import Tool, ToolContext


@dataclass
class ClipAnchorConfig:
    model_name: str = "ViT-L/14"
    token_budget: int = 77
    pooling: str = "mean"  # mean|max
    batch_size_img: int = 64
    batch_size_txt: int = 256


class ClipModelProvider:
    def __init__(self, cfg: ClipAnchorConfig):
        self.cfg = cfg
        self._model = None
        self._preprocess = None
        self._tokenizer = None

    def load(self, device: str = "cuda"):
        if clip is None:
            raise ImportError(
                "OpenAI CLIP is not available. Install with: pip install git+https://github.com/openai/CLIP.git"
            )
        if self._model is None:
            dev = torch.device(device)
            model, preprocess = clip.load(self.cfg.model_name, device=dev, jit=False)
            model.eval()
            self._model = model
            self._preprocess = preprocess
            self._tokenizer = _Tokenizer() if _Tokenizer is not None else None
        return self._model, self._preprocess, self._tokenizer


def count_bpe_tokens(tokenizer: _Tokenizer, text: str) -> int:
    return len(tokenizer.encode(text or ""))


def chunk_text_by_token_budget(tokenizer: _Tokenizer, text: str, max_ctx_len: int = 77) -> List[str]:
    content_budget = max_ctx_len - 2
    text = (text or "").strip()
    if not text:
        return [""]

    words = text.split()
    chunks: List[str] = []
    cur: List[str] = []

    def flush():
        if cur:
            chunks.append(" ".join(cur))

    for w in words:
        if not cur:
            # start new chunk
            if count_bpe_tokens(tokenizer, w) <= content_budget:
                cur = [w]
                continue
            # hard split a single long token
            hard = w
            while count_bpe_tokens(tokenizer, hard) > content_budget and len(hard) > 1:
                hard = hard[: max(1, len(hard) // 2)]
            chunks.append(hard)
            rest = w[len(hard) :].strip()
            if rest:
                cur = [rest]
            continue

        tentative = " ".join(cur + [w])
        if count_bpe_tokens(tokenizer, tentative) <= content_budget:
            cur.append(w)
        else:
            flush()
            cur = []
            # new chunk starts with w
            if count_bpe_tokens(tokenizer, w) <= content_budget:
                cur = [w]
            else:
                hard = w
                while count_bpe_tokens(tokenizer, hard) > content_budget and len(hard) > 1:
                    hard = hard[: max(1, len(hard) // 2)]
                chunks.append(hard)
                rest = w[len(hard) :].strip()
                if rest:
                    cur = [rest]

    if cur:
        chunks.append(" ".join(cur))
    return chunks or [""]


def encode_texts_with_chunking(
    model,
    tokenizer: _Tokenizer,
    texts: List[str],
    device: str,
    batch_size: int = 256,
    pooling: str = "mean",
    max_ctx_len: int = 77,
) -> np.ndarray:
    # 1) chunk
    all_chunks: List[str] = []
    owner: List[int] = []
    for i, t in enumerate(texts):
        pieces = chunk_text_by_token_budget(tokenizer, str(t), max_ctx_len=max_ctx_len)
        for p in pieces:
            all_chunks.append(p)
            owner.append(i)

    # 2) encode
    chunk_emb: List[torch.Tensor] = []
    dev = torch.device(device)
    with torch.no_grad():
        for i in tqdm(range(0, len(all_chunks), batch_size), desc="CLIP encode text chunks"):
            batch = all_chunks[i : i + batch_size]
            tokens = clip.tokenize(batch, truncate=False).to(dev)
            feats = model.encode_text(tokens)
            feats = F.normalize(feats.float(), dim=-1)
            chunk_emb.append(feats.cpu())
    emb = torch.cat(chunk_emb, dim=0)  # [Nc, D]

    # 3) aggregate
    n = len(texts)
    d = emb.shape[1]
    out = torch.zeros((n, d), dtype=torch.float32)
    buckets: Dict[int, List[int]] = {}
    for ci, oi in enumerate(owner):
        buckets.setdefault(oi, []).append(ci)

    for oi, idxs in buckets.items():
        vecs = emb[idxs]
        if pooling == "max":
            out[oi] = torch.max(vecs, dim=0).values
        else:
            out[oi] = torch.mean(vecs, dim=0)
    out = F.normalize(out, dim=-1)
    return out.numpy().astype(np.float32)


class ClipAnchorTextEmbedTool(Tool[Tuple[str, str], np.ndarray]):
    """Compute a single structured entity CLIP text embedding.

    Input: (text_id, text_desc)
    Output: embedding (D,)
    """

    name = "clip_anchor_text"

    def __init__(self, provider: ClipModelProvider, cfg: ClipAnchorConfig):
        self.provider = provider
        self.cfg = cfg

    def run(self, ctx: ToolContext, x: Tuple[str, str]) -> np.ndarray:
        _tid, desc = x
        model, _preprocess, tokenizer = self.provider.load(ctx.device)
        assert tokenizer is not None
        vec = encode_texts_with_chunking(
            model,
            tokenizer,
            [desc],
            device=ctx.device,
            batch_size=1,
            pooling=self.cfg.pooling,
            max_ctx_len=self.cfg.token_budget,
        )[0]
        return vec


class ClipAnchorImageEmbedTool(Tool[Tuple[str, str], np.ndarray]):
    """Compute a single image CLIP embedding.

    Input: (image_id, image_path)
    Output: embedding (D,)
    """

    name = "clip_anchor_image"

    def __init__(self, provider: ClipModelProvider, cfg: ClipAnchorConfig):
        self.provider = provider
        self.cfg = cfg

    def run(self, ctx: ToolContext, x: Tuple[str, str]) -> np.ndarray:
        _iid, image_path = x
        model, preprocess, _tokenizer = self.provider.load(ctx.device)
        dev = torch.device(ctx.device)
        with torch.no_grad():
            try:
                im = Image.open(image_path).convert("RGB")
            except Exception:
                im = Image.new("RGB", (224, 224))
            t = preprocess(im).unsqueeze(0).to(dev)
            feats = model.encode_image(t)
            feats = F.normalize(feats.float(), dim=-1)
        return feats.squeeze(0).cpu().numpy().astype(np.float32)


def build_anchor_matrices(
    provider: ClipModelProvider,
    cfg: ClipAnchorConfig,
    image_ids: List[str],
    image_paths: Dict[str, str],
    text_ids: List[str],
    text_descs: Dict[str, str],
    device: str,
) -> Tuple[np.ndarray, np.ndarray]:
    if clip is None:
        raise ImportError("OpenAI CLIP not available")

    model, preprocess, tokenizer = provider.load(device)
    assert tokenizer is not None

    # text matrix
    texts = [text_descs[tid] for tid in text_ids]
    txt_emb = encode_texts_with_chunking(
        model,
        tokenizer,
        texts,
        device=device,
        batch_size=cfg.batch_size_txt,
        pooling=cfg.pooling,
        max_ctx_len=cfg.token_budget,
    )

    # image matrix
    dev = torch.device(device)
    img_emb_list: List[torch.Tensor] = []
    with torch.no_grad():
        for i in tqdm(range(0, len(image_ids), cfg.batch_size_img), desc="CLIP encode images"):
            batch_ids = image_ids[i : i + cfg.batch_size_img]
            batch = []
            for iid in batch_ids:
                p = image_paths[iid]
                try:
                    im = Image.open(p).convert("RGB")
                    batch.append(preprocess(im))
                    im.close()
                except Exception:
                    batch.append(preprocess(Image.new("RGB", (224, 224))))
            imgs = torch.stack(batch, dim=0).to(dev)
            feats = model.encode_image(imgs)
            feats = F.normalize(feats.float(), dim=-1)
            img_emb_list.append(feats.cpu())
    img_emb = torch.cat(img_emb_list, dim=0).numpy().astype(np.float32)

    return img_emb, txt_emb
