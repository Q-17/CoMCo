from __future__ import annotations

from dataclasses import dataclass
from typing import List

from comco.core.registry import ToolRegistry, sha1_text
from comco.core.state import BlackboardState
from comco.tools.base import ToolContext
from comco.tools.mllm_rerank import OllamaRerankTool


@dataclass
class RerankAgentConfig:
    shortlist_k: int = 10
    max_calls_per_round: int = 256


class RerankAgent:
    """Selective MLLM disambiguation.

    For each hard case image v, take top-K candidates (by current reg score),
    call MLLM to output a best-first ranking, then write back a continuous score
    z(v,s) into `signals[(v,s)]["mllm"]`.
    """

    def __init__(
        self,
        cfg: RerankAgentConfig,
        registry: ToolRegistry,
        ctx: ToolContext,
        rerank_tool: OllamaRerankTool,
        namespace: str,
    ):
        self.cfg = cfg
        self.registry = registry
        self.ctx = ctx
        self.rerank_tool = rerank_tool
        self.namespace = namespace

    def run(self, state: BlackboardState) -> None:
        hard = state.hard_cases[:]
        if self.cfg.max_calls_per_round > 0:
            hard = hard[: self.cfg.max_calls_per_round]

        for iid in hard:
            # shortlist from current reg scores
            shortlist = [tid for tid, _ in state.topk(iid, key="reg", k=self.cfg.shortlist_k)]
            if not shortlist:
                continue

            key = sha1_text(f"{iid}:{'|'.join(shortlist)}")

            def _call():
                return self.rerank_tool.run(
                    self.ctx,
                    (iid, state.image_paths[iid], shortlist, state.text_descs),
                )

            ranking = self.registry.get_or_run_json(
                namespace=f"{self.namespace}/mllm_rerank",
                key=key,
                fn=_call,
            )
            if not ranking:
                continue

            # Ensure we have a complete permutation over the shortlist.
            # If the model returns a partial list (or only a best id), we
            # append missing candidates in their original shortlist order.
            ranking = [tid for tid in ranking if tid in shortlist]
            seen = set(ranking)
            if len(ranking) < len(shortlist):
                ranking.extend([tid for tid in shortlist if tid not in seen])

            # map ranking -> z scores in [0,1]
            # simple: z=1 for best, else linearly decays within shortlist
            n = len(shortlist)
            pos = {tid: idx for idx, tid in enumerate(ranking) if tid in shortlist}
            for tid in shortlist:
                state.ensure_edge(iid, tid)
                if tid in pos:
                    r = pos[tid]
                    z = 1.0 - (r / max(1, n - 1))
                else:
                    z = 0.0
                state.signals[(iid, tid)]["mllm"] = float(z)
