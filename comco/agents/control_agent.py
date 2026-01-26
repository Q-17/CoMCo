from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Set, Tuple

import numpy as np

from comco.core.state import BlackboardState


def jaccard(a: Set[str], b: Set[str]) -> float:
    if not a and not b:
        return 1.0
    inter = len(a & b)
    union = len(a | b)
    return inter / (union + 1e-12)


@dataclass
class ControlAgentConfig:
    group_jaccard_threshold: float = 0.2
    group_topk_for_consensus: int = 10
    lambda_consensus: float = 0.2
    hub_penalty_mu: float = 0.15
    rerank_budget_ratio: float = 0.3


class ControlAgent:
    """Global consistency controller.
    """

    def __init__(self, cfg: ControlAgentConfig):
        self.cfg = cfg

    def _discover_groups(self, state: BlackboardState) -> None:
        image_ids = state.image_ids
        cand_sets: Dict[str, Set[str]] = {iid: set(state.candidates.get(iid, [])) for iid in image_ids}

        # build graph edges by jaccard threshold
        adj: Dict[str, List[str]] = {iid: [] for iid in image_ids}
        thr = self.cfg.group_jaccard_threshold
        for i in range(len(image_ids)):
            for j in range(i + 1, len(image_ids)):
                v1, v2 = image_ids[i], image_ids[j]
                sim = jaccard(cand_sets[v1], cand_sets[v2])
                if sim >= thr:
                    adj[v1].append(v2)
                    adj[v2].append(v1)

        # connected components
        visited: Set[str] = set()
        groups: Dict[str, List[str]] = {}
        image2group: Dict[str, str] = {}
        gid = 0
        for iid in image_ids:
            if iid in visited:
                continue
            stack = [iid]
            comp: List[str] = []
            visited.add(iid)
            while stack:
                x = stack.pop()
                comp.append(x)
                for y in adj[x]:
                    if y not in visited:
                        visited.add(y)
                        stack.append(y)
            group_id = f"g{gid}"
            gid += 1
            groups[group_id] = comp
            for x in comp:
                image2group[x] = group_id

        state.groups = groups
        state.image2group = image2group

    def _compute_group_consensus(self, state: BlackboardState) -> Dict[Tuple[str, str], float]:
        """Return consensus score for each (group_id, text_id)."""
        k = self.cfg.group_topk_for_consensus
        cons: Dict[Tuple[str, str], float] = {}
        for gid, imgs in state.groups.items():
            if not imgs:
                continue
            counts: Dict[str, int] = {}
            for iid in imgs:
                topk = state.topk(iid, key="evid", k=min(k, len(state.candidates.get(iid, []))))
                for tid, _ in topk:
                    counts[tid] = counts.get(tid, 0) + 1
            for tid, c in counts.items():
                cons[(gid, tid)] = c / float(len(imgs))
        return cons

    def _compute_hubness(self, state: BlackboardState) -> None:
        """Compute hubness cf(s) based on candidate frequency."""
        cf: Dict[str, int] = {}
        for iid in state.image_ids:
            for tid in state.candidates.get(iid, []):
                cf[tid] = cf.get(tid, 0) + 1
        if not cf:
            state.hubness = {}
            return
        df_max = max(cf.values())
        log_max = np.log1p(df_max)
        hub = {tid: float(np.log1p(v) / (log_max + 1e-12)) for tid, v in cf.items()}
        state.hubness = hub

    def _calibrate_and_regulate(self, state: BlackboardState) -> None:
        lam = self.cfg.lambda_consensus
        mu = self.cfg.hub_penalty_mu
        cons = self._compute_group_consensus(state)

        for iid in state.image_ids:
            gid = state.image2group.get(iid)
            for tid in state.candidates.get(iid, []):
                state.ensure_edge(iid, tid)
                evid = state.scores[(iid, tid)].get("evid", 0.0)
                c = cons.get((gid, tid), 0.0) if gid is not None else 0.0
                cal = evid + lam * c
                hub = state.hubness.get(tid, 0.0)
                reg = cal - mu * hub

                state.scores[(iid, tid)]["cal"] = float(cal)
                state.scores[(iid, tid)]["reg"] = float(reg)

    def _schedule_hard_cases(self, state: BlackboardState) -> None:
        margins: List[Tuple[str, float]] = []
        for iid in state.image_ids:
            pairs = state.topk(iid, key="reg", k=2)
            if len(pairs) < 2:
                margins.append((iid, 0.0))
                continue
            m = pairs[0][1] - pairs[1][1]
            margins.append((iid, float(m)))
        if margins:
            state.avg_margin = float(np.mean([m for _, m in margins]))

        margins.sort(key=lambda x: x[1])  # smallest margin first
        budget = int(max(1, round(len(state.image_ids) * self.cfg.rerank_budget_ratio)))
        state.hard_cases = [iid for iid, _ in margins[:budget]]

    def _update_weights(self, state: BlackboardState) -> None:
        """
        """
        avg_m = state.avg_margin
        w = dict(state.weights)
        if avg_m < 0.05:
            w["mllm"] = min(0.5, w.get("mllm", 0.0) + 0.05)
            # keep anchor dominant for now
            w["anch"] = max(0.5, w.get("anch", 1.0) - 0.05)
        # renorm
        s = sum(max(0.0, v) for v in w.values()) + 1e-12
        for k in w:
            w[k] = float(max(0.0, w[k]) / s)
        state.weights = w

    def _stop_criterion(self, state: BlackboardState) -> None:
        # Simple: stop if very stable margins and no change in hard cases
        if state.round_idx >= 1 and state.avg_margin > 0.2:
            state.stop = True

    def run(self, state: BlackboardState) -> None:
        self._discover_groups(state)
        self._compute_hubness(state)
        self._calibrate_and_regulate(state)
        self._schedule_hard_cases(state)
        self._update_weights(state)
        self._stop_criterion(state)
