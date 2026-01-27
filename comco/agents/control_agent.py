from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Set, Tuple, Optional

import numpy as np

from comco.core.state import BlackboardState


def jaccard(a: Set[str], b: Set[str]) -> float:
    if not a and not b:
        return 1.0
    inter = len(a & b)
    union = len(a | b)
    return float(inter / (union + 1e-12))


def _renorm(weights: Dict[str, float]) -> Dict[str, float]:
    s = float(sum(max(0.0, v) for v in weights.values()) + 1e-12)
    return {k: float(max(0.0, v) / s) for k, v in weights.items()}


def _clip(weights: Dict[str, float], lo: Dict[str, float], hi: Dict[str, float]) -> Dict[str, float]:
    out = {}
    for k, v in weights.items():
        v0 = float(v)
        v0 = float(max(lo.get(k, -1e9), v0))
        v0 = float(min(hi.get(k, 1e9), v0))
        out[k] = v0
    return out


@dataclass
class ControlAgentConfig:
    group_jaccard_threshold: float = 0.2
    group_topk_for_consensus: int = 10
    lambda_consensus: float = 0.2
    hub_penalty_mu: float = 0.15
    rerank_budget_ratio: float = 0.3

    coherence_low: float = 0.4
    coherence_high: float = 0.7
    ambiguity_low_margin: float = 0.15

    step_coherence: float = 0.15
    step_ambiguity: float = 0.20

    weight_min: Dict[str, float] = None
    weight_max: Dict[str, float] = None

    def __post_init__(self) -> None:
        if self.weight_min is None:
            self.weight_min = {"anch": 0.20, "sem": 0.05, "attr": 0.05, "mllm": 0.00}
        if self.weight_max is None:
            self.weight_max = {"anch": 0.90, "sem": 0.70, "attr": 0.70, "mllm": 0.70}


class ControlAgent:
    def __init__(self, cfg: ControlAgentConfig):
        self.cfg = cfg

    def _discover_groups(self, state: BlackboardState) -> None:
        image_ids = state.image_ids
        cand_sets: Dict[str, Set[str]] = {iid: set(state.candidates.get(iid, [])) for iid in image_ids}

        adj: Dict[str, List[str]] = {iid: [] for iid in image_ids}
        thr = self.cfg.group_jaccard_threshold

        for i in range(len(image_ids)):
            for j in range(i + 1, len(image_ids)):
                a, b = image_ids[i], image_ids[j]
                if jaccard(cand_sets[a], cand_sets[b]) >= thr:
                    adj[a].append(b)
                    adj[b].append(a)

        visited: Set[str] = set()
        groups: Dict[str, List[str]] = {}
        image2group: Dict[str, str] = {}
        gid = 0

        for iid in image_ids:
            if iid in visited:
                continue
            stack = [iid]
            visited.add(iid)
            comp: List[str] = []
            while stack:
                x = stack.pop()
                comp.append(x)
                for y in adj[x]:
                    if y not in visited:
                        visited.add(y)
                        stack.append(y)
            gname = f"g{gid}"
            gid += 1
            groups[gname] = comp
            for x in comp:
                image2group[x] = gname

        state.groups = groups
        state.image2group = image2group

    def _compute_group_consensus(self, state: BlackboardState) -> Dict[Tuple[str, str], float]:
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
            denom = float(len(imgs))
            for tid, c in counts.items():
                cons[(gid, tid)] = float(c / (denom + 1e-12))
        return cons

    def _compute_group_coherence(self, state: BlackboardState, cons: Dict[Tuple[str, str], float]) -> float:
        vals: List[float] = []
        weights: List[float] = []
        for gid, imgs in state.groups.items():
            if not imgs:
                continue
            p = [v for (g, _), v in cons.items() if g == gid and v > 0.0]
            if len(p) <= 1:
                coh = 1.0
            else:
                p = np.asarray(p, dtype=np.float32)
                p = p / (p.sum() + 1e-12)
                h = float(-(p * np.log(p + 1e-12)).sum())
                hmax = float(np.log(len(p) + 1e-12))
                coh = float(1.0 - (h / (hmax + 1e-12)))
            vals.append(coh)
            weights.append(float(len(imgs)))
        if not vals:
            return 1.0
        w = np.asarray(weights, dtype=np.float32)
        v = np.asarray(vals, dtype=np.float32)
        return float((w * v).sum() / (w.sum() + 1e-12))

    def _compute_hubness(self, state: BlackboardState) -> None:
        cf: Dict[str, int] = {}
        for iid in state.image_ids:
            for tid in state.candidates.get(iid, []):
                cf[tid] = cf.get(tid, 0) + 1
        if not cf:
            state.hubness = {}
            return
        df_max = max(cf.values())
        log_max = float(np.log1p(df_max))
        state.hubness = {tid: float(np.log1p(v) / (log_max + 1e-12)) for tid, v in cf.items()}

    def _calibrate_and_regulate(self, state: BlackboardState, cons: Dict[Tuple[str, str], float]) -> None:
        lam = self.cfg.lambda_consensus
        mu = self.cfg.hub_penalty_mu
        for iid in state.image_ids:
            gid = state.image2group.get(iid)
            for tid in state.candidates.get(iid, []):
                state.ensure_edge(iid, tid)
                evid = float(state.scores[(iid, tid)].get("evid", 0.0))
                c = float(cons.get((gid, tid), 0.0)) if gid is not None else 0.0
                cal = evid + lam * c
                hub = float(state.hubness.get(tid, 0.0))
                reg = cal - mu * hub
                state.scores[(iid, tid)]["cal"] = float(cal)
                state.scores[(iid, tid)]["reg"] = float(reg)

    def _schedule_hard_cases(self, state: BlackboardState) -> None:
        margins: List[Tuple[str, float]] = []
        for iid in state.image_ids:
            pairs = state.topk(iid, key="reg", k=2)
            if len(pairs) < 2:
                margins.append((iid, 0.0))
            else:
                margins.append((iid, float(pairs[0][1] - pairs[1][1])))
        state.avg_margin = float(np.mean([m for _, m in margins])) if margins else 0.0
        margins.sort(key=lambda x: x[1])
        budget = int(max(1, round(len(state.image_ids) * self.cfg.rerank_budget_ratio)))
        state.hard_cases = [iid for iid, _ in margins[:budget]]

    def _reweight_by_coherence(self, w: Dict[str, float], coherence: float) -> Dict[str, float]:
        step = self.cfg.step_coherence
        coh = float(np.clip(coherence, 0.0, 1.0))
        if coh < self.cfg.coherence_low:
            t = (self.cfg.coherence_low - coh) / max(1e-6, self.cfg.coherence_low)
            m_anch = 1.0 - step * t
            m_sem = 1.0 + step * t
            m_attr = 1.0 + 0.8 * step * t
        elif coh > self.cfg.coherence_high:
            t = (coh - self.cfg.coherence_high) / max(1e-6, 1.0 - self.cfg.coherence_high)
            m_anch = 1.0 + 0.5 * step * t
            m_sem = 1.0 - 0.4 * step * t
            m_attr = 1.0 - 0.4 * step * t
        else:
            return w
        out = dict(w)
        out["anch"] = float(out.get("anch", 0.0) * m_anch)
        out["sem"] = float(out.get("sem", 0.0) * m_sem)
        out["attr"] = float(out.get("attr", 0.0) * m_attr)
        return _renorm(_clip(out, self.cfg.weight_min, self.cfg.weight_max))

    def _reweight_hard_cases_by_ambiguity(self, w: Dict[str, float], avg_margin: float) -> Dict[str, float]:
        if avg_margin >= self.cfg.ambiguity_low_margin:
            return w
        step = self.cfg.step_ambiguity
        out = dict(w)
        out["mllm"] = float(out.get("mllm", 0.0) + step)
        out["anch"] = float(max(0.0, out.get("anch", 0.0) - 0.6 * step))
        out["sem"] = float(max(0.0, out.get("sem", 0.0) - 0.2 * step))
        out["attr"] = float(max(0.0, out.get("attr", 0.0) - 0.2 * step))
        return _renorm(_clip(out, self.cfg.weight_min, self.cfg.weight_max))

    def _stop_criterion(self, state: BlackboardState) -> None:
        coh = float(getattr(state, "coherence", 0.0))
        if state.round_idx >= 1 and coh > 0.85 and state.avg_margin > 0.20:
            state.stop = True

    def run(self, state: BlackboardState) -> None:
        self._discover_groups(state)
        self._compute_hubness(state)

        cons = self._compute_group_consensus(state)
        state.coherence = self._compute_group_coherence(state, cons)

        self._calibrate_and_regulate(state, cons)
        self._schedule_hard_cases(state)

        w_base = dict(state.weights)
        w_base = _renorm(w_base)
        w_base = self._reweight_by_coherence(w_base, state.coherence)
        state.weights = w_base

        state.weights_hard = self._reweight_hard_cases_by_ambiguity(dict(w_base), state.avg_margin)

        self._stop_criterion(state)
