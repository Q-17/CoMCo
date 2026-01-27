from __future__ import annotations

from dataclasses import dataclass

from comco.core.state import BlackboardState
from comco.agents.match_agent import MatchAgent
from comco.agents.control_agent import ControlAgent
from comco.agents.rerank_agent import RerankAgent


@dataclass
class CoordinatorConfig:
    rounds: int = 3


class Coordinator:
    def __init__(self, cfg: CoordinatorConfig, match: MatchAgent, control: ControlAgent, rerank: RerankAgent):
        self.cfg = cfg
        self.match = match
        self.control = control
        self.rerank = rerank

    def run(self, state: BlackboardState) -> BlackboardState:
        for r in range(self.cfg.rounds):
            state.round_idx = r

            # 1) evidence building
            self.match.run(state)

            # 2) global control 
            self.control.run(state)

            # 3) selective MLLM rerank 
            self.rerank.run(state)

            # 4) re-fuse evidence 
            self.match.run(state)

            if state.stop:
                break

        return state
