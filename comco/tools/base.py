from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, Optional, TypeVar

InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")


@dataclass
class ToolContext:
    run_id: str = ""
    device: str = "cuda"


class Tool(Generic[InputT, OutputT]):
    name: str

    def run(self, ctx: ToolContext, x: InputT) -> OutputT:
        raise NotImplementedError
