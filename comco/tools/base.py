from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, Optional, TypeVar

InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")


@dataclass
class ToolContext:
    """Lightweight shared context passed into tools."""

    # A stable identifier for the current experiment run. Tools may use it for
    # logging, but caching is handled by ToolRegistry keys.
    run_id: str = ""
    device: str = "cuda"


class Tool(Generic[InputT, OutputT]):
    name: str

    def run(self, ctx: ToolContext, x: InputT) -> OutputT:
        raise NotImplementedError
