from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field

from yprov_ask.store import ProvStore

SAMPLES = Path(__file__).resolve().parent.parent / "samples"


@pytest.fixture(scope="session")
def store() -> ProvStore:
    s = ProvStore(":memory:")
    s.ingest_path(SAMPLES)
    return s


class ScriptedModel(BaseChatModel):
    """A chat model that replies from a script, and remembers what it was shown.

    Each reply is an AIMessage; one carrying ``tool_calls`` drives the agent's tool node
    exactly as a real model's would.
    """

    replies: list[AIMessage]
    seen: list[list[BaseMessage]]
    #: The names of the tools of each bind_tools call, and their argument schemas.
    bound: list[dict[str, dict[str, Any]]] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> ScriptedModel:
        self.bound.append({t.name: t.args for t in tools})
        return self

    def _generate(self, messages: list[BaseMessage], stop: Any = None, **kwargs: Any) -> ChatResult:
        self.seen.append(list(messages))
        return ChatResult(generations=[ChatGeneration(message=self.replies.pop(0))])


def call(tool: str, i: int = 0, **args: Any) -> AIMessage:
    return AIMessage(
        "", tool_calls=[{"name": tool, "args": args, "id": f"c{i}", "type": "tool_call"}]
    )
