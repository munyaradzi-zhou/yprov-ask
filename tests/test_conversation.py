"""Conversations: ask remembers earlier turns, and each turn's record links to the one before."""

from __future__ import annotations

import json
from pathlib import Path

import anyio
from langchain_core.messages import AIMessage, HumanMessage
from mcp import Client

from yprov_ask.agent import ProvenanceAgent
from yprov_ask.conversation import Conversations, Turn
from yprov_ask.server import Settings, build_server
from yprov_ask.store import ProvStore

from .conftest import SAMPLES, ScriptedModel, call
from .test_server import _payload


def test_memory_keeps_the_last_turns_and_forgets_idle_ones():
    now = [0.0]
    memory = Conversations(max_turns=2, idle_seconds=60, clock=lambda: now[0])
    c = memory.resume(None)
    for i in range(3):
        assert memory.add(c, Turn(f"q{i}", f"a{i}", f"s{i}")) == i + 1
    assert [t.question for t in c.turns] == ["q1", "q2"]
    assert memory.resume(c.id) is c

    now[0] = 61.0
    fresh = memory.resume(c.id)
    assert fresh.id != c.id and not fresh.turns
    assert memory.resume("not-an-id").id not in (c.id, fresh.id)


def test_a_follow_up_sees_the_earlier_turn_and_links_to_it(tmp_path):
    store = ProvStore()
    store.ingest_path(SAMPLES / "agents" / "faults")
    settings = Settings()
    settings.records = tmp_path
    model = ScriptedModel(
        replies=[
            call("list_decisions", verdict="ESCALATE"),
            AIMessage("Three decisions escalated."),
            call("list_decisions", verdict="ESCALATE", i=1),
            AIMessage("They were all escalated by the storage agent."),
            call("decline", reason="not about the records", i=2),
            call("decline", reason="a greeting", i=3),
            call("list_documents", i=4),
            AIMessage("Nine documents are in the faults collection."),
        ],
        seen=[],
    )
    server = build_server(
        store, settings, agent_factory=lambda: ProvenanceAgent(store, model, records_dir=tmp_path)
    )

    async def main():
        async with Client(server) as client:
            first = _payload(await client.call_tool("ask", {"question": "Which escalated?"}))
            cid = first["conversation_id"]
            second = _payload(
                await client.call_tool("ask", {"question": "By whom?", "conversation_id": cid})
            )
            pizza = _payload(
                await client.call_tool(
                    "ask", {"question": "Best pizza in Trento?", "conversation_id": cid}
                )
            )
            lost = _payload(
                await client.call_tool("ask", {"question": "Hello?", "conversation_id": "gone"})
            )
            fourth = _payload(
                await client.call_tool(
                    "ask", {"question": "And the documents?", "conversation_id": cid}
                )
            )
            return first, second, pizza, lost, fourth

    first, second, pizza, lost, fourth = anyio.run(main)

    assert first["turn"] == 1 and not first["conversation_restarted"]
    assert second["conversation_id"] == first["conversation_id"] and second["turn"] == 2
    assert lost["turn"] == 1 and lost["conversation_restarted"]
    assert lost["conversation_id"] not in (first["conversation_id"], "gone")

    # The follow-up's model call saw the first question and answer, quoted in its one user
    # message, and not the first turn's tool results.
    follow_up = model.seen[2][1:]
    assert [type(m).__name__ for m in follow_up] == ["HumanMessage"]
    assert "Q: Which escalated?\nA: Three decisions escalated." in follow_up[0].content
    assert follow_up[0].content.endswith("Current question: By whom?")

    # The declined turn is numbered, but never shown to the model again.
    assert pizza["verdict"] == "OUT_OF_SCOPE" and pizza["turn"] == 3
    assert fourth["turn"] == 4
    shown = next(m.content for m in model.seen[6] if isinstance(m, HumanMessage))
    assert "By whom?" in shown and "pizza" not in shown

    # The restarted conversation starts empty: only its own question.
    assert [m.content for m in model.seen[5] if isinstance(m, HumanMessage)] == ["Hello?"]

    doc = json.loads(Path(second["record"]).read_text(encoding="utf-8"))
    before = Path(first["record"]).name.removeprefix("qa_").removesuffix("_prov.json")
    receive = next(k for k in doc["activity"] if k.startswith("yp:ReceiveTaskActivity"))
    assert doc["activity"][receive]["yp:turn"] == 2
    assert doc["activity"][receive]["yp:conversation"] == first["conversation_id"]
    assert {"prov:informed": receive, "prov:informant": f"yp:DecisionActivity_{before}_91"} in (
        doc["wasInformedBy"].values()
    )
    assert {"prov:activity": receive, "prov:entity": f"yp:ResponseEntity_{before}_answer"} in (
        doc["used"].values()
    )
    [member] = doc["hadMember"].values()
    assert member["prov:collection"] == f"yp:conversation_{first['conversation_id']}"
