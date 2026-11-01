"""The MCP server, through a real MCP client connected in-process."""

from __future__ import annotations

import json

import anyio
from langchain_core.messages import AIMessage
from mcp import Client

from yprov_ask.agent import ProvenanceAgent
from yprov_ask.server import Settings, build_server
from yprov_ask.store import ProvStore

from .conftest import SAMPLES, ScriptedModel, call


def _server(tmp_path, replies=()):
    store = ProvStore()
    settings = Settings()
    settings.roots = [SAMPLES.resolve()]
    settings.records = tmp_path
    model = ScriptedModel(replies=list(replies), seen=[])
    return build_server(
        store, settings, agent_factory=lambda: ProvenanceAgent(store, model, records_dir=tmp_path)
    ), store


def _payload(result):
    return json.loads(result.content[0].text)


def test_tools_ingest_and_query(tmp_path):
    server, _ = _server(tmp_path)

    async def main():
        async with Client(server) as client:
            names = {t.name for t in (await client.list_tools()).tools}
            assert {"ask", "ingest", "store_get", "list_documents", "list_decisions"} <= names
            ask = next(t for t in (await client.list_tools()).tools if t.name == "ask")
            # The MCP context the SDK injects is not an argument a client sees.
            assert set(ask.input_schema["properties"]) == {
                "question",
                "conversation_id",
                "file_url",
                "file_id",
            }
            out = _payload(await client.call_tool("ingest", {"path": str(SAMPLES / "agents")}))
            assert out["total_documents"] == 33
            decisions = _payload(await client.call_tool("list_decisions", {"verdict": "ESCALATE"}))
            assert decisions["total"] == 3
            blocked = await client.call_tool("ingest", {"path": str(tmp_path)})
            assert blocked.is_error
            assert "outside the allowed directories" in blocked.content[0].text
            missing = await client.call_tool("describe_document", {"doc_id": "nope"})
            assert missing.is_error
            assert "ingested documents are" in missing.content[0].text
            raw = await client.read_resource("prov://documents/example_0__ca1_prov")
            assert json.loads(raw.contents[0].text)["agent"]

    anyio.run(main)


def test_ask_returns_verdict_and_ingests_its_record(tmp_path):
    server, _store = _server(
        tmp_path, [call("list_documents"), AIMessage("There are 9 fault records.")]
    )

    async def main():
        async with Client(server) as client:
            await client.call_tool("ingest", {"path": str(SAMPLES / "agents/faults")})
            out = _payload(await client.call_tool("ask", {"question": "How many records?"}))
            assert out["verdict"] == "ACCEPT"
            assert out["record"].endswith("_prov.json")
            # The answer record is ingested, as kind "qa", and left out of listings unless
            # asked for: it is not one of the records being asked about.
            docs = _payload(await client.call_tool("list_documents", {}))
            assert not any(d["kind"] == "qa" for d in docs["documents"])
            assert docs["total"] == 9
            answers = _payload(await client.call_tool("list_documents", {"include_answers": True}))
            [qa] = [d for d in answers["documents"] if d["kind"] == "qa"]
            assert qa["title"].startswith("ProvenanceQA answer")

    anyio.run(main)
