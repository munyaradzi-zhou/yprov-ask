"""One file at a time: uploads, file ids, and an agent whose tools see that file only."""

from __future__ import annotations

import json

import anyio
from langchain_core.messages import AIMessage, ToolMessage
from mcp import Client
from starlette.testclient import TestClient

from yprov_ask import server as server_module
from yprov_ask.agent import ProvenanceAgent
from yprov_ask.files import OpenFiles
from yprov_ask.server import Settings, build_server
from yprov_ask.store import ProvStore
from yprov_ask.widget import http_app

from .conftest import SAMPLES, ScriptedModel, call
from .test_server import _payload

LOCAL = "http://127.0.0.1:8000"
RUN = SAMPLES / "yprov4ml" / "example_0" / "prov_example_GR0_0.json"


def _setup(tmp_path, replies):
    store = ProvStore()
    store.ingest_path(SAMPLES)  # many other documents, none of which may leak into answers
    settings = Settings()
    settings.records = tmp_path / "records"
    settings.cache = tmp_path / "cache"
    model = ScriptedModel(replies=list(replies), seen=[])
    server = build_server(
        store,
        settings,
        agent_factory=lambda: ProvenanceAgent(store, model, records_dir=settings.records),
    )
    return server, store, model


def _upload(http, path, name=None):
    return http.post(f"/files?name={name or path.name}", content=path.read_bytes())


def test_registry_hands_out_tokens_and_forgets_idle_files():
    now = [0.0]
    files = OpenFiles(idle_seconds=10, clock=lambda: now[0])
    a, b = files.open("doc", "a.json", "upload"), files.open("doc", "a.json", "upload")
    assert a.file_id != b.file_id and len(a.file_id) == 32
    assert files.get(a.file_id).doc_id == "doc"
    now[0] = 11
    assert files.get(a.file_id) is None


def test_upload_checks_and_deduplicates(tmp_path, monkeypatch):
    server, store, _ = _setup(tmp_path, [])
    before = len(store.documents())
    with TestClient(http_app(server, origins=[]), base_url=LOCAL) as http:
        first = _upload(http, RUN, "my run.json").json()
        second = _upload(http, RUN, "my run.json").json()
        assert first["doc_id"] == second["doc_id"] and first["file_id"] != second["file_id"]
        assert first["kind"] == "yprov4ml" and first["activities"] > 0
        # The content's fingerprint, the same as in the doc id: a renamed copy of another
        # file shows by it.
        import hashlib

        assert first["content_id"] == hashlib.sha256(RUN.read_bytes()).hexdigest()[:10]
        assert len(store.documents()) == before + 1

        bad = http.post("/files?name=x.json", content=b'{"hello": "world"}')
        assert bad.status_code == 422 and "PROV-JSON" in bad.json()["error"]
        monkeypatch.setattr(server_module, "MAX_BYTES", 100)
        big = _upload(http, RUN)
        assert big.status_code == 413


def test_the_agent_sees_only_the_open_file(tmp_path):
    server, store, model = _setup(
        tmp_path,
        [
            call("search_nodes", query="Validation"),
            AIMessage("The file logs validation metrics."),
            call("describe_document", i=1),
            AIMessage("It is a training run."),
        ],
    )
    with TestClient(http_app(server, origins=[]), base_url=LOCAL) as http:
        opened = _upload(http, RUN).json()
    doc_id = opened["doc_id"]
    documents_before = len(store.documents())

    async def main():
        async with Client(server) as client:
            first = _payload(
                await client.call_tool(
                    "ask", {"question": "Which validation metrics?", "file_id": opened["file_id"]}
                )
            )
            unknown = await client.call_tool("ask", {"question": "Hi", "file_id": "nope"})
            # The same file in the same conversation: the conversation continues.
            other = _payload(
                await client.call_tool(
                    "ask",
                    {
                        "question": "What is it?",
                        "file_id": opened["file_id"],
                        "conversation_id": first["conversation_id"],
                    },
                )
            )
            return first, unknown, other

    first, unknown, other = anyio.run(main)

    # The model was offered tools pinned to the file: no list_documents, no doc_id anywhere.
    offered = model.bound[-1]
    assert "list_documents" not in offered and "decline" in offered
    assert all("doc_id" not in args for args in offered.values())
    # And what they returned came from that file alone, though the store holds 36 documents.
    [result] = [m for m in model.seen[1] if isinstance(m, ToolMessage)]
    matches = json.loads(result.content)["matches"]
    assert matches and {m["doc_id"] for m in matches} == {doc_id}
    assert first["file"]["doc_id"] == doc_id

    assert unknown.is_error and "unknown file id" in unknown.content[0].text
    assert other["turn"] == 2 and not other["conversation_restarted"]
    # Answer records are written, but the store stays the user's documents.
    assert len(store.documents()) == documents_before


def test_switching_file_starts_a_new_conversation(tmp_path):
    other_run = SAMPLES / "yprov4ml" / "example_2" / "prov_example_GR0_2.json"
    server, _, _ = _setup(
        tmp_path,
        [
            call("describe_document"),
            AIMessage("A run."),
            call("describe_document", i=1),
            AIMessage("Another run."),
        ],
    )
    with TestClient(http_app(server, origins=[]), base_url=LOCAL) as http:
        a = _upload(http, RUN).json()
        b = _upload(http, other_run).json()

    async def main():
        async with Client(server) as client:
            one = _payload(
                await client.call_tool("ask", {"question": "What?", "file_id": a["file_id"]})
            )
            two = _payload(
                await client.call_tool(
                    "ask",
                    {
                        "question": "And?",
                        "file_id": b["file_id"],
                        "conversation_id": one["conversation_id"],
                    },
                )
            )
            return one, two

    one, two = anyio.run(main)
    assert two["conversation_restarted"] and two["turn"] == 1
    assert two["conversation_id"] != one["conversation_id"]
    assert two["file"]["doc_id"] == b["doc_id"]
