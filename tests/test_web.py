"""The web page's API, relaying to the MCP server through a real MCP client session."""

from __future__ import annotations

from langchain_core.messages import AIMessage
from mcp import Client
from starlette.testclient import TestClient

from yprov_ask.client import record_id
from yprov_ask.web import create_app

from .conftest import SAMPLES, call
from .test_server import _server


def test_page_tools_documents_and_ask(tmp_path):
    server, store = _server(
        tmp_path,
        [
            call("list_decisions", verdict="ESCALATE"),
            AIMessage("Three decisions escalated."),
        ],
    )
    store.ingest_path(SAMPLES / "agents" / "faults")
    app = create_app(lambda: Client(server), label="in-process")

    with TestClient(app) as http:
        assert "Provenance Q&amp;A" in http.get("/").text

        info = http.get("/api/server").json()
        assert info["name"] == "yprov-ask"
        tools = {t["name"]: t for t in info["tools"]}
        assert tools["list_documents"]["read_only"] and not tools["ask"]["read_only"]
        assert "doc_id" in tools["describe_document"]["input_schema"]["properties"]

        docs = http.post("/api/tools/list_documents", json={}).json()["result"]
        assert docs["total"] == 9

        missing = http.post("/api/tools/describe_document", json={"doc_id": "nope"})
        assert missing.status_code == 422
        assert "ingested documents are" in missing.json()["error"]

        out = http.post("/api/tools/ask", json={"question": "Which escalated?"}).json()["result"]
        assert out["verdict"] == "ACCEPT"
        assert out["tool_calls"] == [
            {"tool": "list_decisions", "args": {"verdict": "ESCALATE"}, "ok": True}
        ]
        record = http.get(f"/api/documents/{record_id(out['record'])}").json()
        types = sorted(a["prov:type"]["$"] for a in record["activity"].values())
        assert types == [
            "yp:DecisionActivity",
            "yp:ReceiveTaskActivity",
            "yp:ResponseActivity",
            "yp:ToolCallActivity",
        ]
        assert http.get("/api/documents/nope").status_code == 404


def test_record_id_takes_the_file_name_on_any_platform():
    assert record_id(r"C:\x\records\qa_1-a_prov.json") == "qa_1-a_prov"
    assert record_id("/srv/records/qa_1-a_prov.json") == "qa_1-a_prov"
    assert record_id(None) is None
