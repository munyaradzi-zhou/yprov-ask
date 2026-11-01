"""The chat widget's HTTP side: the files it is served as, CORS, and the raw MCP exchange
the widget's script makes (initialize, then tools/call ask), with a scripted model."""

from __future__ import annotations

import json

from langchain_core.messages import AIMessage
from starlette.testclient import TestClient

from yprov_ask.widget import cors_origins, http_app

from .conftest import SAMPLES, call
from .test_server import _server

EXPLORER = "https://explorer.yprov.disi.unitn.it"
LOCAL = "http://127.0.0.1:8000"  # the SDK checks Host on a loopback server
MCP_HEADERS = {"Accept": "application/json, text/event-stream", "Origin": EXPLORER}


def _reply(response, id_):
    """The JSON-RPC reply to request ``id_``, from a JSON or server-sent-events body, as
    the widget's readReply finds it."""
    if response.headers["content-type"].startswith("text/event-stream"):
        for event in response.text.replace("\r\n", "\n").split("\n\n"):
            data = "\n".join(
                line[5:].lstrip() for line in event.split("\n") if line.startswith("data:")
            )
            if data and json.loads(data).get("id") == id_:
                return json.loads(data)
        raise AssertionError(f"no reply to {id_} in {response.text!r}")
    return response.json()


def _progress(response, token):
    """The messages of the progress notifications for ``token``, in order."""
    messages = []
    for event in response.text.replace("\r\n", "\n").split("\n\n"):
        for line in event.split("\n"):
            if line.startswith("data:"):
                message = json.loads(line[5:])
                params = message.get("params") or {}
                if message.get("method") == "notifications/progress" and (
                    params.get("progressToken") == token
                ):
                    messages.append(params.get("message"))
    return messages


def test_cors_origins_from_the_environment(monkeypatch):
    monkeypatch.setenv("YPROV_ASK_CORS", f" {EXPLORER}/ , http://localhost:3000")
    assert cors_origins() == [EXPLORER, "http://localhost:3000"]
    assert cors_origins("") == []


def test_widget_files_and_cors(tmp_path):
    server, _ = _server(tmp_path)
    with TestClient(http_app(server, origins=[EXPLORER]), base_url=LOCAL) as http:
        script = http.get("/widget/yprov-ask.js")
        assert script.status_code == 200
        assert script.headers["content-type"].startswith("text/javascript")
        assert 'customElements.define("yprov-ask"' in script.text
        assert "<yprov-ask" in http.get("/widget/").text
        from yprov_ask import __version__

        assert http.get("/healthz").json() == {"status": "ok", "version": __version__}

        allowed = http.options(
            "/mcp",
            headers={
                "Origin": EXPLORER,
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "content-type,mcp-session-id",
            },
        )
        assert allowed.status_code == 200
        assert allowed.headers["access-control-allow-origin"] == EXPLORER

        refused = http.options(
            "/mcp",
            headers={
                "Origin": "https://elsewhere.example",
                "Access-Control-Request-Method": "POST",
            },
        )
        assert "access-control-allow-origin" not in refused.headers


def test_the_widgets_mcp_exchange(tmp_path):
    server, store = _server(
        tmp_path,
        [
            call("list_decisions", verdict="ESCALATE"),
            AIMessage("Three decisions escalated."),
        ],
    )
    store.ingest_path(SAMPLES / "agents" / "faults")
    with TestClient(http_app(server, origins=[EXPLORER]), base_url=LOCAL) as http:
        init = http.post(
            "/mcp",
            headers=MCP_HEADERS,
            json={
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2025-06-18",
                    "capabilities": {},
                    "clientInfo": {"name": "yprov-ask", "version": "0.1.0"},
                },
            },
        )
        assert init.status_code == 200
        # The widget reads this header, so CORS must expose it to a page on another origin.
        assert "mcp-session-id" in init.headers["access-control-expose-headers"].lower()
        session = {"Mcp-Session-Id": init.headers["mcp-session-id"]}
        version = _reply(init, 1)["result"]["protocolVersion"]
        session["MCP-Protocol-Version"] = version

        note = http.post(
            "/mcp",
            headers={**MCP_HEADERS, **session},
            json={"jsonrpc": "2.0", "method": "notifications/initialized"},
        )
        assert note.status_code == 202

        asked = http.post(
            "/mcp",
            headers={**MCP_HEADERS, **session},
            json={
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/call",
                "params": {
                    "name": "ask",
                    "arguments": {"question": "Which escalated?"},
                    "_meta": {"progressToken": "p2"},
                },
            },
        )
        # Before the reply, one progress notification per step, as the widget shows them.
        assert _progress(asked, "p2") == [
            "Looking up: list_decisions",
            "Checking every figure against the records",
            "Writing the provenance record",
        ]
        result = _reply(asked, 2)["result"]
        assert not result["isError"]
        out = result["structuredContent"]
        assert out["answer"] == "Three decisions escalated."
        assert out["verdict"] == "ACCEPT"
        assert out["tool_calls"] == [
            {"tool": "list_decisions", "args": {"verdict": "ESCALATE"}, "ok": True}
        ]

        # An origin that isn't allowed is refused by the SDK's own check, not only by CORS.
        stranger = http.post(
            "/mcp",
            headers={**MCP_HEADERS, **session, "Origin": "https://elsewhere.example"},
            json={"jsonrpc": "2.0", "id": 3, "method": "tools/list"},
        )
        assert stranger.status_code == 403
