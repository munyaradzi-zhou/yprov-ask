"""Files fetched by URL from the yProv store: the host allow-list, the checks on what comes
back, and ask with the file the user has open. The store is a mock transport: no network."""

from __future__ import annotations

import json
from pathlib import Path

import anyio
import httpx
import pytest
from langchain_core.messages import AIMessage, SystemMessage
from mcp import Client

from yprov_ask import fetch as fetch_module
from yprov_ask.agent import FILE_PROMPT, ProvenanceAgent
from yprov_ask.fetch import FetchError, allowed_hosts, cache_path, check_url, fetch
from yprov_ask.server import Settings, build_server
from yprov_ask.store import ProvStore

from .conftest import SAMPLES, ScriptedModel, call
from .test_server import _payload

STORE = "store.example.org"
PROV = sorted((SAMPLES / "yprov4ml").rglob("*.json"))[0].read_bytes()


def _store(request: httpx.Request) -> httpx.Response:
    path = request.url.path
    if path == "/files/run.json":
        return httpx.Response(200, content=PROV, headers={"content-type": "application/json"})
    if path == "/api/run":
        return httpx.Response(200, json={"result": PROV.decode("utf-8")})
    if path == "/moved":
        return httpx.Response(302, headers={"location": "http://10.0.0.1/secret"})
    if path == "/notprov":
        return httpx.Response(200, json={"hello": "world"})
    return httpx.Response(404)


@pytest.fixture
def http():
    with httpx.Client(transport=httpx.MockTransport(_store)) as client:
        yield client


def test_only_configured_hosts_and_http_urls():
    assert allowed_hosts(" Store.Example.org , localhost:9000") == [STORE, "localhost:9000"]
    check_url(f"https://{STORE}/files/run.json", [STORE])
    check_url("http://localhost:9000/x", ["localhost:9000"])
    for url, hosts, why in [
        (f"https://{STORE}/x", [], "no store hosts"),
        ("file:///etc/passwd", [STORE], "not an http"),
        ("http://10.0.0.1/x", [STORE], "not one of the store hosts"),
        ("http://localhost:9001/x", ["localhost:9000"], "not one of the store hosts"),
    ]:
        with pytest.raises(FetchError, match=why):
            check_url(url, hosts)


def test_cache_names_are_readable_and_stable(tmp_path):
    a = cache_path(f"https://{STORE}/files/run%201.json", tmp_path)
    assert a.name.startswith("run_1-") and a.suffix == ".json"
    assert a == cache_path(f"https://{STORE}/files/run%201.json", tmp_path)
    assert a != cache_path(f"https://{STORE}/other/run%201.json", tmp_path)
    assert cache_path(f"https://{STORE}/", tmp_path).name.startswith("store_file-")


def test_fetch_checks_what_comes_back(tmp_path, http, monkeypatch):
    path = fetch(f"https://{STORE}/files/run.json", [STORE], tmp_path, http)
    assert json.loads(path.read_text(encoding="utf-8")) == json.loads(PROV)
    wrapped = fetch(f"https://{STORE}/api/run", [STORE], tmp_path, http)
    assert json.loads(wrapped.read_text(encoding="utf-8")) == json.loads(PROV)

    for url, why in [
        ("/moved", "redirects are not followed"),
        ("/missing", "HTTP 404"),
        ("/notprov", "is not a PROV-JSON document"),
    ]:
        with pytest.raises(FetchError, match=why):
            fetch(f"https://{STORE}{url}", [STORE], tmp_path, http)

    monkeypatch.setattr(fetch_module, "MAX_BYTES", 100)
    with pytest.raises(FetchError, match="larger than"):
        fetch(f"https://{STORE}/files/run.json", [STORE], tmp_path, http)


def _server_with_store(tmp_path, http, replies):
    store = ProvStore()
    settings = Settings()
    settings.records = tmp_path / "records"
    settings.cache = tmp_path / "fetched"
    settings.store_hosts = [STORE]
    model = ScriptedModel(replies=list(replies), seen=[])
    server = build_server(
        store,
        settings,
        agent_factory=lambda: ProvenanceAgent(store, model, records_dir=settings.records),
        http_client=http,
    )
    return server, model


def test_ask_about_the_open_file(tmp_path, http):
    server, model = _server_with_store(
        tmp_path,
        http,
        [call("describe_document"), AIMessage("It is a training run.")],
    )

    async def main():
        async with Client(server) as client:
            got = _payload(
                await client.call_tool("store_get", {"url": f"https://{STORE}/files/run.json"})
            )
            opened = _payload(
                await client.call_tool(
                    "ask",
                    {
                        "question": "What is this file?",
                        "file_url": f"https://{STORE}/files/run.json",
                    },
                )
            )
            # A file that can't be opened is an error: never an answer over everything else.
            refused = await client.call_tool(
                "ask", {"question": "And this?", "file_url": "http://10.0.0.1/internal"}
            )
            denied = await client.call_tool("store_get", {"url": "http://10.0.0.1/internal"})
            return got, opened, refused, denied

    got, opened, refused, denied = anyio.run(main)

    doc_id = got["doc_id"]
    assert doc_id.startswith("run-") and got["total_documents"] == 1
    # The same URL again is the same document, not a copy.
    assert opened["file"] == {"doc_id": doc_id, "from": f"https://{STORE}/files/run.json"}
    system = [m.content for m in model.seen[0] if isinstance(m, SystemMessage)]
    assert system[0] == FILE_PROMPT and system[1].startswith(f"The open file is {doc_id}")
    assert opened["tool_calls"] == [{"tool": "describe_document", "args": {}, "ok": True}]

    record = json.loads(Path(opened["record"]).read_text(encoding="utf-8"))
    [source] = [v for k, v in record["entity"].items() if k.startswith("yp:SourceEntity")]
    assert source["prov:value"] == f"https://{STORE}/files/run.json"
    assert source["yp:doc_id"] == doc_id

    assert refused.is_error and "could not open the file" in refused.content[0].text
    assert len(model.seen) == 2  # the refused question never reached the model
    assert denied.is_error and "not one of the store hosts" in denied.content[0].text


def test_a_dropped_connection_is_retried_but_an_http_error_is_not(tmp_path, monkeypatch):
    monkeypatch.setattr(fetch_module, "RETRY_DELAY_SECONDS", 0)
    calls = []

    def flaky(request):
        calls.append(request.url.path)
        if len(calls) == 1:
            raise httpx.ReadError("connection reset")
        return _store(request)

    with httpx.Client(transport=httpx.MockTransport(flaky)) as client:
        assert fetch(f"https://{STORE}/files/run.json", [STORE], tmp_path, client).exists()
        assert len(calls) == 2
        with pytest.raises(FetchError, match="HTTP 404"):
            fetch(f"https://{STORE}/missing", [STORE], tmp_path, client)
        assert len(calls) == 3  # not retried

    def down(request):
        raise httpx.ConnectError("refused")

    with (
        httpx.Client(transport=httpx.MockTransport(down)) as client,
        pytest.raises(FetchError, match="could not fetch"),
    ):
        fetch(f"https://{STORE}/files/run.json", [STORE], tmp_path, client)
