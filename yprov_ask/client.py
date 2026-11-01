"""Our own MCP client: the connection the web page and the terminal chat both use.

Nothing here talks to the store or the agent directly. Every question and every lookup
goes over MCP to the server, exactly as Claude Desktop or any other client would send it,
so a demo through these front ends exercises the same interface the server publishes.

    connect(None)                          start the server as a subprocess, talk over stdio
    connect("http://host:8000/mcp")        talk to a running server over streamable HTTP
"""

from __future__ import annotations

import json
import os
import re
import sys
from typing import Any

import anyio
from mcp import Client, StdioServerParameters

ERROR_PREFIX = re.compile(r"^Error executing tool \w+: ")


class CallFailed(Exception):
    """The server answered a tool call with an error; the message is the server's."""


def connect(server: str | None = None) -> Client:
    if server:
        # SDK 2.3's protocol probe fails against its own HTTP server; legacy mode works.
        return Client(server, mode="legacy")
    # The SDK passes a subprocess only a handful of variables by default; the server needs
    # its own configuration and the API key. Its stderr is this terminal, so it logs only
    # warnings: a failed tool call is already shown to the user as the call's result.
    env = {"YPROV_ASK_LOG": "WARNING", **os.environ}
    params = StdioServerParameters(
        command=sys.executable, args=["-m", "yprov_ask", "serve"], env=env, cwd=os.getcwd()
    )
    return Client(params)


def payload(result: Any) -> Any:
    """The tool's result: its structured content if the server sent it, else its text."""
    text = "".join(getattr(c, "text", "") for c in result.content)
    if result.is_error:
        # "Error executing tool NAME: " is the SDK's wrapper; the rest is the server's message.
        raise CallFailed(ERROR_PREFIX.sub("", text) or "the tool failed without a message")
    structured = getattr(result, "structured_content", None)
    if structured is not None:
        return structured
    try:
        return json.loads(text)
    except ValueError:
        return text


async def call(client: Client, tool: str, arguments: dict[str, Any] | None = None) -> Any:
    return payload(await client.call_tool(tool, arguments or {}))


async def read_document(client: Client, doc_id: str) -> Any:
    result = await client.read_resource(f"prov://documents/{doc_id}")
    return json.loads(result.contents[0].text)


def record_id(path: str | None) -> str | None:
    """The store's doc_id for an answer's PROV record: the file name without .json."""
    if not path:
        return None
    return re.split(r"[\\/]", path)[-1].removesuffix(".json")


HELP = """\
Type a question, or:
  /docs            the ingested documents
  /tools           the tools the server publishes
  /tool NAME JSON  call one tool yourself, e.g. /tool list_decisions {"verdict": "ESCALATE"}
  /record          the PROV record of the last answer
  /quit"""


def _show_answer(out: dict[str, Any]) -> None:
    g = out["grounding"]
    print(f"\n{out['answer']}\n")
    print(
        f"  [{out['verdict']}] {g['figures_checked']} figure(s) checked against tool results,"
        f" {len(g['ungrounded_figures'])} unsupported ({g['rule']})"
    )
    for i, c in enumerate(out["tool_calls"], start=1):
        args = json.dumps(c["args"], separators=(", ", ": "))
        print(f"  {i}. {c['tool']}({args}){'' if c['ok'] else '  -> error'}")
    print(f"  {out['model']}, {out['seconds']} s, record {record_id(out.get('record'))}\n")


async def chat(server: str | None = None) -> None:
    async with connect(server) as client:
        info = client.server_info
        docs = await call(client, "list_documents")
        where = server or "stdio"
        print(f"Connected to {info.name} {info.version} over {where}: {docs['total']} documents.")
        print(HELP)
        last: dict[str, Any] | None = None
        while True:
            try:
                line = (await anyio.to_thread.run_sync(input, "\nquestion> ")).strip()
            except (EOFError, KeyboardInterrupt):
                return
            if not line:
                continue
            try:
                if line in ("/quit", "/exit"):
                    return
                if line == "/docs":
                    for d in (await call(client, "list_documents"))["documents"]:
                        print(f"  {d['doc_id']:40} {d['collection']:10} {d['kind']}")
                elif line == "/tools":
                    for t in (await client.list_tools()).tools:
                        print(f"  {t.name:18} {(t.description or '').splitlines()[0]}")
                elif line.startswith("/tool "):
                    name, _, raw = line[6:].strip().partition(" ")
                    out = await call(client, name, json.loads(raw) if raw.strip() else {})
                    print(json.dumps(out, indent=2))
                elif line == "/record":
                    rid = record_id(last.get("record")) if last else None
                    if not rid:
                        print("  no answer yet")
                    else:
                        print(json.dumps(await read_document(client, rid), indent=2))
                elif line.startswith("/"):
                    print(HELP)
                else:
                    print("  thinking: the agent is querying the records ...")
                    last = await call(client, "ask", {"question": line})
                    _show_answer(last)
            except (CallFailed, ValueError) as exc:
                print(f"  error: {exc}")
