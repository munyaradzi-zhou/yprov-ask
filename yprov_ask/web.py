"""A web page for the provenance agent, for people who will not open a terminal.

The page is served by a small Starlette app that holds one MCP session to the server and
relays the browser's requests over it. It adds no logic of its own: the answer, the
verdict, the tool calls and the PROV record all come back from the server's tools.

    browser ──JSON──► web app (MCP client) ──MCP, stdio or HTTP──► MCP server ──► agent

    GET  /                       the page
    GET  /api/server             server name, version, instructions, tools and their schemas
    POST /api/tools/{name}       call one tool with the JSON body as its arguments
    GET  /api/documents/{doc_id} one ingested document (the prov://documents resource)
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from mcp import Client
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse
from starlette.routing import Route

from .client import CallFailed, call, read_document

PAGE = Path(__file__).with_name("static") / "index.html"


def create_app(connect: Callable[[], Client], label: str = "stdio") -> Starlette:
    """The web app. ``connect`` returns an unopened MCP client; the app opens it once at
    start-up and closes it at shutdown."""

    @asynccontextmanager
    async def lifespan(app: Starlette) -> AsyncIterator[None]:
        async with connect() as client:
            app.state.client = client
            yield

    def client_of(request: Request) -> Client:
        return request.app.state.client

    async def page(request: Request) -> HTMLResponse:
        return HTMLResponse(PAGE.read_text(encoding="utf-8"))

    async def server(request: Request) -> JSONResponse:
        client = client_of(request)
        tools = (await client.list_tools()).tools
        return JSONResponse(
            {
                "name": client.server_info.name,
                "version": client.server_info.version,
                "transport": label,
                "instructions": client.instructions,
                "tools": [
                    {
                        "name": t.name,
                        "description": t.description or "",
                        "input_schema": t.input_schema,
                        "read_only": bool(t.annotations and t.annotations.read_only_hint),
                    }
                    for t in tools
                ],
            }
        )

    async def tool(request: Request) -> JSONResponse:
        try:
            body = await request.body()
            arguments: dict[str, Any] = json.loads(body) if body.strip() else {}
        except ValueError:
            return JSONResponse({"error": "the request body is not JSON"}, status_code=400)
        try:
            result = await call(client_of(request), request.path_params["name"], arguments)
        except CallFailed as exc:
            return JSONResponse({"error": str(exc)}, status_code=422)
        return JSONResponse({"result": result})

    async def document(request: Request) -> JSONResponse:
        try:
            doc = await read_document(client_of(request), request.path_params["doc_id"])
        except Exception as exc:  # the SDK raises its own error type for a missing resource
            return JSONResponse({"error": str(exc)}, status_code=404)
        return JSONResponse(doc)

    return Starlette(
        routes=[
            Route("/", page),
            Route("/api/server", server),
            Route("/api/tools/{name}", tool, methods=["POST"]),
            Route("/api/documents/{doc_id:path}", document),
        ],
        lifespan=lifespan,
    )
