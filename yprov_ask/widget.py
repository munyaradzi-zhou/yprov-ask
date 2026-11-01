"""The chat widget for the yProv Explorer, served next to the MCP endpoint.

The widget is one script, ``static/widget/yprov-ask.js``, that defines a ``<yprov-ask>``
element. Its styles live in a Shadow DOM, so it can go into a page we don't own (the
Explorer is a React app) without either side's CSS leaking into the other. It is an MCP
client itself: it calls the server's ``ask`` tool over streamable HTTP, with no relay.

    Explorer page ── <yprov-ask endpoint=".../mcp"> ──MCP over HTTP──► /mcp ─► ask ─► agent

``http_app`` is what ``serve --http`` runs:

    POST/GET/DELETE /mcp              the MCP endpoint (streamable HTTP)
    GET  /widget/yprov-ask.js         the widget
    GET  /widget/                     a demo page: the widget in an Explorer-like layout
    GET  /healthz                     {"status": "ok", "version": ...} for container checks

A page on another origin (the Explorer) may call /mcp only if its origin is listed in
YPROV_ASK_CORS (comma-separated, e.g. ``https://explorer.yprov.disi.unitn.it``).
"""

from __future__ import annotations

import os
from pathlib import Path

from mcp.server.mcpserver import MCPServer
from mcp.server.transport_security import TransportSecuritySettings
from starlette.applications import Starlette
from starlette.middleware.cors import CORSMiddleware
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse, Response

from . import __version__

WIDGET_DIR = Path(__file__).with_name("static") / "widget"

#: What the SDK allows by itself when bound to a loopback address.
_LOCAL_HOSTS = ["127.0.0.1:*", "localhost:*", "[::1]:*"]
_LOCAL_ORIGINS = ["http://127.0.0.1:*", "http://localhost:*", "http://[::1]:*"]


def cors_origins(value: str | None = None) -> list[str]:
    """The origins allowed to call /mcp from a browser, from YPROV_ASK_CORS."""
    raw = os.environ.get("YPROV_ASK_CORS", "") if value is None else value
    return [o.strip().rstrip("/") for o in raw.split(",") if o.strip()]


def add_widget_routes(server: MCPServer) -> None:
    @server.custom_route("/widget/yprov-ask.js", methods=["GET"])
    async def script(request: Request) -> Response:
        return FileResponse(WIDGET_DIR / "yprov-ask.js", media_type="text/javascript")

    @server.custom_route("/widget/", methods=["GET"])
    async def demo(request: Request) -> Response:
        return FileResponse(WIDGET_DIR / "demo.html", media_type="text/html")

    @server.custom_route("/healthz", methods=["GET"])
    async def health(request: Request) -> Response:
        # Liveness only: answers without touching the store or the model.
        return JSONResponse({"status": "ok", "version": __version__})


def http_app(
    server: MCPServer, host: str = "127.0.0.1", origins: list[str] | None = None
) -> Starlette:
    """The MCP endpoint, the widget and its demo page, with CORS for ``origins``."""
    origins = cors_origins() if origins is None else origins
    add_widget_routes(server)
    security = None
    if host in ("127.0.0.1", "localhost", "::1"):
        # The SDK guards a loopback server against DNS rebinding by checking Origin; a
        # browser page from an allowed origin has to pass that check as well as CORS.
        security = TransportSecuritySettings(
            enable_dns_rebinding_protection=True,
            allowed_hosts=_LOCAL_HOSTS,
            allowed_origins=_LOCAL_ORIGINS + origins,
        )
    app = server.streamable_http_app(transport_security=security, host=host)
    if origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=origins,
            allow_methods=["GET", "POST", "DELETE"],
            allow_headers=["content-type", "accept", "mcp-session-id", "mcp-protocol-version"],
            # The widget reads the session id the server assigns on initialize.
            expose_headers=["mcp-session-id"],
        )
    return app
