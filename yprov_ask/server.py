"""The MCP server: the store's queries and the question-answering agent, over MCP.

Two ways in, for two kinds of client:

    query tools   list_documents, describe_document, search_nodes, get_node,
                  trace_lineage, metric_summary, metric_series, list_decisions
                  Deterministic. A client that brings its own model (Claude Desktop,
                  Claude Code, an IDE) calls these and does its own reasoning.

    ask           One question in natural language, answered by this server's LangGraph
                  agent, with the grounding verdict and a PROV record of how it answered.
                  For a client with no model of its own, or when the answer must be
                  checked and recorded the same way every time. Pass the conversation_id
                  a previous answer returned, and the agent sees the earlier turns; pass
                  file_url, and the agent is told which document the user has open.

    store_get     Fetches a PROV-JSON file by URL from an allowed store host and ingests it.

    open_file     Opens one file (by URL) for questions and returns its file id. A file from the
                  user's computer is uploaded with POST /files instead (HTTP only). ask with a
                  file_id or file_url answers from that file alone.

    ingest        Adds PROV-JSON files to the store, from directories the operator allowed.

Configuration is by environment, so the same code runs under stdio and over HTTP:

    YPROV_ASK_DB       SQLite file of the store            (default: provqa.sqlite)
    YPROV_ASK_INGEST   paths ingested at start, separated by os.pathsep
    YPROV_ASK_ROOTS    directories ``ingest`` may read     (default: the ingest paths)
    YPROV_ASK_RECORDS  where answers' PROV records go      (default: records)
    YPROV_ASK_MODEL    chat model id                       (default: agent.DEFAULT_MODEL)
    YPROV_ASK_STORE_HOSTS  hosts store_get and file_url may fetch from (comma-separated)
    YPROV_ASK_CACHE    where fetched files are kept        (default: fetched, beside records)
    YPROV_ASK_LOG      server log level                    (default: INFO)
    OPENAI_API_KEY          needed only by ``ask``
"""

from __future__ import annotations

import inspect
import json
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any, get_type_hints

import anyio
import httpx
from mcp.server.mcpserver import Context, MCPServer
from mcp.server.mcpserver.exceptions import ResourceNotFoundError
from mcp.server.mcpserver.exceptions import ToolError as AnticipatedToolError
from mcp.types import ToolAnnotations
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from . import __version__
from .conversation import Conversations, Turn
from .fetch import MAX_BYTES, FetchError, allowed_hosts, fetch
from .files import OpenFiles, save_upload
from .record import ConversationLink, Source
from .store import ProvStore
from .tools import QUERY_TOOLS, ToolError, list_documents, resolve_doc

INSTRUCTIONS = """\
Answers questions about W3C PROV provenance files (yProv4ML training runs and yProv4Agents
agent records). Call list_documents first. Use the query tools to look things up yourself,
or call ask(question) to have this server's agent answer and check the answer against the
records. Every figure ask returns has been checked against tool results; its verdict says
whether the check passed."""


def _paths(value: str | None) -> list[Path]:
    return [Path(p).expanduser().resolve() for p in (value or "").split(os.pathsep) if p]


def _bind(func: Callable[..., Any], store: ProvStore) -> Callable[..., Any]:
    """The query function with ``store`` bound, keeping the rest of its signature, so the
    MCP SDK derives the same input schema the function declares."""
    hints = get_type_hints(func)
    params = [
        p.replace(annotation=hints[p.name])
        for p in list(inspect.signature(func).parameters.values())[1:]
    ]

    def tool(**kwargs: Any) -> Any:
        try:
            return func(store, **kwargs)
        except ToolError as exc:
            # The SDK shows the client only "Error executing tool" for any exception but
            # its own ToolError; these messages say what does exist, and are the point.
            raise AnticipatedToolError(str(exc)) from exc

    tool.__name__ = func.__name__
    tool.__doc__ = func.__doc__
    tool.__signature__ = inspect.Signature(params, return_annotation=dict[str, Any])  # type: ignore[attr-defined]
    return tool


class Settings:
    def __init__(self) -> None:
        self.db = Path(os.environ.get("YPROV_ASK_DB", "provqa.sqlite"))
        self.ingest = _paths(os.environ.get("YPROV_ASK_INGEST"))
        self.roots = _paths(os.environ.get("YPROV_ASK_ROOTS")) or list(self.ingest)
        self.records = Path(os.environ.get("YPROV_ASK_RECORDS", "records")).resolve()
        self.model = os.environ.get("YPROV_ASK_MODEL")
        self.store_hosts = allowed_hosts()
        self.cache = Path(
            os.environ.get("YPROV_ASK_CACHE", self.records.parent / "fetched")
        ).resolve()
        self.log_level = os.environ.get("YPROV_ASK_LOG", "INFO").upper()


def build_server(
    store: ProvStore,
    settings: Settings,
    agent_factory: Callable[[], Any] | None = None,
    http_client: httpx.Client | None = None,
) -> MCPServer:
    """An MCP server over ``store``. ``agent_factory`` builds the agent on the first
    ``ask``; tests pass one with a scripted model, and an ``http_client`` that answers
    store URLs without a network."""
    server = MCPServer(
        "yprov-ask",
        instructions=INSTRUCTIONS,
        version=__version__,
        log_level=settings.log_level,
    )
    read_only = ToolAnnotations(readOnlyHint=True, openWorldHint=False)
    for func in QUERY_TOOLS:
        server.add_tool(_bind(func, store), annotations=read_only)

    agent: dict[str, Any] = {}
    conversations = Conversations()
    open_files = OpenFiles()

    def fetch_and_ingest(url: str) -> Source:
        path = fetch(url, settings.store_hosts, settings.cache, http_client)
        result = store.ingest_file(path)
        if result is None:  # fetch() already checked it is PROV-JSON
            raise FetchError(f"{url} could not be ingested")
        return Source(url, result.doc_id)

    def ingest_upload(raw: bytes, name: str) -> Source:
        path = save_upload(raw, name, settings.cache / "uploads")
        result = store.ingest_file(path)
        if result is None:  # save_upload() already checked it is PROV-JSON
            raise FetchError(f"{name} could not be ingested")
        return Source(f"upload:{name}", result.doc_id)

    def opened(source: Source, name: str, origin: str) -> dict[str, Any]:
        item = open_files.open(source.doc_id, name, origin)
        row = store.document(source.doc_id)
        return {
            "file_id": item.file_id,
            "name": name,
            "doc_id": source.doc_id,
            # The first characters of the content's SHA-256: the same file always has the same
            # one, so a renamed copy of another file shows (a jt2 run uploaded as "jt").
            "content_id": row["sha256"].removeprefix("sha256:")[:10],
            "kind": row["kind"],
            "entities": row["n_entities"],
            "activities": row["n_activities"],
            "agents": row["n_agents"],
            "relations": row["n_relations"],
        }

    @server.custom_route("/files", methods=["POST"])
    async def upload(request: Request) -> Response:
        """A file from the user's computer: the raw PROV-JSON as the body, ?name= its name."""
        name = request.query_params.get("name") or "upload.json"
        body = bytearray()
        async for chunk in request.stream():
            body += chunk
            if len(body) > MAX_BYTES:
                return JSONResponse({"error": "the file is larger than 20 MB"}, status_code=413)
        try:
            source = await anyio.to_thread.run_sync(lambda: ingest_upload(bytes(body), name))
        except FetchError as exc:
            return JSONResponse({"error": str(exc)}, status_code=422)
        return JSONResponse(opened(source, name, "upload"))

    def get_agent() -> Any:
        if "agent" not in agent:
            if agent_factory is not None:
                agent["agent"] = agent_factory()
            else:
                if not os.environ.get("OPENAI_API_KEY"):
                    raise AnticipatedToolError(
                        "ask needs OPENAI_API_KEY: set it in the environment, or put"
                        f" OPENAI_API_KEY=... in {Path.cwd() / '.env'}, then restart. The"
                        " Documents and Tools (the query tools) work without it."
                    )
                from .agent import ProvenanceAgent

                agent["agent"] = ProvenanceAgent(
                    store, model_name=settings.model, records_dir=settings.records
                )
        return agent["agent"]

    @server.tool(annotations=ToolAnnotations(readOnlyHint=False, openWorldHint=True))
    async def ask(
        question: str,
        ctx: Context,
        conversation_id: str | None = None,
        file_url: str | None = None,
        file_id: str | None = None,
    ) -> dict[str, Any]:
        """Answer a question about the ingested provenance in natural language.

        The answer is written by a language model that may only use the query tools, then
        checked in code: every figure it states must be one a tool returned. The result
        carries the answer, the verdict of that check (ACCEPT or REVISE), the tool calls
        made, and the path of the PROV record describing how the answer was produced.

        To ask a follow-up, pass the conversation_id from the previous result: the agent
        then sees the conversation's last 3 answered turns. Without one, or with one the server
        no longer knows, a new conversation starts; the result always carries its id,
        and conversation_restarted is true when a given id was not found.

        One file: pass file_id (from open_file, or from uploading with POST /files) or
        file_url (the yProv Explorer's ?file=). The agent then answers from that file alone:
        its tools are pinned to it, and a question about anything else is declined. A file
        that can't be opened is an error, never a fallback to everything the server holds.
        A conversation is about one file; asking about another starts a new one.

        A client that asks for progress gets a notification at each step (each lookup, the
        figure check, the record), with a short message saying what the agent is doing.
        """
        steps = 0

        def on_step(message: str) -> None:
            # Called from the agent's worker thread; the notification is sent on the loop.
            nonlocal steps
            steps += 1
            anyio.from_thread.run(ctx.report_progress, steps, None, message)

        source = None
        if file_id:
            item = open_files.get(file_id)
            if item is None:
                raise AnticipatedToolError(
                    "unknown file id: the server no longer has this file; open it again"
                )
            source = Source(
                item.origin if item.origin != "upload" else f"upload:{item.name}", item.doc_id
            )
        elif file_url:
            await ctx.report_progress(0, None, "Opening the file you are viewing")
            try:
                source = await anyio.to_thread.run_sync(lambda: fetch_and_ingest(file_url))
            except FetchError as exc:
                raise AnticipatedToolError(f"could not open the file: {exc}") from exc
        conversation = conversations.resume(conversation_id, source.doc_id if source else None)
        history = conversation.for_model()
        previous = conversation.previous
        link = ConversationLink(
            conversation.id, conversation.count + 1, previous.session if previous else None
        )
        answer = await anyio.to_thread.run_sync(
            lambda: get_agent().ask(
                question, history=history, conversation=link, source=source, on_step=on_step
            )
        )
        turn = conversations.add(
            conversation,
            Turn(question, answer.answer, answer.session, declined=answer.declined is not None),
        )
        if answer.record_path and source is None:
            # The record is provenance like any other: ingesting it lets a later question
            # ask what was answered before, and whether it was accepted. Not with one file
            # open: the record is written, but the store stays the user's file.
            store.ingest_file(Path(answer.record_path))
        out = answer.as_dict()
        out["conversation_id"] = conversation.id
        out["turn"] = turn
        out["conversation_restarted"] = bool(conversation_id) and conversation_id != conversation.id
        if source is not None:
            out["file"] = {"doc_id": source.doc_id, "from": source.url}
        return out

    @server.tool(annotations=ToolAnnotations(readOnlyHint=False, openWorldHint=True))
    async def open_file(url: str) -> dict[str, Any]:
        """Open one PROV-JSON file, by URL, for questions; returns its file_id for ask.

        Only the store hosts this server was configured with may be fetched. (A file from
        the user's computer is uploaded over HTTP, POST /files, which returns the same.)
        """
        try:
            source = await anyio.to_thread.run_sync(lambda: fetch_and_ingest(url))
        except FetchError as exc:
            raise AnticipatedToolError(str(exc)) from exc
        return opened(source, url.rstrip("/").rsplit("/", 1)[-1] or url, url)

    @server.tool(annotations=ToolAnnotations(readOnlyHint=False, openWorldHint=True))
    async def store_get(url: str) -> dict[str, Any]:
        """Fetch a PROV-JSON file from the yProv store by URL, and add it to the store here.

        Only the store hosts this server was configured with may be fetched. Fetching the
        same URL again replaces the earlier version. Returns the document's doc_id, for the
        query tools.
        """
        try:
            source = await anyio.to_thread.run_sync(lambda: fetch_and_ingest(url))
        except FetchError as exc:
            raise AnticipatedToolError(str(exc)) from exc
        return {
            "url": url,
            "doc_id": source.doc_id,
            "total_documents": list_documents(store)["total"],
        }

    @server.tool(annotations=ToolAnnotations(readOnlyHint=False, openWorldHint=False))
    async def ingest(path: str) -> dict[str, Any]:
        """Add a PROV-JSON file, or every .json file under a directory, to the store.

        Only paths under the directories the server was started with may be read.
        """
        target = Path(path).expanduser().resolve()
        if not any(target == root or root in target.parents for root in settings.roots):
            allowed = ", ".join(str(r) for r in settings.roots) or "none configured"
            raise AnticipatedToolError(f"{target} is outside the allowed directories: {allowed}")
        try:
            results = await anyio.to_thread.run_sync(lambda: store.ingest_path(target))
        except FileNotFoundError as exc:
            raise AnticipatedToolError(str(exc)) from exc
        return {
            "ingested": [
                {
                    "doc_id": r.doc_id,
                    "kind": r.kind,
                    "status": r.status,
                    "metric_series": r.metric_series,
                }
                for r in results
            ],
            "total_documents": list_documents(store)["total"],
        }

    @server.resource("prov://documents", mime_type="application/json")
    def documents() -> str:
        """The ingested documents."""
        return json.dumps(list_documents(store), indent=2)

    @server.resource("prov://documents/{doc_id}", mime_type="application/json")
    def document(doc_id: str) -> str:
        """One ingested document, exactly as it was read from disk."""
        try:
            row = store.document(resolve_doc(store, doc_id))
        except ToolError as exc:
            raise ResourceNotFoundError(str(exc)) from exc
        return row["raw"]

    return server


def create() -> MCPServer:
    """The server as configured by the environment, with its start-up paths ingested."""
    settings = Settings()
    store = ProvStore(settings.db)
    for path in settings.ingest:
        store.ingest_path(path)
    return build_server(store, settings)
