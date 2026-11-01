"""Command line: ingest files, ask a question, or run the MCP server.

python -m yprov_ask ingest samples
python -m yprov_ask docs
python -m yprov_ask ask "Which run had the lowest validation loss?"
python -m yprov_ask serve                      # stdio, for Claude Desktop / Code
python -m yprov_ask serve --http --port 8000   # streamable HTTP at /mcp, widget at /widget/
python -m yprov_ask web                        # the web page, an MCP client, port 8080
python -m yprov_ask chat                       # the same, in the terminal

web and chat start the server themselves over stdio, or use a running one with
--server http://host:8000/mcp.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

SERVER_HELP = "URL of a running MCP server (default: start one over stdio)"


def _load_env(path: Path) -> None:
    """Read KEY=VALUE lines from .env into the environment, without overriding it."""
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def main(argv: list[str] | None = None) -> int:
    _load_env(Path.cwd() / ".env")
    parser = argparse.ArgumentParser(prog="yprov_ask", description=__doc__.splitlines()[0])
    parser.add_argument("--db", default=os.environ.get("YPROV_ASK_DB", "provqa.sqlite"))
    sub = parser.add_subparsers(dest="command", required=True)

    p_ingest = sub.add_parser("ingest", help="add PROV-JSON files or directories to the store")
    p_ingest.add_argument("paths", nargs="+")
    sub.add_parser("docs", help="list the ingested documents")
    p_ask = sub.add_parser("ask", help="answer one question with the agent")
    p_ask.add_argument("question")
    p_ask.add_argument("--model", default=None)
    p_ask.add_argument("--json", action="store_true", help="print the full result as JSON")
    p_serve = sub.add_parser("serve", help="run the MCP server")
    p_serve.add_argument("--http", action="store_true", help="streamable HTTP instead of stdio")
    p_serve.add_argument("--host", default="127.0.0.1")
    p_serve.add_argument("--port", type=int, default=8000)
    p_web = sub.add_parser("web", help="serve the web page, a client of the MCP server")
    p_web.add_argument("--server", default=None, help=SERVER_HELP)
    p_web.add_argument("--host", default="127.0.0.1")
    p_web.add_argument("--port", type=int, default=8080)
    p_chat = sub.add_parser("chat", help="ask questions in the terminal, over MCP")
    p_chat.add_argument("--server", default=None, help=SERVER_HELP)
    args = parser.parse_args(argv)
    os.environ["YPROV_ASK_DB"] = args.db

    from .store import ProvStore

    if args.command == "serve":
        from .server import create

        server = create()
        if args.http:
            import uvicorn

            from .widget import http_app

            print(
                f"MCP at http://{args.host}:{args.port}/mcp;"
                f" the chat widget's demo at http://{args.host}:{args.port}/widget/",
                flush=True,
            )
            uvicorn.run(
                http_app(server, host=args.host),
                host=args.host,
                port=args.port,
                log_level=os.environ.get("YPROV_ASK_LOG", "INFO").lower(),
            )
        else:
            server.run()
        return 0

    if args.command == "web":
        import uvicorn

        from .client import connect
        from .web import create_app

        app = create_app(lambda: connect(args.server), label=args.server or "stdio")
        print(f"Open http://{args.host}:{args.port}/ in a browser.", flush=True)
        uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
        return 0

    if args.command == "chat":
        import anyio

        from .client import chat

        anyio.run(chat, args.server)
        return 0

    store = ProvStore(args.db)
    if args.command == "ingest":
        for path in args.paths:
            for r in store.ingest_path(path):
                series = f", {r.metric_series} metric series" if r.metric_series else ""
                print(f"{r.status:9} {r.doc_id}  ({r.kind}{series})")
    elif args.command == "docs":
        from .tools import list_documents

        for d in list_documents(store)["documents"]:
            print(f"{d['doc_id']:40} {d['kind']:13} {d['title']}")
    elif args.command == "ask":
        from .agent import ProvenanceAgent

        answer = ProvenanceAgent(store, model_name=args.model).ask(args.question)
        if answer.record_path:
            store.ingest_file(Path(answer.record_path))
        if args.json:
            print(json.dumps(answer.as_dict(), indent=2))
        else:
            print(answer.answer)
            print(
                f"\n[{answer.verdict}] {answer.figures_checked} figure(s) checked,"
                f" {len(answer.ungrounded)} unsupported; {len(answer.tool_calls)} tool call(s);"
                f" {answer.seconds:.1f} s; record: {answer.record_path}"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
