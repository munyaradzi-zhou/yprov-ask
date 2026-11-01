"""Ask questions in natural language about W3C PROV-JSON files, over MCP.

Layers, each usable without the ones above it:

    store.py   ingest PROV-JSON (and yProv4ML metric CSVs) into SQLite
    tools.py   deterministic queries over the store: the only source of facts
    agent.py   a LangGraph agent that answers with those tools and checks its own figures
    server.py  the MCP server exposing the tools and the agent
    client.py  our MCP client (web.py serves it as a page, `chat` in a terminal)

The agent's answers are themselves written as PROV-JSON (record.py), in the same ``yp:``
vocabulary the yProv4Agents demonstrator uses, so an answer can be ingested and audited
like any other record.
"""

import os
import stat

# Some Windows endpoint-security products point SSLKEYLOGFILE at a named pipe. OpenSSL then
# tries to open it through a C runtime it was not linked against and aborts the process
# ("OPENSSL_Uplink: no OPENSSL_Applink") the moment the first TLS context is created, which
# langchain_openai does at import. A key log that is not a regular file cannot be what the
# user meant to write, so it is dropped before anything imports ssl. Windows reports a pipe
# such as \\.\aswMonFltProxy\... as a regular file, so device paths are refused by name.
_keylog = os.environ.get("SSLKEYLOGFILE")
if _keylog:
    try:
        _regular = stat.S_ISREG(os.stat(_keylog).st_mode)
    except OSError:
        _regular = False
    if not _regular or _keylog.startswith(("\\\\.\\", "\\\\?\\")):
        del os.environ["SSLKEYLOGFILE"]

__version__ = "0.1.0"
