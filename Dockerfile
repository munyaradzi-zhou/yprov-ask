# The MCP server over streamable HTTP, with the chat widget, on one port.
#
#   docker build --build-arg VERSION=0.1.0 -t yprov-ask:0.1.0 .
#   docker run --rm -p 8000:8000 -e OPENAI_API_KEY \
#       -e YPROV_ASK_CORS=https://explorer.yprov.disi.unitn.it yprov-ask:0.1.0
#
#   http://localhost:8000/mcp            MCP endpoint (the widget's endpoint attribute)
#   http://localhost:8000/widget/        the widget in an Explorer-like demo page
#   http://localhost:8000/widget/yprov-ask.js   the script the Explorer loads
#   http://localhost:8000/healthz        liveness, for the container runtime
#
# Configuration is by environment only (see yprov_ask/server.py): OPENAI_API_KEY,
# YPROV_ASK_MODEL, YPROV_ASK_CORS (browser origins allowed to call /mcp),
# YPROV_ASK_STORE_HOSTS (hosts a file_url may be fetched from). Files the user opens
# arrive by POST /files or by URL; the samples are ingested only for questions with no file.
# Everything the server writes (store, answer records, fetched files) is under /tmp: it
# runs as an unprivileged user and keeps no state across restarts.
FROM python:3.12.10-slim

ARG VERSION=dev
LABEL org.opencontainers.image.title="yprov-ask" \
      org.opencontainers.image.version=$VERSION \
      org.opencontainers.image.description="Answers questions about W3C PROV files, over MCP"

RUN pip install --no-cache-dir uv==0.7.2
WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-install-project
COPY yprov_ask ./yprov_ask
COPY samples ./samples
RUN uv sync --frozen

ENV YPROV_ASK_INGEST=/app/samples \
    YPROV_ASK_DB=/tmp/provqa.sqlite \
    YPROV_ASK_RECORDS=/tmp/records \
    PATH=/app/.venv/bin:$PATH
EXPOSE 8000
USER nobody
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=4)"]
CMD ["python", "-m", "yprov_ask", "serve", "--http", "--host", "0.0.0.0", "--port", "8000"]
