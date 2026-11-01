"""Provenance files fetched by URL: the file a user has open in the yProv Explorer.

The Explorer loads a file from a URL (or through an API) and keeps that URL in its own
address, as ``?file=...``. The widget sends it with each question; the server fetches the
file, ingests it, and tells the agent which document the user means.

Only hosts listed in YPROV_ASK_STORE_HOSTS may be fetched (comma-separated, e.g.
``store.yprov.disi.unitn.it`` or ``localhost:9000``). A server that fetched any URL a
browser sent would let anyone use it to reach addresses inside the cluster; redirects are
not followed for the same reason. A fetched file is saved in the cache directory under a
name derived from its URL, so fetching the same URL again replaces its earlier version in
the store instead of adding a copy.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from pathlib import Path
from urllib.parse import unquote, urlsplit

import httpx

MAX_BYTES = 20 * 1024 * 1024
TIMEOUT_SECONDS = 20.0

#: Extra attempts after a dropped or refused connection, and the pause before each.
RETRIES = 2
RETRY_DELAY_SECONDS = 0.5


class FetchError(Exception):
    """The file could not be fetched; the message says why, for the user."""


def allowed_hosts(value: str | None = None) -> list[str]:
    raw = os.environ.get("YPROV_ASK_STORE_HOSTS", "") if value is None else value
    return [h.strip().lower() for h in raw.split(",") if h.strip()]


def check_url(url: str, hosts: list[str]) -> None:
    parts = urlsplit(url)
    if parts.scheme not in ("http", "https") or not parts.hostname:
        raise FetchError(f"not an http(s) URL: {url}")
    if not hosts:
        raise FetchError(
            "this server fetches no files by URL: no store hosts are configured"
            " (YPROV_ASK_STORE_HOSTS)"
        )
    host = parts.hostname.lower()
    with_port = f"{host}:{parts.port}" if parts.port else host
    if host not in hosts and with_port not in hosts:
        raise FetchError(f"{parts.netloc} is not one of the store hosts: {', '.join(hosts)}")


def cache_path(url: str, directory: Path) -> Path:
    """A readable file name for ``url``, unique to it: ``<last path segment>-<hash>.json``."""
    segment = unquote(urlsplit(url).path.rstrip("/").rsplit("/", 1)[-1])
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", segment.removesuffix(".json")).strip("._")[:60]
    digest = hashlib.sha256(url.encode("utf-8")).hexdigest()[:10]
    return directory / f"{stem or 'store_file'}-{digest}.json"


def _prov_json(raw: bytes, source: str = "the URL's content") -> dict:
    """The PROV-JSON document in ``raw``; ``source`` names it in the error ("the file")."""
    try:
        doc = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise FetchError(f"{source} is not JSON") from exc
    # Some store APIs wrap the document as a JSON string in "result", as the Explorer's
    # "load from API" expects.
    if isinstance(doc, dict) and isinstance(doc.get("result"), str):
        try:
            doc = json.loads(doc["result"])
        except json.JSONDecodeError as exc:
            raise FetchError(f"the 'result' in {source} is not JSON") from exc
    if not isinstance(doc, dict) or not any(k in doc for k in ("entity", "activity", "agent")):
        raise FetchError(
            f"{source} is not a PROV-JSON document (no entity, activity or agent section)"
        )
    return doc


def _download(http: httpx.Client, url: str) -> bytes:
    with http.stream("GET", url, headers={"Accept": "application/json"}) as response:
        if response.is_redirect:
            raise FetchError(f"{url} redirects elsewhere; redirects are not followed")
        if response.status_code != 200:
            raise FetchError(f"{url} answered HTTP {response.status_code}")
        body = bytearray()
        for chunk in response.iter_bytes():
            body += chunk
            if len(body) > MAX_BYTES:
                raise FetchError(f"{url} is larger than {MAX_BYTES // (1024 * 1024)} MB")
    return bytes(body)


def fetch(url: str, hosts: list[str], directory: Path, client: httpx.Client | None = None) -> Path:
    """Fetch the PROV-JSON document at ``url`` into ``directory``; returns the file."""
    check_url(url, hosts)
    own = client is None
    http = client or httpx.Client(timeout=TIMEOUT_SECONDS, follow_redirects=False)
    try:
        for attempt in range(1 + RETRIES):
            try:
                body = _download(http, url)
                break
            except httpx.TransportError as exc:
                # A dropped or refused connection may be momentary; an HTTP answer is not
                # retried, since asking again gets the same answer.
                if attempt == RETRIES:
                    raise FetchError(f"could not fetch {url}: {exc}") from exc
                time.sleep(RETRY_DELAY_SECONDS)
            except httpx.HTTPError as exc:
                raise FetchError(f"could not fetch {url}: {exc}") from exc
    finally:
        if own:
            http.close()

    doc = _prov_json(body)
    directory.mkdir(parents=True, exist_ok=True)
    path = cache_path(url, directory)
    temp = path.with_suffix(".part")
    temp.write_text(json.dumps(doc, indent=1), encoding="utf-8")
    temp.replace(path)
    return path
