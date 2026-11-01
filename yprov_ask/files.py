"""Open files: the single provenance file a conversation is about.

The yProv Explorer shows one file at a time, loaded from the user's computer or from a URL,
and questions are about that file only. The server keeps each file it is given (ingested
once per content) and hands the client a **file id**: a random token, so one user cannot
reach another's upload by guessing a name. A question carries the file id (or the URL), and
the agent's tools are then pinned to that one document.

Uploads arrive over HTTP (``POST /files``, the raw PROV-JSON as the body, ``?name=`` for its
file name), because a file of several megabytes does not belong inside every question. They
are saved under the cache directory by name and content hash, so the same file uploaded
twice is one document.

The registry is in-process, like the conversations: forgotten after a day idle and on
restart; a client then gets "unknown file id" and uploads again.
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from .fetch import MAX_BYTES, FetchError, _prov_json

DEFAULT_IDLE_SECONDS = 24 * 3600


@dataclass
class OpenFile:
    file_id: str
    doc_id: str
    name: str
    origin: str  # "upload" or the URL it was fetched from
    last_used: float


class OpenFiles:
    def __init__(
        self,
        idle_seconds: float = DEFAULT_IDLE_SECONDS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.idle_seconds = idle_seconds
        self.clock = clock
        self._lock = threading.Lock()
        self._items: dict[str, OpenFile] = {}

    def open(self, doc_id: str, name: str, origin: str) -> OpenFile:
        with self._lock:
            self._forget_idle()
            item = OpenFile(uuid.uuid4().hex, doc_id, name, origin, self.clock())
            self._items[item.file_id] = item
            return item

    def get(self, file_id: str) -> OpenFile | None:
        with self._lock:
            self._forget_idle()
            item = self._items.get(file_id)
            if item is not None:
                item.last_used = self.clock()
            return item

    def _forget_idle(self) -> None:
        cutoff = self.clock() - self.idle_seconds
        for key in [k for k, v in self._items.items() if v.last_used < cutoff]:
            del self._items[key]


def save_upload(raw: bytes, name: str, directory: Path) -> Path:
    """Check an uploaded file is PROV-JSON and save it; returns where it was saved."""
    if len(raw) > MAX_BYTES:
        raise FetchError(f"the file is larger than {MAX_BYTES // (1024 * 1024)} MB")
    doc = _prov_json(raw, "the file")  # raises FetchError unless it is PROV-JSON
    if raw.lstrip()[:16].replace(b" ", b"").startswith(b'{"result"'):
        # The {"result": "<json>"} shape some store APIs return: keep the document itself.
        raw = json.dumps(doc).encode("utf-8")
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", Path(name).name.removesuffix(".json")).strip("._")
    digest = hashlib.sha256(raw).hexdigest()[:10]
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{(stem or 'upload')[:60]}-{digest}.json"
    if not path.exists():
        temp = path.with_suffix(".part")
        temp.write_bytes(raw)
        temp.replace(path)
    return path
