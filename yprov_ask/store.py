"""The provenance store: PROV-JSON files flattened into SQLite.

A PROV-JSON document is a dictionary of sections (``entity``, ``activity``, ``agent``, and
one per relation type) keyed by identifier. Questions cut across documents ("which run had
the lowest validation loss?"), so the store keeps every ingested document in three tables
rather than as files:

    documents   one row per file: path, content hash, kind, counts
    nodes       entities, activities and agents, with their attributes as JSON
    relations   used, wasGeneratedBy, wasDerivedFrom, ... as (subject, object) pairs

yProv4ML writes each logged metric as a CSV beside the PROV file and refers to it from a
``provml:Metric`` entity. Where that CSV can be found, its points go into ``metric_points``,
so a question about a loss curve is answered from the numbers, not from the file name.

Ingestion is idempotent by content: the same bytes ingested twice are one document, and a
changed file replaces its earlier version.
"""

from __future__ import annotations

import csv
import hashlib
import json
import sqlite3
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

#: Sections that hold nodes, and the kind each is stored as.
NODE_SECTIONS = {"entity": "entity", "activity": "activity", "agent": "agent"}

#: Relation sections, and the attribute names of their two ends. The first end is the
#: "subject" (the thing that depends), the second the "object" (what it depends on), so a
#: walk upstream always goes subject -> object.
RELATIONS: dict[str, tuple[str, str]] = {
    "used": ("prov:activity", "prov:entity"),
    "wasGeneratedBy": ("prov:entity", "prov:activity"),
    "wasDerivedFrom": ("prov:generatedEntity", "prov:usedEntity"),
    "wasInformedBy": ("prov:informed", "prov:informant"),
    "wasAssociatedWith": ("prov:activity", "prov:agent"),
    "wasAttributedTo": ("prov:entity", "prov:agent"),
    "actedOnBehalfOf": ("prov:delegate", "prov:responsible"),
    "wasStartedBy": ("prov:activity", "prov:trigger"),
    "wasEndedBy": ("prov:activity", "prov:trigger"),
    "wasInvalidatedBy": ("prov:entity", "prov:activity"),
    "specializationOf": ("prov:specificEntity", "prov:generalEntity"),
    "alternateOf": ("prov:alternate1", "prov:alternate2"),
    "hadMember": ("prov:collection", "prov:entity"),
}

_SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    doc_id      TEXT PRIMARY KEY,
    path        TEXT NOT NULL UNIQUE,
    sha256      TEXT NOT NULL,
    kind        TEXT NOT NULL,
    title       TEXT NOT NULL,
    ingested_at TEXT NOT NULL,
    n_entities  INTEGER NOT NULL,
    n_activities INTEGER NOT NULL,
    n_agents    INTEGER NOT NULL,
    n_relations INTEGER NOT NULL,
    raw         TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS nodes (
    doc_id   TEXT NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
    node_id  TEXT NOT NULL,
    kind     TEXT NOT NULL,
    label    TEXT,
    prov_type TEXT,
    attrs    TEXT NOT NULL,
    PRIMARY KEY (doc_id, node_id, kind)
);
CREATE TABLE IF NOT EXISTS relations (
    doc_id   TEXT NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
    rel_id   TEXT NOT NULL,
    rel_type TEXT NOT NULL,
    subject  TEXT,
    object   TEXT,
    attrs    TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS metric_points (
    doc_id   TEXT NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
    node_id  TEXT NOT NULL,
    epoch    INTEGER,
    step     INTEGER,
    value    REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS relations_subject ON relations(doc_id, subject);
CREATE INDEX IF NOT EXISTS relations_object ON relations(doc_id, object);
CREATE INDEX IF NOT EXISTS metric_points_node ON metric_points(doc_id, node_id);
"""


@dataclass(frozen=True)
class IngestResult:
    doc_id: str
    path: str
    kind: str
    status: str  # "added", "updated" or "unchanged"
    metric_series: int


def plain(value: Any) -> Any:
    """A PROV-JSON attribute value without its typing wrapper.

    PROV-JSON writes typed literals two ways: ``{"$": 0, "type": "xsd:int"}`` and, in
    yProv4ML, ``"4958085^^xsd:double"``. A question about a value wants the value.
    """
    if isinstance(value, dict) and "$" in value:
        return plain(value["$"])
    if isinstance(value, list):
        return [plain(v) for v in value]
    if isinstance(value, str) and "^^" in value:
        literal, _, datatype = value.rpartition("^^")
        if datatype in ("xsd:double", "xsd:float", "xsd:decimal"):
            try:
                return float(literal)
            except ValueError:
                return literal
        if datatype in ("xsd:int", "xsd:integer", "xsd:long"):
            try:
                return int(literal)
            except ValueError:
                return literal
        return literal
    return value


def _attrs(raw: dict[str, Any]) -> dict[str, Any]:
    return {key: plain(value) for key, value in raw.items()}


def descriptions(raw: Any) -> list[dict[str, Any]]:
    """The attribute sets PROV-JSON gives for one identifier: usually one dictionary, but
    the format allows a list of them when the same node or relation is described more than
    once (Ophidia writes its datacubes this way)."""
    if isinstance(raw, dict):
        return [raw]
    if isinstance(raw, list):
        return [d for d in raw if isinstance(d, dict)]
    return []


def merge(parts: list[dict[str, Any]]) -> dict[str, Any]:
    """One node's descriptions as one: a key with a single value keeps it; a key whose
    descriptions disagree keeps every distinct value, in order, as a list."""
    merged: dict[str, Any] = {}
    for part in parts:
        for key, value in part.items():
            if key not in merged:
                merged[key] = value
                continue
            values = merged[key] if isinstance(merged[key], _Several) else _Several([merged[key]])
            if value not in values:
                values.append(value)
                merged[key] = values
    return {k: list(v) if isinstance(v, _Several) else v for k, v in merged.items()}


class _Several(list):
    """Values gathered for one key while merging (told apart from a list-valued attribute)."""


#: What produced a document, by the vocabulary it uses; the kinds list_documents reports.
KINDS = {
    "qa": "an answer record of this agent",
    "yprov4agents": "yProv4Agents: agents' tasks and decisions",
    "yprov4ml": "yProv4ML: a machine-learning run",
    "yprov4wfs": "yProv4WFs: a workflow run, one activity per task (e.g. dask)",
    "openeo": "yProv4WFs-openEO: an openEO process graph",
    "ophidia": "Ophidia: a run of Ophidia operators",
    "analytics4yprov": "analytics4yprov: a workflow graph reduced to pipelines",
    "prov": "W3C PROV from another producer",
}


def classify(doc: dict[str, Any]) -> str:
    """What produced a document (one of ``KINDS``), read from the vocabulary it uses.

    The producers declare their namespaces; yProv4WFs-openEO writes the yProv4WFs namespace
    and tells itself apart by the engine of its workflow activity (Openeo-LocalProcessing);
    analytics4yprov adds its own namespace to a yProv4WFs graph. An answer record of this
    agent says it was emitted by yprov-ask.
    """
    for raw in (doc.get("entity") or {}).values():
        if any(d.get("yp:emitted_by") == "yprov-ask" for d in descriptions(raw)):
            return "qa"
    prefixes = doc.get("prefix") or {}
    namespaces = " ".join(str(v) for v in prefixes.values())
    if "yProv4Agents" in namespaces or "yp" in prefixes:
        return "yprov4agents"
    if "yProv4ML" in namespaces or "provml" in prefixes:
        return "yprov4ml"
    if "ophidia" in prefixes:
        return "ophidia"
    if "analytics4yprov" in prefixes:
        return "analytics4yprov"
    if "yprov4wfs" in prefixes:
        for raw in (doc.get("activity") or {}).values():
            if any(
                str(d.get("yprov4wfs:engine", "")).lower().startswith("openeo")
                for d in descriptions(raw)
            ):
                return "openeo"
        return "yprov4wfs"
    return "prov"


def _title(doc: dict[str, Any], kind: str, path: Path) -> str:
    if kind == "qa":
        return f"ProvenanceQA answer ({path.stem})"
    named = {
        "yprov4wfs": "yProv4WFs workflow run",
        "openeo": "openEO process graph",
        "ophidia": "Ophidia workflow run",
        "analytics4yprov": "analytics4yprov reduced graph",
    }
    if kind in named:
        return f"{named[kind]} ({path.stem})"
    if kind == "yprov4ml":
        name = (doc.get("prefix") or {}).get("default")
        if name:
            return f"yProv4ML run {name}"
    if kind == "yprov4agents":
        for raw in (doc.get("agent") or {}).values():
            label = plain(merge(descriptions(raw)).get("prov:label"))
            if label:
                return f"{label} record ({path.stem})"
    return path.stem


def _metric_csv(json_path: Path, identifier: str) -> Path | None:
    """Where a yProv4ML metric file is on this machine, if it is anywhere.

    yProv4ML records the path relative to the directory it wrote into
    (``prov/example_0/metrics_GR0/x.csv``), which is rarely where the files end up. The
    file is looked for under each ancestor of the PROV file, with and without that first
    component.
    """
    rel = Path(identifier)
    if rel.is_absolute():
        return rel if rel.is_file() else None
    candidates = [rel]
    if len(rel.parts) > 1:
        candidates.append(Path(*rel.parts[1:]))
    for base in [json_path.parent, *json_path.parent.parents]:
        for candidate in candidates:
            path = base / candidate
            if path.is_file():
                return path
    return None


def _read_series(path: Path) -> Iterator[tuple[int | None, int | None, float]]:
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            try:
                value = float(row["value"])
            except (KeyError, TypeError, ValueError):
                continue
            epoch = row.get("epoch")
            step = row.get("timestep") or row.get("step")
            yield (
                int(float(epoch)) if epoch not in (None, "") else None,
                int(float(step)) if step not in (None, "") else None,
                value,
            )


class ProvStore:
    """SQLite-backed store of ingested PROV-JSON documents."""

    def __init__(self, path: str | Path = ":memory:") -> None:
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        # The MCP server answers from a worker thread; reads and writes are serialised by
        # SQLite itself, so sharing one connection across threads is safe here.
        self.db = sqlite3.connect(self.path, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA foreign_keys = ON")
        self.db.executescript(_SCHEMA)
        # Stores written before answer records had a kind of their own hold
        # them as yprov4agents documents; give them their kind, so tools can leave them out.
        with self.db:
            self.db.execute(
                "UPDATE documents SET kind = 'qa'"
                " WHERE kind = 'yprov4agents' AND title LIKE 'ProvenanceQA record (%'"
            )

    def close(self) -> None:
        self.db.close()

    # ── ingestion ─────────────────────────────────────────────────────────────────────

    def ingest_path(self, path: str | Path) -> list[IngestResult]:
        """Ingest a PROV-JSON file, or every ``*.json`` file under a directory."""
        root = Path(path).expanduser().resolve()
        if root.is_dir():
            files: Iterable[Path] = sorted(p for p in root.rglob("*.json") if p.is_file())
        elif root.is_file():
            files = [root]
        else:
            raise FileNotFoundError(f"no file or directory at {root}")
        results = []
        for file in files:
            result = self.ingest_file(file)
            if result is not None:
                results.append(result)
        return results

    def ingest_file(self, path: Path) -> IngestResult | None:
        """Ingest one file. Returns None for JSON that is not a PROV document."""
        raw = path.read_bytes()
        try:
            doc = json.loads(raw)
        except (json.JSONDecodeError, UnicodeDecodeError):
            return None
        if not isinstance(doc, dict) or not any(k in doc for k in NODE_SECTIONS):
            return None

        sha = hashlib.sha256(raw).hexdigest()
        resolved = str(path.resolve())
        existing = self.db.execute(
            "SELECT doc_id, sha256 FROM documents WHERE path = ?", (resolved,)
        ).fetchone()
        if existing and existing["sha256"] == sha:
            kind = classify(doc)
            stored = self.document(existing["doc_id"])
            if stored is not None and stored["kind"] != kind:
                # Ingested before this kind was recognised: correct the kind and the title.
                with self.db:
                    self.db.execute(
                        "UPDATE documents SET kind = ?, title = ? WHERE doc_id = ?",
                        (kind, _title(doc, kind, path), existing["doc_id"]),
                    )
            n_series = self.db.execute(
                "SELECT COUNT(DISTINCT node_id) FROM metric_points WHERE doc_id = ?",
                (existing["doc_id"],),
            ).fetchone()[0]
            return IngestResult(existing["doc_id"], resolved, classify(doc), "unchanged", n_series)

        status = "updated" if existing else "added"
        with self.db:
            if existing:
                self.db.execute("DELETE FROM documents WHERE doc_id = ?", (existing["doc_id"],))
            doc_id = self._new_doc_id(path)
            n_series = self._insert(doc_id, path, resolved, sha, doc)
        return IngestResult(doc_id, resolved, classify(doc), status, n_series)

    def _new_doc_id(self, path: Path) -> str:
        """A short, readable id: the file stem, made unique if two files share it."""
        base = path.stem
        doc_id, n = base, 2
        while self.db.execute("SELECT 1 FROM documents WHERE doc_id = ?", (doc_id,)).fetchone():
            doc_id, n = f"{base}~{n}", n + 1
        return doc_id

    def _insert(self, doc_id: str, path: Path, resolved: str, sha: str, doc: dict[str, Any]) -> int:
        kind = classify(doc)
        counts = {
            section: len(doc.get(section) or {}) for section in ("entity", "activity", "agent")
        }
        # A relation id given a list of descriptions is several relations.
        n_relations = sum(
            len(descriptions(raw))
            for section in RELATIONS
            for raw in (doc.get(section) or {}).values()
        )
        self.db.execute(
            "INSERT INTO documents VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (
                doc_id,
                resolved,
                sha,
                kind,
                _title(doc, kind, path),
                datetime.now(UTC).isoformat(timespec="seconds"),
                counts["entity"],
                counts["activity"],
                counts["agent"],
                n_relations,
                json.dumps(doc),
            ),
        )
        n_series = 0
        for section, node_kind in NODE_SECTIONS.items():
            for node_id, raw_attrs in (doc.get(section) or {}).items():
                attrs = _attrs(merge(descriptions(raw_attrs)))
                prov_type = attrs.get("prov:type")
                self.db.execute(
                    "INSERT OR REPLACE INTO nodes VALUES (?,?,?,?,?,?)",
                    (
                        doc_id,
                        node_id,
                        node_kind,
                        str(attrs.get("prov:label")) if "prov:label" in attrs else None,
                        str(prov_type) if prov_type is not None else None,
                        json.dumps(attrs, default=str),
                    ),
                )
                if prov_type == "provml:Metric" and attrs.get("dcterms:identifier"):
                    csv_path = _metric_csv(path, str(attrs["dcterms:identifier"]))
                    if csv_path is not None:
                        points = list(_read_series(csv_path))
                        self.db.executemany(
                            "INSERT INTO metric_points VALUES (?,?,?,?,?)",
                            [(doc_id, node_id, e, s, v) for e, s, v in points],
                        )
                        n_series += bool(points)
        for section, (subject_key, object_key) in RELATIONS.items():
            for rel_id, raw_attrs in (doc.get(section) or {}).items():
                for attrs in map(_attrs, descriptions(raw_attrs)):
                    self.db.execute(
                        "INSERT INTO relations VALUES (?,?,?,?,?,?)",
                        (
                            doc_id,
                            rel_id,
                            section,
                            attrs.get(subject_key),
                            attrs.get(object_key),
                            json.dumps(attrs, default=str),
                        ),
                    )
        return n_series

    def remove(self, doc_id: str) -> bool:
        with self.db:
            cursor = self.db.execute("DELETE FROM documents WHERE doc_id = ?", (doc_id,))
        return cursor.rowcount > 0

    # ── reads used by tools.py ────────────────────────────────────────────────────────

    def documents(self) -> list[sqlite3.Row]:
        return self.db.execute(
            "SELECT doc_id, path, sha256, kind, title, ingested_at, n_entities, n_activities,"
            " n_agents, n_relations FROM documents ORDER BY doc_id"
        ).fetchall()

    def document(self, doc_id: str) -> sqlite3.Row | None:
        return self.db.execute("SELECT * FROM documents WHERE doc_id = ?", (doc_id,)).fetchone()

    def nodes(self, doc_id: str | None = None, kind: str | None = None) -> list[sqlite3.Row]:
        sql, args = "SELECT * FROM nodes WHERE 1=1", []
        if doc_id is not None:
            sql += " AND doc_id = ?"
            args.append(doc_id)
        if kind is not None:
            sql += " AND kind = ?"
            args.append(kind)
        return self.db.execute(sql + " ORDER BY doc_id, kind, node_id", args).fetchall()

    def node(self, doc_id: str, node_id: str) -> list[sqlite3.Row]:
        return self.db.execute(
            "SELECT * FROM nodes WHERE doc_id = ? AND node_id = ?", (doc_id, node_id)
        ).fetchall()

    def relations_of(self, doc_id: str, node_id: str) -> list[sqlite3.Row]:
        return self.db.execute(
            "SELECT * FROM relations WHERE doc_id = ? AND (subject = ? OR object = ?)"
            " ORDER BY rel_type, rel_id",
            (doc_id, node_id, node_id),
        ).fetchall()

    def relations(self, doc_id: str) -> list[sqlite3.Row]:
        return self.db.execute(
            "SELECT * FROM relations WHERE doc_id = ? ORDER BY rel_type, rel_id", (doc_id,)
        ).fetchall()

    def series(self, doc_id: str, node_id: str) -> list[sqlite3.Row]:
        return self.db.execute(
            "SELECT epoch, step, value FROM metric_points WHERE doc_id = ? AND node_id = ?"
            " ORDER BY rowid",
            (doc_id, node_id),
        ).fetchall()
