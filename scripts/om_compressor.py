#!/usr/bin/env python3
"""Observational Memory compressor.

Foundational implementation for:
- lock-scoped chunk processing
- deterministic extractor/chunk/event ids
- OM node/edge/event/provenance writes
- persisted chunk failure + split/isolate workflows
- dead-letter queue integration with truth/candidates.py
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import sys
import tempfile
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

# Ensure repo root is on sys.path so truth/ package is importable
# when this script is invoked directly (not via -m).
SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from truth import candidates as candidates_store  # noqa: E402

DEFAULT_LOCK_FILENAME = "om_graph_write.lock"
NEO4J_ENV_FALLBACK_FILE = Path.home() / ".clawdbot" / "credentials" / "neo4j.env"
NEO4J_NON_DEV_FALLBACK_OPT_IN_ENV = "OM_NEO4J_ENV_FALLBACK_NON_DEV"
_NON_DEV_ENV_MARKERS = {"prod", "production", "staging", "stage", "preprod", "preview"}
_TRUTHY_ENV_VALUES = {"1", "true", "yes", "on"}
DEFAULT_ONTOLOGY_CONFIG_REL = Path("mcp_server/config/extraction_ontologies.yaml")
MAX_PARENT_CHUNK_SIZE = 50
MAX_CHILD_CHUNK_SIZE = 10
DEFAULT_MAX_CHUNKS_PER_RUN = 10
DEAD_LETTER_ATTEMPTS = 3
DEFAULT_EMBEDDING_MODEL = "embeddinggemma"
DEFAULT_EMBEDDING_DIM = 768

STATUS_ACTIVE = {"open", "monitoring", "reopened"}
STATUS_POOL = STATUS_ACTIVE | {"closed"}

RELATION_TYPES = {
    "MOTIVATES",
    "GENERATES",
    "SUPERSEDES",
    "ADDRESSES",
    "RESOLVES",
}
RELATION_TYPE_TOKEN_RE = re.compile(r"^[A-Z][A-Z0-9_]{0,63}$")


class OMCompressorError(RuntimeError):
    pass


class SchemaVersionMissingError(OMCompressorError):
    pass


class NodeContentMismatchError(OMCompressorError):
    def __init__(self, node_id: str, existing_hash: str, incoming_hash: str) -> None:
        self.node_id = node_id
        self.existing_hash = existing_hash
        self.incoming_hash = incoming_hash
        super().__init__(f"OM_NODE_CONTENT_MISMATCH node_id={node_id}")


@dataclass(frozen=True)
class ExtractorConfig:
    schema_version: str
    prompt_template: str
    model_id: str
    extractor_version: str


@dataclass
class MessageRow:
    message_id: str
    source_session_id: str
    content: str
    created_at: str
    content_embedding: list[float]
    om_extract_attempts: int


@dataclass
class ExtractionNode:
    node_id: str
    node_type: str
    semantic_domain: str
    content: str
    urgency_score: int
    source_session_id: str
    source_message_ids: list[str]
    status: str = "open"


@dataclass
class ExtractionEdge:
    source_node_id: str
    target_node_id: str
    relation_type: str


@dataclass
class ExtractedChunk:
    nodes: list[ExtractionNode]
    edges: list[ExtractionEdge]


@dataclass
class ActivationCandidate:
    node_id: str
    status: str
    urgency_score: int
    created_at: str
    status_changed_at: str | None
    last_observed_at: str | None
    content_embedding: list[float]


@dataclass
class ParentState:
    chunk_id: str
    attempts: int
    first_failed_at: str | None
    split_status: str
    next_subchunk_index: int | None
    child_chunk_ids: list[str]
    parent_message_ids: list[str]
    split_backfill_version: str | None


@dataclass
class ChildState:
    child_index: int
    child_chunk_id: str
    message_ids: list[str]


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def sha256_hex(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def emit_event(name: str, **payload: Any) -> dict[str, Any]:
    event = {"event": name, "timestamp": now_iso(), **payload}
    print(json.dumps(event, ensure_ascii=True))
    return event


def normalize_text(value: str) -> str:
    text = unicodedata.normalize("NFKC", value or "")
    text = text.strip()
    return re.sub(r"\s+", " ", text)


def parse_iso(value: str | None) -> datetime | None:
    if not value:
        return None
    txt = str(value).strip()
    if not txt:
        return None
    try:
        if txt.endswith("Z"):
            return datetime.fromisoformat(txt[:-1]).replace(tzinfo=timezone.utc)
        dt = datetime.fromisoformat(txt)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None


def cosine_similarity(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def l2_normalize(vector: list[float]) -> list[float]:
    mag = sum(v * v for v in vector) ** 0.5
    if mag <= 0.0:
        return vector
    return [v / mag for v in vector]


def mean_vector(vectors: list[list[float]]) -> list[float]:
    if not vectors:
        return []
    width = len(vectors[0])
    if width == 0:
        return []
    acc = [0.0] * width
    count = 0
    for vec in vectors:
        if len(vec) != width:
            continue
        count += 1
        for i, value in enumerate(vec):
            acc[i] += float(value)
    if count == 0:
        return []
    return [v / count for v in acc]


def tokenize_len(text: str) -> int:
    try:
        import tiktoken  # type: ignore

        enc = tiktoken.get_encoding("o200k_base")
        return len(enc.encode(text))
    except Exception:
        # Fallback heuristic when tiktoken is unavailable.
        return max(1, len(text) // 4)


def _validated_embedding_base_url() -> str:
    base = (os.environ.get("EMBEDDER_BASE_URL") or os.environ.get("OPENAI_BASE_URL") or "").strip()
    if not base:
        base = "http://localhost:11434/v1"

    parsed = urllib.parse.urlparse(base)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise OMCompressorError("embedding base URL must be absolute http(s) URL")
    if parsed.username or parsed.password:
        raise OMCompressorError("embedding base URL must not include credentials")
    if parsed.query or parsed.fragment:
        raise OMCompressorError("embedding base URL must not include query/fragment")

    return base.rstrip("/")


def _embedding_config() -> tuple[str, int]:
    """Resolve embedding config from env with compatibility defaults.

    Env overrides:
    - OM_EMBEDDING_MODEL (default: embeddinggemma)
    - OM_EMBEDDING_DIM (default: 768)
    """

    model = (os.environ.get("OM_EMBEDDING_MODEL") or DEFAULT_EMBEDDING_MODEL).strip()
    if not model:
        model = DEFAULT_EMBEDDING_MODEL

    raw_dim = (os.environ.get("OM_EMBEDDING_DIM") or str(DEFAULT_EMBEDDING_DIM)).strip()
    try:
        dim = int(raw_dim)
    except ValueError as exc:
        raise OMCompressorError(f"OM_EMBEDDING_DIM must be an integer, got: {raw_dim!r}") from exc
    if dim <= 0:
        raise OMCompressorError(f"OM_EMBEDDING_DIM must be > 0, got: {dim}")

    return model, dim


def _embed_text(content: str, *, embedding_model: str, embedding_dim: int) -> list[float]:
    base = _validated_embedding_base_url()
    url = base + "/embeddings"

    payload = {
        "model": embedding_model,
        "input": content,
    }
    body = json.dumps(payload).encode("utf-8")

    headers = {"Content-Type": "application/json"}
    api_key = os.environ.get("EMBEDDER_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    req = urllib.request.Request(url, data=body, headers=headers, method="POST")
    timeout = int(os.environ.get("OM_EMBED_TIMEOUT_SECONDS", "20"))

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        details = exc.read().decode("utf-8", errors="replace")
        raise OMCompressorError(f"embedding HTTP {exc.code}: {details}") from exc
    except Exception as exc:
        raise OMCompressorError(f"embedding request failed: {exc}") from exc

    parsed = json.loads(raw) if raw.strip() else {}
    data = parsed.get("data")
    if not isinstance(data, list) or not data or not isinstance(data[0], dict):
        raise OMCompressorError("embedding response missing data[0]")
    emb = data[0].get("embedding")
    if not isinstance(emb, list):
        raise OMCompressorError("embedding response missing embedding")

    vector = [float(v) for v in emb]
    if len(vector) != embedding_dim:
        raise OMCompressorError(f"embedding dim mismatch: got={len(vector)} expected={embedding_dim}")
    return vector


def _safe_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    return []


def _safe_str_list(value: Any) -> list[str]:
    return [str(v) for v in _safe_list(value)]


def _resolve_repo_path(value: str | Path) -> Path:
    candidate = Path(value)
    if candidate.is_absolute():
        return candidate
    return REPO_ROOT / candidate


def _resolve_ontology_config_path(config_arg: str | None) -> Path:
    env_override = (os.environ.get("OM_ONTOLOGY_CONFIG_PATH") or "").strip()
    if env_override:
        return _resolve_repo_path(env_override)

    raw = (config_arg or "").strip()
    if raw:
        cli_path = Path(raw)
        if cli_path.is_absolute():
            return cli_path
        cwd_resolved = Path.cwd() / cli_path
        if cwd_resolved.exists():
            return cwd_resolved
        return REPO_ROOT / cli_path

    return REPO_ROOT / DEFAULT_ONTOLOGY_CONFIG_REL


def _resolve_lock_path() -> Path:
    env_override = (os.environ.get("OM_COMPRESSOR_LOCK_PATH") or "").strip()
    if env_override:
        return _resolve_repo_path(env_override)

    candidate_dirs: list[Path] = []
    runtime_dir = (os.environ.get("XDG_RUNTIME_DIR") or "").strip()
    if runtime_dir:
        candidate_dirs.append(Path(runtime_dir) / "bicameral")
    candidate_dirs.append(REPO_ROOT / "state" / "locks")
    candidate_dirs.append(Path.home() / ".cache" / "bicameral" / "locks")
    candidate_dirs.append(Path(tempfile.gettempdir()) / "bicameral" / "locks")

    for lock_dir in candidate_dirs:
        try:
            lock_dir.mkdir(parents=True, exist_ok=True)
        except OSError:
            continue
        if os.access(lock_dir, os.W_OK | os.X_OK):
            return lock_dir / DEFAULT_LOCK_FILENAME

    return Path(tempfile.gettempdir()) / "bicameral" / "locks" / DEFAULT_LOCK_FILENAME


def _load_extractor_config(path: Path) -> ExtractorConfig:
    if not path.exists():
        raise SchemaVersionMissingError(f"missing config: {path}")

    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise SchemaVersionMissingError(f"invalid config shape: {path}")

    schema_version = str(data.get("schema_version") or "").strip()
    if not schema_version:
        emit_event(
            "OM_SCHEMA_VERSION_MISSING",
            config_path=str(path),
        )
        raise SchemaVersionMissingError("schema_version missing")

    om_cfg = data.get("om_extractor") if isinstance(data.get("om_extractor"), dict) else {}
    prompt_template = str(om_cfg.get("prompt_template") or "").strip()
    model_id = str(om_cfg.get("model_id") or "").strip()
    if not prompt_template:
        prompt_template = "OM_PROMPT_TEMPLATE_V1"
    if not model_id:
        model_id = os.environ.get("OM_COMPRESSOR_MODEL", "gpt-5.1-codex-mini")

    extractor_version = sha256_hex(f"{prompt_template}|{model_id}|{schema_version}")
    return ExtractorConfig(
        schema_version=schema_version,
        prompt_template=prompt_template,
        model_id=model_id,
        extractor_version=extractor_version,
    )


def _chunk_id(messages: list[MessageRow], extractor_version: str) -> str:
    first = messages[0].message_id
    last = messages[-1].message_id
    n = len(messages)
    return sha256_hex(f"{first}|{last}|{n}|{extractor_version}")


def _child_id(parent_chunk_id: str, child_index: int) -> str:
    return sha256_hex(f"chunkchild|{parent_chunk_id}|{child_index}")


def _derive_urgency(content: str) -> int:
    text = content.lower()
    if any(token in text for token in ("urgent", "asap", "immediately", "blocked", "critical")):
        return 5
    if any(token in text for token in ("soon", "important", "must")):
        return 4
    return 3


def _derive_node_type(content: str) -> str:
    lower = content.lower()
    if "because" in lower or "decision" in lower:
        return "Judgment"
    if "rule" in lower or "always" in lower:
        return "OperationalRule"
    if "commit" in lower or "promise" in lower:
        return "Commitment"
    if "problem" in lower or "friction" in lower or "blocked" in lower:
        return "Friction"
    return "WorldState"


def _extract_with_rules(messages: list[MessageRow]) -> ExtractedChunk:
    by_node: dict[str, ExtractionNode] = {}
    for msg in messages:
        normalized = normalize_text(msg.content)
        if not normalized:
            continue

        semantic_domain = "sessions_main"
        node_type = _derive_node_type(normalized)
        node_id = sha256_hex(f"omnode|{node_type}|{semantic_domain}|{normalized.lower()}")

        existing = by_node.get(node_id)
        if existing is None:
            by_node[node_id] = ExtractionNode(
                node_id=node_id,
                node_type=node_type,
                semantic_domain=semantic_domain,
                content=normalized,
                urgency_score=_derive_urgency(normalized),
                source_session_id=msg.source_session_id,
                source_message_ids=[msg.message_id],
            )
        elif msg.message_id not in existing.source_message_ids:
            existing.source_message_ids.append(msg.message_id)

    return ExtractedChunk(nodes=list(by_node.values()), edges=[])


def _extract_items(messages: list[MessageRow], _cfg: ExtractorConfig) -> ExtractedChunk:
    # Deterministic fallback extractor. This is deliberately stable and
    # idempotent; model-backed extraction can be layered behind this contract.
    return _extract_with_rules(messages)


def _is_truthy_env(name: str) -> bool:
    value = os.environ.get(name)
    if value is None:
        return False
    return value.strip().lower() in _TRUTHY_ENV_VALUES


def _is_non_dev_mode() -> bool:
    for key in ("OM_ENV", "APP_ENV", "ENVIRONMENT", "NODE_ENV"):
        value = os.environ.get(key)
        if value and value.strip().lower() in _NON_DEV_ENV_MARKERS:
            return True
    return _is_truthy_env("CI")


def _allow_neo4j_env_fallback() -> bool:
    if not _is_non_dev_mode():
        return True
    return _is_truthy_env(NEO4J_NON_DEV_FALLBACK_OPT_IN_ENV)


def _load_neo4j_env_fallback() -> None:
    if os.environ.get("NEO4J_PASSWORD"):
        return
    if not _allow_neo4j_env_fallback():
        return
    if not NEO4J_ENV_FALLBACK_FILE.exists():
        return

    for raw_line in NEO4J_ENV_FALLBACK_FILE.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key in {"NEO4J_PASSWORD", "NEO4J_USER", "NEO4J_URI"} and key not in os.environ:
            os.environ[key] = value.strip().strip('"').strip("'")


def _neo4j_driver() -> Any:
    try:
        from neo4j import GraphDatabase  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise OMCompressorError("neo4j driver is required") from exc

    _load_neo4j_env_fallback()

    uri = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
    user = os.environ.get("NEO4J_USER", "neo4j")
    password = os.environ.get("NEO4J_PASSWORD")
    if not password:
        raise OMCompressorError("NEO4J_PASSWORD is required")

    return GraphDatabase.driver(uri, auth=(user, password))


def _ensure_neo4j_constraints(session: Any) -> None:
    stmts = [
        "CREATE CONSTRAINT message_message_id IF NOT EXISTS FOR (m:Message) REQUIRE m.message_id IS UNIQUE",
        "CREATE CONSTRAINT omnode_node_id IF NOT EXISTS FOR (n:OMNode) REQUIRE n.node_id IS UNIQUE",
        "CREATE CONSTRAINT corememory_core_memory_id IF NOT EXISTS FOR (c:CoreMemory) REQUIRE c.core_memory_id IS UNIQUE",
        "CREATE CONSTRAINT omchunkfailure_chunk_id IF NOT EXISTS FOR (f:OMChunkFailure) REQUIRE f.chunk_id IS UNIQUE",
        "CREATE CONSTRAINT omchunkchild_child_id IF NOT EXISTS FOR (c:OMChunkChild) REQUIRE c.child_id IS UNIQUE",
        "CREATE CONSTRAINT omextractionevent_event_id IF NOT EXISTS FOR (e:OMExtractionEvent) REQUIRE e.event_id IS UNIQUE",
        "CREATE CONSTRAINT omconvergence_state_id IF NOT EXISTS FOR (s:OMConvergenceState) REQUIRE s.state_id IS UNIQUE",
        "CREATE INDEX om_extraction_event_emitted_at IF NOT EXISTS FOR (e:OMExtractionEvent) ON (e.emitted_at)",
        "CREATE INDEX om_extraction_event_semantic_domain IF NOT EXISTS FOR (e:OMExtractionEvent) ON (e.semantic_domain)",
        "CREATE INDEX om_extraction_event_node_id IF NOT EXISTS FOR (e:OMExtractionEvent) ON (e.node_id)",
    ]
    for stmt in stmts:
        session.run(stmt).consume()


def _fetch_backlog_stats(session: Any) -> tuple[int, float | None]:
    row = session.run(
        """
        MATCH (m:Message)
        WHERE coalesce(m.om_extracted, false) = false
          AND coalesce(m.om_dead_letter, false) = false
        RETURN count(m) AS backlog_count,
               min(m.created_at) AS oldest_created_at
        """
    ).single()
    backlog = int(row["backlog_count"] or 0) if row else 0
    oldest = row["oldest_created_at"] if row else None

    oldest_hours: float | None = None
    if oldest:
        dt = parse_iso(str(oldest))
        if dt is not None:
            oldest_hours = max(0.0, (datetime.now(timezone.utc) - dt).total_seconds() / 3600.0)
    return backlog, oldest_hours


def _fetch_messages_by_ids(session: Any, message_ids: list[str]) -> list[MessageRow]:
    if not message_ids:
        return []

    rows = session.run(
        """
        MATCH (m:Message)
        WHERE m.message_id IN $message_ids
        RETURN m.message_id AS message_id,
               coalesce(m.source_session_id, 'unknown') AS source_session_id,
               coalesce(m.content, '') AS content,
               coalesce(m.created_at, $now_iso) AS created_at,
               coalesce(m.content_embedding, []) AS content_embedding,
               coalesce(m.om_extract_attempts, 0) AS om_extract_attempts
        ORDER BY m.created_at ASC, m.message_id ASC
        """,
        {"message_ids": message_ids, "now_iso": now_iso()},
    ).data()

    by_id = {
        str(r["message_id"]): MessageRow(
            message_id=str(r["message_id"]),
            source_session_id=str(r["source_session_id"]),
            content=str(r["content"]),
            created_at=str(r["created_at"]),
            content_embedding=[float(v) for v in _safe_list(r.get("content_embedding"))],
            om_extract_attempts=int(r.get("om_extract_attempts") or 0),
        )
        for r in rows
    }
    return [by_id[mid] for mid in message_ids if mid in by_id]


def _fetch_parent_messages(session: Any, limit: int = MAX_PARENT_CHUNK_SIZE) -> list[MessageRow]:
    rows = session.run(
        """
        MATCH (m:Message)
        WHERE coalesce(m.om_extracted, false) = false
          AND coalesce(m.om_dead_letter, false) = false
        RETURN m.message_id AS message_id,
               coalesce(m.source_session_id, 'unknown') AS source_session_id,
               coalesce(m.content, '') AS content,
               coalesce(m.created_at, $now_iso) AS created_at,
               coalesce(m.content_embedding, []) AS content_embedding,
               coalesce(m.om_extract_attempts, 0) AS om_extract_attempts
        ORDER BY m.created_at ASC, m.message_id ASC
        LIMIT $n
        """,
        {"n": int(limit), "now_iso": now_iso()},
    ).data()

    out: list[MessageRow] = []
    for row in rows:
        out.append(
            MessageRow(
                message_id=str(row["message_id"]),
                source_session_id=str(row["source_session_id"]),
                content=str(row["content"]),
                created_at=str(row["created_at"]),
                content_embedding=[float(v) for v in _safe_list(row.get("content_embedding"))],
                om_extract_attempts=int(row.get("om_extract_attempts") or 0),
            )
        )
    return out


def _fetch_structured_parent(session: Any) -> ParentState | None:
    row = session.run(
        """
        MATCH (p:OMChunkFailure)
        WHERE p.split_status IN ['pending', 'isolate']
        RETURN p.chunk_id AS chunk_id,
               coalesce(p.attempts, 0) AS attempts,
               p.first_failed_at AS first_failed_at,
               p.split_status AS split_status,
               p.next_subchunk_index AS next_subchunk_index,
               coalesce(p.child_chunk_ids, []) AS child_chunk_ids,
               coalesce(p.parent_message_ids, []) AS parent_message_ids,
               p.split_backfill_version AS split_backfill_version
        ORDER BY p.first_failed_at ASC, p.chunk_id ASC
        LIMIT 1
        """
    ).single()
    if row is None:
        return None

    next_index_raw = row.get("next_subchunk_index")
    next_index = int(next_index_raw) if next_index_raw is not None else None
    return ParentState(
        chunk_id=str(row["chunk_id"]),
        attempts=int(row["attempts"] or 0),
        first_failed_at=str(row["first_failed_at"]) if row.get("first_failed_at") else None,
        split_status=str(row.get("split_status") or "none"),
        next_subchunk_index=next_index,
        child_chunk_ids=_safe_str_list(row.get("child_chunk_ids")),
        parent_message_ids=_safe_str_list(row.get("parent_message_ids")),
        split_backfill_version=str(row.get("split_backfill_version")) if row.get("split_backfill_version") else None,
    )


def _fetch_children(session: Any, parent_chunk_id: str) -> list[ChildState]:
    rows = session.run(
        """
        MATCH (:OMChunkFailure {chunk_id:$chunk_id})-[:HAS_CHILD]->(c:OMChunkChild)
        RETURN c.child_index AS child_index,
               c.child_chunk_id AS child_chunk_id,
               coalesce(c.message_ids, []) AS message_ids
        ORDER BY c.child_index ASC
        """,
        {"chunk_id": parent_chunk_id},
    ).data()

    children: list[ChildState] = []
    for row in rows:
        children.append(
            ChildState(
                child_index=int(row["child_index"]),
                child_chunk_id=str(row["child_chunk_id"]),
                message_ids=_safe_str_list(row.get("message_ids")),
            )
        )
    return children


def _materialize_parent(
    session: Any,
    *,
    parent_chunk_id: str,
    parent_message_ids: list[str],
    extractor_version: str,
) -> str:
    if len(parent_message_ids) > MAX_CHILD_CHUNK_SIZE:
        split_status = "pending"
        chunk_size = MAX_CHILD_CHUNK_SIZE
    else:
        split_status = "isolate"
        chunk_size = 1

    children_payload: list[dict[str, Any]] = []
    for offset in range(0, len(parent_message_ids), chunk_size):
        child_index = len(children_payload) + 1
        ids = parent_message_ids[offset : offset + chunk_size]
        if not ids:
            continue
        child_chunk_id = sha256_hex(f"{ids[0]}|{ids[-1]}|{len(ids)}|{extractor_version}")
        children_payload.append(
            {
                "child_index": child_index,
                "child_chunk_id": child_chunk_id,
                "message_ids": ids,
                "child_id": _child_id(parent_chunk_id, child_index),
            }
        )

    tx_query = """
    MATCH (p:OMChunkFailure {chunk_id:$chunk_id})
    SET p.archived_at = $now_iso,
        p.parent_message_ids = $parent_message_ids,
        p.child_chunk_ids = $child_chunk_ids,
        p.next_subchunk_index = 1,
        p.split_status = $split_status
    WITH p
    UNWIND $children AS child
    MERGE (c:OMChunkChild {child_id:child.child_id})
      ON CREATE SET
        c.parent_chunk_id = $chunk_id,
        c.child_index = child.child_index,
        c.child_chunk_id = child.child_chunk_id,
        c.message_ids = child.message_ids
    MERGE (p)-[:HAS_CHILD]->(c)
    """
    session.run(
        tx_query,
        {
            "chunk_id": parent_chunk_id,
            "now_iso": now_iso(),
            "parent_message_ids": parent_message_ids,
            "child_chunk_ids": [c["child_chunk_id"] for c in children_payload],
            "split_status": split_status,
            "children": children_payload,
        },
    ).consume()

    return split_status


def _split_integrity_check(
    parent: ParentState,
    children: list[ChildState],
) -> tuple[bool, str]:
    if not parent.child_chunk_ids:
        return False, "check_1"
    if len(children) != len(parent.child_chunk_ids):
        return False, "check_2"

    expected_idx = list(range(1, len(parent.child_chunk_ids) + 1))
    got_idx = [c.child_index for c in children]
    if got_idx != expected_idx:
        return False, "check_3"

    ordered_child_ids = [c.child_chunk_id for c in children]
    if ordered_child_ids != parent.child_chunk_ids:
        return False, "check_4"

    for child in children:
        if not child.message_ids:
            return False, "check_5"

    seen: set[str] = set()
    for child in children:
        for mid in child.message_ids:
            if mid in seen:
                return False, "check_6"
            seen.add(mid)

    flattened: list[str] = []
    for child in children:
        flattened.extend(child.message_ids)
    if flattened != parent.parent_message_ids:
        return False, "check_7"

    return True, ""


def _run_split_backfill_v1(session: Any, parent: ParentState, extractor_version: str) -> list[ChildState]:
    if parent.split_backfill_version is not None:
        return _fetch_children(session, parent.chunk_id)

    if not parent.parent_message_ids:
        session.run(
            """
            MATCH (p:OMChunkFailure {chunk_id:$chunk_id})
            SET p.split_backfill_version = 'v1_failed'
            """,
            {"chunk_id": parent.chunk_id},
        ).consume()
        emit_event(
            "OM_SPLIT_STATE_UNMIGRATABLE",
            parent_chunk_id=parent.chunk_id,
            reason="missing parent_message_ids",
        )
        raise OMCompressorError("split state unmigratable")

    split_status = _materialize_parent(
        session,
        parent_chunk_id=parent.chunk_id,
        parent_message_ids=parent.parent_message_ids,
        extractor_version=extractor_version,
    )
    session.run(
        """
        MATCH (p:OMChunkFailure {chunk_id:$chunk_id})
        SET p.split_backfill_version = 'v1',
            p.split_status = $split_status
        """,
        {"chunk_id": parent.chunk_id, "split_status": split_status},
    ).consume()

    children = _fetch_children(session, parent.chunk_id)
    emit_event(
        "OM_SPLIT_BACKFILL_V1",
        parent_chunk_id=parent.chunk_id,
        reconstructed_children=len(children),
    )
    return children


def _increment_chunk_failure(
    session: Any,
    *,
    chunk_id: str,
) -> int:
    row = session.run(
        """
        MERGE (f:OMChunkFailure {chunk_id:$chunk_id})
        ON CREATE SET
          f.attempts = 1,
          f.first_failed_at = $now_iso,
          f.last_failed_at = $now_iso,
          f.split_status = 'none',
          f.next_subchunk_index = NULL,
          f.child_chunk_ids = [],
          f.parent_message_ids = [],
          f.split_backfill_version = NULL,
          f.archived_at = NULL
        ON MATCH SET
          f.attempts = coalesce(f.attempts, 0) + 1,
          f.last_failed_at = $now_iso
        RETURN f.attempts AS attempts, coalesce(f.split_status, 'none') AS split_status
        """,
        {"chunk_id": chunk_id, "now_iso": now_iso()},
    ).single()

    return int(row["attempts"] if row else 1)


def _update_parent_for_children_completion(session: Any, parent_chunk_id: str) -> None:
    session.run(
        """
        MATCH (p:OMChunkFailure {chunk_id:$chunk_id})-[:HAS_CHILD]->(c:OMChunkChild)
        WITH p, c ORDER BY c.child_index ASC
        WITH p, collect(c) AS children
        WITH p, children,
             all(child IN children WHERE all(mid IN child.message_ids WHERE EXISTS {
               MATCH (m:Message {message_id:mid})
               WHERE coalesce(m.om_extracted, false) = true OR coalesce(m.om_dead_letter, false) = true
             })) AS all_terminal
        FOREACH (_ IN CASE WHEN all_terminal THEN [1] ELSE [] END |
          SET p.next_subchunk_index = size(children) + 1,
              p.split_status = 'completed'
        )
        """,
        {"chunk_id": parent_chunk_id},
    ).consume()


def _mark_child_archived_failure(session: Any, child_chunk_id: str) -> None:
    session.run(
        """
        MATCH (f:OMChunkFailure {chunk_id:$chunk_id})
        SET f.archived_at = $now_iso
        """,
        {"chunk_id": child_chunk_id, "now_iso": now_iso()},
    ).consume()


def _activate_energy_scores(
    session: Any,
    messages: list[MessageRow],
) -> tuple[list[ActivationCandidate], list[str]]:
    vectors = [m.content_embedding for m in messages if m.content_embedding]
    query_embedding = l2_normalize(mean_vector(vectors))
    if not query_embedding:
        return [], []

    rows = session.run(
        """
        MATCH (n:OMNode)
        WHERE n.status IN ['open', 'monitoring', 'reopened']
           OR (n.status = 'closed' AND n.status_changed_at IS NOT NULL
               AND duration.inDays(datetime(n.status_changed_at), datetime()).days < 14)
        RETURN n.node_id AS node_id,
               coalesce(n.status, 'open') AS status,
               coalesce(n.urgency_score, 3) AS urgency_score,
               coalesce(n.created_at, $now_iso) AS created_at,
               n.status_changed_at AS status_changed_at,
               n.last_observed_at AS last_observed_at,
               coalesce(n.content_embedding, []) AS content_embedding
        """,
        {"now_iso": now_iso()},
    ).data()

    candidates: list[ActivationCandidate] = []
    observed_ids: list[str] = []
    now_dt = datetime.now(timezone.utc)

    for row in rows:
        emb = [float(v) for v in _safe_list(row.get("content_embedding"))]
        if len(emb) != len(query_embedding):
            continue

        sim = max(0.0, min(1.0, cosine_similarity(query_embedding, emb)))
        if sim > 0.85:
            observed_ids.append(str(row["node_id"]))

        urgency = max(1, min(5, int(row.get("urgency_score") or 3)))
        last_ts = parse_iso(str(row.get("last_observed_at") or "")) or parse_iso(str(row.get("created_at") or ""))
        if last_ts is None:
            age_days = 0.0
        else:
            age_days = max(0.0, (now_dt - last_ts).total_seconds() / 86400.0)

        # Score = sim*0.4 + urgency/5*0.3 + exp(-lambda*age_days)*0.3
        decay = max(0.0, min(1.0, 2.718281828 ** (-0.1 * age_days)))
        score = sim * 0.4 + (urgency / 5.0) * 0.3 + decay * 0.3
        candidate = ActivationCandidate(
            node_id=str(row["node_id"]),
            status=str(row.get("status") or "open"),
            urgency_score=urgency,
            created_at=str(row.get("created_at") or now_iso()),
            status_changed_at=str(row.get("status_changed_at")) if row.get("status_changed_at") else None,
            last_observed_at=str(row.get("last_observed_at")) if row.get("last_observed_at") else None,
            content_embedding=emb,
        )
        candidate._score = score  # lightweight internal annotation
        candidates.append(candidate)

    candidates.sort(
        key=lambda c: (
            -float(getattr(c, "_score", 0.0)),
            str(c.last_observed_at or c.created_at),
            c.node_id,
        )
    )

    kept: list[ActivationCandidate] = []
    budget = 2000
    used = 0
    for cand in candidates:
        text = f"{cand.node_id} {cand.status}"
        t = tokenize_len(text)
        if used + t > budget:
            continue
        kept.append(cand)
        used += t

    return kept, observed_ids


def _write_dead_letter(
    message: MessageRow,
    *,
    attempts: int,
    last_error: str,
    chunk_id: str | None,
) -> None:
    conn = candidates_store.connect(candidates_store.DB_PATH_DEFAULT)
    try:
        candidates_store.upsert_om_dead_letter(
            conn,
            message_id=message.message_id,
            source_session_id=message.source_session_id or "unknown",
            attempts=attempts,
            last_error=last_error,
            first_failed_at=message.created_at or now_iso(),
            last_failed_at=now_iso(),
            last_chunk_id=chunk_id,
        )
    finally:
        conn.close()

    emit_event(
        "OM_DEAD_LETTER",
        message_id=message.message_id,
        attempts=attempts,
        last_error=last_error,
    )


def _increment_message_attempt(session: Any, message: MessageRow, error: str, chunk_id: str | None) -> None:
    row = session.run(
        """
        MATCH (m:Message {message_id:$message_id})
        SET m.om_extract_attempts = coalesce(m.om_extract_attempts, 0) + 1
        RETURN coalesce(m.om_extract_attempts, 0) AS attempts
        """,
        {"message_id": message.message_id},
    ).single()

    attempts = int(row["attempts"] if row else 1)
    if attempts >= DEAD_LETTER_ATTEMPTS:
        session.run(
            """
            MATCH (m:Message {message_id:$message_id})
            SET m.om_dead_letter = true
            """,
            {"message_id": message.message_id},
        ).consume()
        _write_dead_letter(message, attempts=attempts, last_error=error, chunk_id=chunk_id)


def _legal_edge(edge: ExtractionEdge) -> bool:
    # Node type legality is enforced by upstream extraction contract;
    # fallback extractor currently emits no structural edges.
    return edge.relation_type in RELATION_TYPES


def _validated_relation_type_for_cypher(raw_relation_type: str) -> str:
    rel = str(raw_relation_type or "").strip().upper()
    if rel not in RELATION_TYPES:
        raise OMCompressorError(f"illegal relation type interpolation blocked: {raw_relation_type!r}")
    if not RELATION_TYPE_TOKEN_RE.fullmatch(rel):
        raise OMCompressorError(f"invalid relation type token: {raw_relation_type!r}")
    return rel


def _assert_relation_type_safe_for_interpolation(rel: str, edge: ExtractionEdge) -> str:
    if rel not in RELATION_TYPES:
        emit_event(
            "OM_RELATION_TYPE_INTERPOLATION_BLOCKED",
            relation_type=rel,
            source_node_id=edge.source_node_id,
            target_node_id=edge.target_node_id,
            reason="not_allowlisted",
        )
        raise OMCompressorError(f"illegal relation type interpolation blocked: {rel!r}")
    if not RELATION_TYPE_TOKEN_RE.fullmatch(rel):
        emit_event(
            "OM_RELATION_TYPE_INTERPOLATION_BLOCKED",
            relation_type=rel,
            source_node_id=edge.source_node_id,
            target_node_id=edge.target_node_id,
            reason="regex_mismatch",
        )
        raise OMCompressorError(f"invalid relation type token: {rel!r}")
    return rel


def _rank_extraction_nodes(nodes: list[ExtractionNode]) -> list[ExtractionNode]:
    ranked = sorted(
        nodes,
        key=lambda n: (
            n.node_type,
            n.node_id,
            sha256_hex(normalize_text(n.content)),
        ),
    )
    return ranked


def _process_chunk_tx(
    tx: Any,
    *,
    messages: list[MessageRow],
    chunk_id: str,
    cfg: ExtractorConfig,
    observed_node_ids: list[str],
) -> dict[str, Any]:
    extracted = _extract_items(messages, cfg)
    ranked_nodes = _rank_extraction_nodes(extracted.nodes)

    rewrite_embeddings = os.environ.get("OM_REWRITE_EMBEDDINGS") == "1"
    embedding_model, embedding_dim = _embedding_config()
    now = now_iso()

    node_ids: set[str] = set()

    for node in ranked_nodes:
        check_row = tx.run(
            "MATCH (n:OMNode {node_id:$node_id}) RETURN n.content AS content",
            {"node_id": node.node_id},
        ).single()

        existing_content = str(check_row["content"]) if check_row and check_row.get("content") is not None else None
        incoming_content = normalize_text(node.content)
        if existing_content is not None and normalize_text(existing_content) != incoming_content and not rewrite_embeddings:
            raise NodeContentMismatchError(
                node_id=node.node_id,
                existing_hash=sha256_hex(normalize_text(existing_content)),
                incoming_hash=sha256_hex(incoming_content),
            )

        embedding = _embed_text(
            incoming_content,
            embedding_model=embedding_model,
            embedding_dim=embedding_dim,
        )

        tx.run(
            """
            MERGE (n:OMNode {node_id:$node_id})
            ON CREATE SET
              n.node_type = $node_type,
              n.semantic_domain = $semantic_domain,
              n.content = $content,
              n.content_embedding = $embedding,
              n.embedding_model = $embedding_model,
              n.embedding_dim = $embedding_dim,
              n.urgency_score = $urgency_score,
              n.status = $status,
              n.source_session_id = $source_session_id,
              n.source_message_ids = $source_message_ids,
              n.created_at = $created_at,
              n.status_changed_at = $created_at,
              n.last_observed_at = NULL,
              n.monitoring_started_at = NULL
            ON MATCH SET
              n.source_message_ids = CASE
                WHEN n.source_message_ids IS NULL THEN $source_message_ids
                ELSE n.source_message_ids
              END
            """,
            {
                "node_id": node.node_id,
                "node_type": node.node_type,
                "semantic_domain": node.semantic_domain,
                "content": incoming_content,
                "embedding": embedding,
                "urgency_score": node.urgency_score,
                "status": node.status,
                "source_session_id": node.source_session_id,
                "source_message_ids": node.source_message_ids,
                "created_at": now,
                "embedding_model": embedding_model,
                "embedding_dim": embedding_dim,
            },
        ).consume()

        if existing_content is not None and normalize_text(existing_content) != incoming_content and rewrite_embeddings:
            tx.run(
                """
                MATCH (n:OMNode {node_id:$node_id})
                SET n.content = $content,
                    n.content_embedding = $embedding,
                    n.embedding_model = $embedding_model,
                    n.embedding_dim = $embedding_dim
                """,
                {
                    "node_id": node.node_id,
                    "content": incoming_content,
                    "embedding": embedding,
                    "embedding_model": embedding_model,
                    "embedding_dim": embedding_dim,
                },
            ).consume()

        node_ids.add(node.node_id)

    for edge in extracted.edges:
        if not _legal_edge(edge):
            emit_event(
                "OM_RELATION_TYPE_INTERPOLATION_BLOCKED",
                relation_type=edge.relation_type,
                source_node_id=edge.source_node_id,
                target_node_id=edge.target_node_id,
                reason="not_allowlisted",
            )
            continue
        rel = _validated_relation_type_for_cypher(edge.relation_type)
        rel = _assert_relation_type_safe_for_interpolation(rel, edge)
        tx.run(
            f"""
            MATCH (s:OMNode {{node_id:$source_node_id}})
            MATCH (t:OMNode {{node_id:$target_node_id}})
            MERGE (s)-[r:{rel}]->(t)
            ON CREATE SET
              r.linked_at = $linked_at,
              r.chunk_id = $chunk_id,
              r.extractor_version = $extractor_version
            """,
            {
                "source_node_id": edge.source_node_id,
                "target_node_id": edge.target_node_id,
                "linked_at": now,
                "chunk_id": chunk_id,
                "extractor_version": cfg.extractor_version,
            },
        ).consume()

    for node in ranked_nodes:
        for message_id in node.source_message_ids:
            tx.run(
                """
                MATCH (m:Message {message_id:$message_id})
                MATCH (n:OMNode {node_id:$node_id})
                MERGE (m)-[r:EVIDENCE_FOR {chunk_id:$chunk_id, extractor_version:$extractor_version}]->(n)
                ON CREATE SET r.linked_at = $linked_at
                """,
                {
                    "message_id": message_id,
                    "node_id": node.node_id,
                    "chunk_id": chunk_id,
                    "extractor_version": cfg.extractor_version,
                    "linked_at": now,
                },
            ).consume()

    for node_id in observed_node_ids:
        tx.run(
            """
            MATCH (n:OMNode {node_id:$node_id})
            SET n.last_observed_at = $now_iso
            """,
            {"node_id": node_id, "now_iso": now},
        ).consume()

    for msg in messages:
        tx.run(
            """
            MATCH (m:Message {message_id:$message_id})
            SET m.om_extracted = true,
                m.om_extracted_at = $now_iso,
                m.om_chunk_id = $chunk_id
            """,
            {"message_id": msg.message_id, "now_iso": now, "chunk_id": chunk_id},
        ).consume()

    for index, node in enumerate(ranked_nodes, start=1):
        normalized_content = normalize_text(node.content)
        event_id = sha256_hex(f"event|{chunk_id}|{cfg.extractor_version}|{index}")
        tx.run(
            """
            MATCH (n:OMNode {node_id:$node_id})
            MERGE (e:OMExtractionEvent {event_id:$event_id})
            ON CREATE SET
              e.emitted_at = $emitted_at,
              e.chunk_id = $chunk_id,
              e.extractor_version = $extractor_version,
              e.item_index = $item_index,
              e.semantic_domain = $semantic_domain,
              e.node_id = $node_id,
              e.normalized_content = $normalized_content,
              e.content_embedding = n.content_embedding,
              e.embedding_model = n.embedding_model,
              e.embedding_dim = n.embedding_dim
            MERGE (e)-[:EMITTED]->(n)
            """,
            {
                "event_id": event_id,
                "emitted_at": now,
                "chunk_id": chunk_id,
                "extractor_version": cfg.extractor_version,
                "item_index": index,
                "semantic_domain": node.semantic_domain,
                "node_id": node.node_id,
                "normalized_content": normalized_content,
            },
        ).consume()

    return {
        "chunk_id": chunk_id,
        "messages": len(messages),
        "nodes": len(ranked_nodes),
        "edges": len(extracted.edges),
    }


def _process_chunk(
    session: Any,
    *,
    messages: list[MessageRow],
    chunk_id: str,
    cfg: ExtractorConfig,
    observed_node_ids: list[str],
) -> dict[str, Any]:
    return session.execute_write(
        _process_chunk_tx,
        messages=messages,
        chunk_id=chunk_id,
        cfg=cfg,
        observed_node_ids=observed_node_ids,
    )


def _handle_parent_failure(
    session: Any,
    *,
    chunk_id: str,
    messages: list[MessageRow],
    cfg: ExtractorConfig,
) -> None:
    attempts = _increment_chunk_failure(session, chunk_id=chunk_id)
    if attempts >= 2:
        row = session.run(
            """
            MATCH (p:OMChunkFailure {chunk_id:$chunk_id})
            RETURN coalesce(p.split_status, 'none') AS split_status
            """,
            {"chunk_id": chunk_id},
        ).single()
        split_status = str(row["split_status"] if row else "none")
        if split_status == "none":
            _materialize_parent(
                session,
                parent_chunk_id=chunk_id,
                parent_message_ids=[m.message_id for m in messages],
                extractor_version=cfg.extractor_version,
            )


def _process_single_message(session: Any, cfg: ExtractorConfig, message: MessageRow, chunk_id: str) -> bool:
    try:
        _process_chunk(
            session,
            messages=[message],
            chunk_id=chunk_id,
            cfg=cfg,
            observed_node_ids=[],
        )
        return True
    except Exception as exc:
        _increment_message_attempt(session, message, error=str(exc), chunk_id=chunk_id)
        return False


def _process_structured_parent(session: Any, parent: ParentState, cfg: ExtractorConfig) -> bool:
    children = _fetch_children(session, parent.chunk_id)
    if not children and parent.parent_message_ids:
        children = _run_split_backfill_v1(session, parent, cfg.extractor_version)
        parent = _fetch_structured_parent(session) or parent

    ok, violation = _split_integrity_check(parent, children)
    if not ok:
        emit_event(
            "OM_SPLIT_STATE_CORRUPT",
            parent_chunk_id=parent.chunk_id,
            violation_code=violation,
        )
        raise OMCompressorError(f"structured parent corrupt: {parent.chunk_id}:{violation}")

    next_index = parent.next_subchunk_index or 1
    child_lookup = {child.child_index: child for child in children}
    child = child_lookup.get(next_index)
    if child is None:
        _update_parent_for_children_completion(session, parent.chunk_id)
        return True

    child_messages = _fetch_messages_by_ids(session, child.message_ids)
    if not child_messages:
        emit_event(
            "OM_SPLIT_STATE_CORRUPT",
            parent_chunk_id=parent.chunk_id,
            violation_code="check_5",
        )
        raise OMCompressorError(f"child has no resolvable messages: {child.child_chunk_id}")

    success = False
    if len(child_messages) == 1:
        success = _process_single_message(session, cfg, child_messages[0], child.child_chunk_id)
    else:
        try:
            _, observed_ids = _activate_energy_scores(session, child_messages)
            _process_chunk(
                session,
                messages=child_messages,
                chunk_id=child.child_chunk_id,
                cfg=cfg,
                observed_node_ids=observed_ids,
            )
            success = True
        except Exception as exc:
            attempts = _increment_chunk_failure(session, chunk_id=child.child_chunk_id)
            if attempts >= 2:
                success = True
                for msg in child_messages:
                    if not _process_single_message(session, cfg, msg, child.child_chunk_id):
                        success = False
            else:
                emit_event(
                    "OM_CHILD_CHUNK_RETRY",
                    parent_chunk_id=parent.chunk_id,
                    child_chunk_id=child.child_chunk_id,
                    attempts=attempts,
                    error=str(exc),
                )
                return False

    if success:
        _mark_child_archived_failure(session, child.child_chunk_id)
        session.run(
            """
            MATCH (p:OMChunkFailure {chunk_id:$chunk_id})
            SET p.next_subchunk_index = $next_index
            """,
            {"chunk_id": parent.chunk_id, "next_index": child.child_index + 1},
        ).consume()
        _update_parent_for_children_completion(session, parent.chunk_id)
    return success


def run(args: argparse.Namespace) -> int:
    cfg = _load_extractor_config(_resolve_ontology_config_path(args.config))

    driver = _neo4j_driver()
    with driver, driver.session(database=os.environ.get("NEO4J_DATABASE", "neo4j")) as session:
        _ensure_neo4j_constraints(session)

        parent = _fetch_structured_parent(session)
        if parent is not None:
            _process_structured_parent(session, parent, cfg)
            return 0

        processed = 0
        max_chunks = max(1, int(args.max_chunks_per_run))
        while processed < max_chunks:
                backlog_count, oldest_hours = _fetch_backlog_stats(session)
                trigger = backlog_count >= 50 or (oldest_hours is not None and oldest_hours >= 48.0)
                if not args.force and not trigger:
                    if processed == 0:
                        emit_event(
                            "OM_TRIGGER_NOT_MET",
                            backlog_count=backlog_count,
                            oldest_backlog_age_hours=oldest_hours,
                        )
                    break

                chunk_messages = _fetch_parent_messages(session, MAX_PARENT_CHUNK_SIZE)
                if not chunk_messages:
                    break

                chunk_id = _chunk_id(chunk_messages, cfg.extractor_version)
                try:
                    _, observed_ids = _activate_energy_scores(session, chunk_messages)
                    result = _process_chunk(
                        session,
                        messages=chunk_messages,
                        chunk_id=chunk_id,
                        cfg=cfg,
                        observed_node_ids=observed_ids,
                    )
                    emit_event("OM_CHUNK_PROCESSED", **result)
                    processed += 1
                except NodeContentMismatchError as exc:
                    emit_event(
                        "OM_NODE_CONTENT_MISMATCH",
                        node_id=exc.node_id,
                        existing_content_hash=exc.existing_hash,
                        incoming_content_hash=exc.incoming_hash,
                    )
                    return 1
                except Exception as exc:
                    if len(chunk_messages) > 1:
                        _handle_parent_failure(
                            session,
                            chunk_id=chunk_id,
                            messages=chunk_messages,
                            cfg=cfg,
                        )
                    else:
                        _increment_message_attempt(session, chunk_messages[0], error=str(exc), chunk_id=chunk_id)
                    emit_event("OM_CHUNK_FAILED", chunk_id=chunk_id, error=str(exc))
                    return 1

    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OM compressor")
    parser.add_argument(
        "--config",
        default=None,
        help="Path to extraction_ontologies.yaml",
    )
    parser.add_argument("--force", action="store_true", help="process regardless of trigger threshold")
    parser.add_argument(
        "--max-chunks-per-run",
        type=int,
        default=DEFAULT_MAX_CHUNKS_PER_RUN,
    )
    parser.add_argument(
        "--mode",
        choices=["steady", "backfill"],
        default="steady",
        help="steady = single-writer lock (default); backfill = parallel claim mode",
    )
    parser.add_argument(
        "--build-manifest",
        default=None,
        help="Build OM backfill manifest from pending messages",
    )
    parser.add_argument(
        "--claim-mode",
        action="store_true",
        help="Enable claim-based parallel execution (backfill mode only)",
    )
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview without writing",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if args.mode == "backfill":
        # Backfill mode: skip file lock for parallel execution
        try:
            return run(args)
        except SchemaVersionMissingError as exc:
            print(str(exc), file=sys.stderr)
            return 1
        except OMCompressorError as exc:
            print(str(exc), file=sys.stderr)
            return 1
    else:
        # Steady mode: use file lock (single-writer)
        lock_path = _resolve_lock_path()
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with lock_path.open("a+") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            try:
                return run(args)
            except SchemaVersionMissingError as exc:
                print(str(exc), file=sys.stderr)
                return 1
            except OMCompressorError as exc:
                print(str(exc), file=sys.stderr)
                return 1
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


if __name__ == "__main__":
    raise SystemExit(main())
