#!/usr/bin/env python3
"""Fast-write append path for Observational Memory transcript primitives.

This script writes Episode/Message nodes immediately and enforces:
- embedding_model/dim defaults (embeddinggemma/768), configurable via
  OM_EMBEDDING_MODEL and OM_EMBEDDING_DIM
- extraction lifecycle defaults on new Message rows
- FAST_WRITE_DISABLED / FAST_WRITE_ENABLED state file semantics
"""

from __future__ import annotations

import argparse
import hashlib
import ipaddress
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_EMBEDDING_MODEL = "embeddinggemma"
DEFAULT_EMBEDDING_DIM = 768
DEFAULT_OM_GROUP_ID = "s1_observational_memory"
DEFAULT_STATE_FILE = "state/om_fast_write_state.json"
NEO4J_ENV_FALLBACK_FILE = Path.home() / ".clawdbot" / "credentials" / "neo4j.env"
NEO4J_NON_DEV_FALLBACK_OPT_IN_ENV = "OM_NEO4J_ENV_FALLBACK_NON_DEV"
_NON_DEV_ENV_MARKERS = {"prod", "production", "staging", "stage", "preprod", "preview"}
_TRUTHY_ENV_VALUES = {"1", "true", "yes", "on"}


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
        raise RuntimeError(f"OM_EMBEDDING_DIM must be an integer, got: {raw_dim!r}") from exc
    if dim <= 0:
        raise RuntimeError(f"OM_EMBEDDING_DIM must be > 0, got: {dim}")

    return model, dim


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _sha256_hex(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _emit(event: str, **payload: Any) -> None:
    body = {"event": event, "timestamp": _now_iso(), **payload}
    print(json.dumps(body, ensure_ascii=True))


def _normalize_iso(ts: str | None) -> str:
    if not ts:
        return _now_iso()
    text = str(ts).strip()
    if not text:
        return _now_iso()
    if text.endswith("Z"):
        return text
    try:
        dt = datetime.fromisoformat(text)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    except Exception:
        return _now_iso()


def _om_group_id(raw: str | None = None) -> str:
    value = (raw if raw is not None else os.environ.get("OM_GROUP_ID") or DEFAULT_OM_GROUP_ID).strip()
    if not value:
        value = DEFAULT_OM_GROUP_ID
    if not all(ch.isalnum() or ch in {"_", "-", "."} for ch in value):
        raise RuntimeError(f"invalid OM_GROUP_ID: {value!r}")
    return value


def _append_gitignore_entry(state_dir: Path) -> None:
    gi = state_dir / ".gitignore"
    existing = gi.read_text(encoding="utf-8") if gi.exists() else ""
    lines = [line.strip() for line in existing.splitlines()]
    if "om_fast_write_state.json" in lines:
        return
    new_text = existing
    if new_text and not new_text.endswith("\n"):
        new_text += "\n"
    new_text += "om_fast_write_state.json\n"
    gi.write_text(new_text, encoding="utf-8")


def write_fast_write_state(runtime_repo: Path, status: str, reason: str) -> Path:
    state_path = runtime_repo / DEFAULT_STATE_FILE
    state_path.parent.mkdir(parents=True, exist_ok=True)
    _append_gitignore_entry(state_path.parent)

    payload = {
        "status": status,
        "reason": reason,
        "updated_at": _now_iso(),
    }
    state_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")

    event = "FAST_WRITE_ENABLED" if status == "enabled" else "FAST_WRITE_DISABLED"
    _emit(event, reason=reason, state_file=str(state_path))
    return state_path


def _post_json(url: str, payload: dict[str, Any], timeout: int) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    api_key = os.environ.get("EMBEDDER_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    req = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            return json.loads(raw) if raw.strip() else {}
    except urllib.error.HTTPError as exc:
        details = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"embedding request failed status={exc.code} body={details}") from exc


def _validated_embedding_base_url() -> str:
    """Resolve embedding base URL (delegates to shared env_utils)."""
    try:
        from graphiti_core.utils.env_utils import (
            EndpointResolutionError,
            resolve_embedder_base_url,
        )
        try:
            return resolve_embedder_base_url()
        except EndpointResolutionError as exc:
            raise RuntimeError(str(exc)) from exc
    except ImportError:
        # Fallback: standalone resolution for minimal environments.
        # Enforces parity with resolve_embedder_base_url() security rules:
        # - Must be absolute http(s) with a non-empty host.
        # - No embedded credentials, query strings, or fragments.
        # - Link-local (169.254.x.x / fe80::) always blocked (cloud-metadata SSRF vector).
        # - Private/loopback addresses ALLOWED (local Ollama is a primary use-case).
        base = (os.environ.get("EMBEDDER_BASE_URL") or os.environ.get("OPENAI_BASE_URL") or "").strip()
        if not base:
            base = "http://localhost:11434/v1"
        parsed = urllib.parse.urlparse(base)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise RuntimeError(  # noqa: B904 — new validation error, not re-raise
                f"embedding base URL must be an absolute http(s) URL, got: {base!r}"
            )
        if parsed.username or parsed.password:
            raise RuntimeError(  # noqa: B904
                f"embedding base URL must not include embedded credentials: {base!r}"
            )
        if parsed.query:
            raise RuntimeError(  # noqa: B904
                f"embedding base URL must not include a query string: {base!r}"
            )
        if parsed.fragment:
            raise RuntimeError(  # noqa: B904
                f"embedding base URL must not include a fragment: {base!r}"
            )
        host = (urllib.parse.urlparse(f"//{parsed.netloc}").hostname or "").strip()
        try:
            addr = ipaddress.ip_address(host)
            if addr.is_link_local:
                raise RuntimeError(  # noqa: B904
                    f"embedding base URL {base!r} targets a link-local/cloud-metadata address; "
                    "this is always blocked."
                )
        except ValueError:
            pass  # hostname (not numeric IP) — no link-local check needed
        return base.rstrip("/")


def embed_text(content: str, *, embedding_model: str, embedding_dim: int) -> list[float]:
    """Embed text through an OpenAI-compatible /embeddings endpoint.

    Fast-write is fail-closed if embedding fails.
    """

    base = _validated_embedding_base_url()
    url = base + "/embeddings"

    payload = {
        "model": embedding_model,
        "input": content,
    }
    timeout = int(os.environ.get("OM_EMBED_TIMEOUT_SECONDS", "20"))
    response = _post_json(url, payload, timeout)
    data = response.get("data")
    if not isinstance(data, list) or not data:
        raise RuntimeError("embedding response missing data[0]")

    embedding = data[0].get("embedding") if isinstance(data[0], dict) else None
    if not isinstance(embedding, list):
        raise RuntimeError("embedding response missing vector")

    try:
        vector = [float(x) for x in embedding]
    except Exception as exc:
        raise RuntimeError("embedding vector contains non-numeric values") from exc

    if len(vector) != embedding_dim:
        raise RuntimeError(
            f"embedding dimension mismatch: got {len(vector)} expected {embedding_dim}"
        )
    return vector


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


def _neo4j_driver_from_env() -> Any:
    try:
        from neo4j import GraphDatabase  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("neo4j driver is required (pip install neo4j)") from exc

    _load_neo4j_env_fallback()

    uri = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
    user = os.environ.get("NEO4J_USER", "neo4j")
    password = os.environ.get("NEO4J_PASSWORD")
    if not password:
        raise RuntimeError("NEO4J_PASSWORD is required")

    return GraphDatabase.driver(uri, auth=(user, password))


def _ensure_constraints(session: Any) -> None:
    statements = [
        "CREATE CONSTRAINT message_message_id IF NOT EXISTS FOR (m:Message) REQUIRE m.message_id IS UNIQUE",
    ]
    for stmt in statements:
        session.run(stmt).consume()


@dataclass(frozen=True)
class FastWritePayload:
    source_session_id: str
    role: str
    content: str
    created_at: str
    message_id: str
    episode_id: str
    group_id: str



def _coerce_payload(raw: dict[str, Any]) -> FastWritePayload:
    session_id = str(raw.get("source_session_id") or raw.get("session_id") or "").strip()
    role = str(raw.get("role") or "user").strip() or "user"
    content = str(raw.get("content") or "").strip()
    created_at = _normalize_iso(str(raw.get("created_at") or "").strip())

    if not session_id:
        raise ValueError("source_session_id/session_id is required")
    if not content:
        raise ValueError("content is required")

    message_id = str(raw.get("message_id") or "").strip()
    if not message_id:
        message_id = _sha256_hex(f"msg|{session_id}|{created_at}|{role}|{content}")

    episode_id = str(raw.get("episode_id") or "").strip()
    if not episode_id:
        episode_id = _sha256_hex(f"episode|{session_id}")

    group_id = _om_group_id(str(raw.get("group_id") or "").strip() or None)

    return FastWritePayload(
        source_session_id=session_id,
        role=role,
        content=content,
        created_at=created_at,
        message_id=message_id,
        episode_id=episode_id,
        group_id=group_id,
    )


def fast_write(payload: FastWritePayload) -> dict[str, Any]:
    embedding_model, embedding_dim = _embedding_config()
    vector = embed_text(payload.content, embedding_model=embedding_model, embedding_dim=embedding_dim)

    query = """
    MERGE (e:Episode {episode_id:$episode_id})
    ON CREATE SET
      e.group_id = $group_id,
      e.source_session_id = $source_session_id,
      e.started_at = $created_at,
      e.last_message_at = $created_at
    ON MATCH SET
      e.group_id = coalesce(e.group_id, $group_id),
      e.last_message_at = CASE
        WHEN e.last_message_at IS NULL OR e.last_message_at < $created_at THEN $created_at
        ELSE e.last_message_at
      END
    MERGE (m:Message {message_id:$message_id})
    ON CREATE SET
      m.group_id = $group_id,
      m.episode_id = $episode_id,
      m.source_session_id = $source_session_id,
      m.role = $role,
      m.content = $content,
      m.created_at = $created_at,
      m.content_embedding = $content_embedding,
      m.embedding_model = $embedding_model,
      m.embedding_dim = $embedding_dim,
      m.graphiti_extracted_at = NULL,
      m.om_extracted = false,
      m.om_extract_attempts = 0,
      m.om_dead_letter = false
    ON MATCH SET
      m.group_id = coalesce(m.group_id, $group_id)
    MERGE (e)-[hm:HAS_MESSAGE]->(m)
    ON CREATE SET hm.group_id = $group_id
    ON MATCH SET hm.group_id = coalesce(hm.group_id, $group_id)
    RETURN m.message_id AS message_id, e.episode_id AS episode_id
    """

    params = {
        "episode_id": payload.episode_id,
        "group_id": payload.group_id,
        "source_session_id": payload.source_session_id,
        "created_at": payload.created_at,
        "message_id": payload.message_id,
        "role": payload.role,
        "content": payload.content,
        "content_embedding": vector,
        "embedding_model": embedding_model,
        "embedding_dim": embedding_dim,
    }

    driver = _neo4j_driver_from_env()
    with driver, driver.session(database=os.environ.get("NEO4J_DATABASE", "neo4j")) as session:
        _ensure_constraints(session)
        row = session.run(query, params).single()
        if row is None:
            raise RuntimeError("fast-write did not return message row")

    return {
        "message_id": payload.message_id,
        "episode_id": payload.episode_id,
        "group_id": payload.group_id,
        "embedding_model": embedding_model,
        "embedding_dim": embedding_dim,
    }


def _load_payload(args: argparse.Namespace) -> dict[str, Any]:
    if args.payload_file:
        return json.loads(Path(args.payload_file).read_text(encoding="utf-8"))
    if args.payload_json:
        return json.loads(args.payload_json)

    return {
        "source_session_id": args.session_id,
        "role": args.role,
        "content": args.content,
        "created_at": args.created_at,
        "message_id": args.message_id,
        "episode_id": args.episode_id,
        "group_id": args.group_id,
    }


def _cmd_set_state(args: argparse.Namespace) -> int:
    runtime_repo = Path(args.runtime_repo).resolve()
    if not runtime_repo.exists():
        print(f"runtime repo not found: {runtime_repo}", file=sys.stderr)
        return 2

    status = "enabled" if args.enabled else "disabled"
    write_fast_write_state(runtime_repo, status=status, reason=args.reason)
    return 0


def _cmd_write(args: argparse.Namespace) -> int:
    if args.require_runtime_repo and not os.environ.get("RUNTIME_REPO_ROOT"):
        _emit("RUNTIME_REPO_ROOT_MISSING", reason="set RUNTIME_REPO_ROOT before fast-write")
        return 2

    payload = _coerce_payload(_load_payload(args))
    try:
        result = fast_write(payload)
        print(json.dumps(result, ensure_ascii=True))
    except Exception as exc:
        _emit("OM_FAST_WRITE_FAILED", error=str(exc), message_id=payload.message_id)
        return 1

    runtime_repo_raw = args.runtime_repo or os.environ.get("RUNTIME_REPO_ROOT")
    if runtime_repo_raw:
        runtime_repo = Path(runtime_repo_raw).resolve()
        if runtime_repo.exists():
            write_fast_write_state(runtime_repo, status="enabled", reason="hook_wired")

    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="OM fast-write")
    sub = parser.add_subparsers(dest="cmd", required=True)

    set_state = sub.add_parser("set-state", help="write fast-write runtime state")
    set_state.add_argument("--runtime-repo", required=True)
    group = set_state.add_mutually_exclusive_group(required=True)
    group.add_argument("--enabled", action="store_true")
    group.add_argument("--disabled", action="store_true")
    set_state.add_argument("--reason", default="manual")
    set_state.set_defaults(func=_cmd_set_state)

    write = sub.add_parser("write", help="append one transcript message")
    write.add_argument("--session-id")
    write.add_argument("--role", default="user")
    write.add_argument("--content")
    write.add_argument("--created-at")
    write.add_argument("--message-id")
    write.add_argument("--episode-id")
    write.add_argument("--group-id", default=DEFAULT_OM_GROUP_ID)
    write.add_argument("--payload-json", help="full JSON payload")
    write.add_argument("--payload-file", help="path to JSON payload")
    write.add_argument("--runtime-repo", help="runtime repo root for state file update")
    write.add_argument("--require-runtime-repo", action="store_true")
    write.set_defaults(func=_cmd_write)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
