#!/usr/bin/env python3
"""Historical transcript importer (FR-1).

Batch-loads session logs into Neo4j as Episode/Message nodes.

Behavior:
- Parses agents/*/sessions/*.jsonl with the same filtering as
  ingest/parse_sessions_v1.py (user + assistant messages, excluding
  assistant thinking blocks and clawdbot delivery-mirror noise).
- Upserts (:Episode {episode_id}) and (:Message {message_id}) nodes
  with a (:Episode)-[:HAS_MESSAGE]->(:Message) relationship.
- Computes content_hash for idempotent re-embedding skips.
- Backfills content_hash and graphiti_extracted_at on pre-existing rows.
- Emits structured stats JSON on completion.

Usage:
  python3 scripts/import_transcripts_to_neo4j.py \\
      --sessions-dir ~/.clawdbot/agents/main/sessions \\
      --confirm

  python3 scripts/import_transcripts_to_neo4j.py \\
      --sessions-dir ~/.clawdbot \\
      --dry-run --max-files 5
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_EMBEDDING_MODEL = 'embeddinggemma'
DEFAULT_EMBEDDING_DIM = 768

NEO4J_ENV_FALLBACK_FILE = Path.home() / '.clawdbot' / 'credentials' / 'neo4j.env'
NEO4J_NON_DEV_FALLBACK_OPT_IN_ENV = 'OM_NEO4J_ENV_FALLBACK_NON_DEV'
_NON_DEV_ENV_MARKERS = {'prod', 'production', 'staging', 'stage', 'preprod', 'preview'}
_TRUTHY_ENV_VALUES = {'1', 'true', 'yes', 'on'}


# ---------------------------------------------------------------------------
# Stats accumulator
# ---------------------------------------------------------------------------

@dataclass
class ImportStats:
    files_seen: int = 0
    messages_seen: int = 0
    messages_inserted: int = 0
    messages_updated: int = 0
    embeddings_computed: int = 0
    embeddings_skipped: int = 0
    errors: list[dict[str, str]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            'files_seen': self.files_seen,
            'messages_seen': self.messages_seen,
            'messages_inserted': self.messages_inserted,
            'messages_updated': self.messages_updated,
            'embeddings_computed': self.embeddings_computed,
            'embeddings_skipped': self.embeddings_skipped,
            'errors': self.errors,
        }


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------

def normalize_text(value: str) -> str:
    text = unicodedata.normalize('NFKC', value or '')
    text = text.strip()
    return re.sub(r'\s+', ' ', text)


def sha256_hex(value: str) -> str:
    return hashlib.sha256(value.encode('utf-8')).hexdigest()


def now_iso() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace('+00:00', 'Z')
    )


def parse_iso(value: str | None) -> datetime | None:
    if not value:
        return None
    txt = str(value).strip()
    if not txt:
        return None
    try:
        if txt.endswith('Z'):
            return datetime.fromisoformat(txt[:-1]).replace(tzinfo=timezone.utc)
        dt = datetime.fromisoformat(txt)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None


def iso_or_none(dt: datetime | None) -> str | None:
    if not dt:
        return None
    return dt.astimezone(timezone.utc).isoformat().replace('+00:00', 'Z')


# ---------------------------------------------------------------------------
# JSONL parsing (mirrors ingest/parse_sessions_v1.py)
# ---------------------------------------------------------------------------

def read_jsonl(file_path: Path) -> Iterator[dict[str, Any]]:
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def extract_text_content(
    content: list[dict[str, Any]] | str | None,
    include_thinking: bool = False,
) -> str:
    """Extract text from message.content (same rules as v1 parser)."""
    if not content:
        return ''
    if isinstance(content, str):
        return content
    parts: list[str] = []
    for item in content:
        if not isinstance(item, dict):
            continue
        if item.get('type') == 'text':
            txt = item.get('text', '')
            if txt:
                parts.append(txt)
    return '\n'.join(parts).strip()


def parse_session_messages(
    file_path: Path,
) -> tuple[str, list[dict[str, Any]]]:
    """Parse a session JSONL file, returning (session_id, messages).

    Applies the same filtering as ingest/parse_sessions_v1.py:
    - type == "message" entries only
    - role in (user, assistant)
    - Exclude assistant when provider == "clawdbot" or model == "delivery-mirror"
    - Exclude empty content and thinking blocks

    Each message dict has keys: role, content, created_at, line_index.
    """
    entries = list(read_jsonl(file_path))

    # Derive session_id from the session entry or filename stem.
    session_meta = next((e for e in entries if e.get('type') == 'session'), {})
    session_id = str(session_meta.get('id') or file_path.stem)

    messages: list[dict[str, Any]] = []
    for line_index, entry in enumerate(entries):
        if entry.get('type') != 'message':
            continue

        msg = entry.get('message') or {}
        role = msg.get('role')
        if role not in ('user', 'assistant'):
            continue

        # Exclude clawdbot delivery-mirror duplicates.
        provider = msg.get('provider') or entry.get('provider')
        model = msg.get('model') or entry.get('model')
        if role == 'assistant' and (provider == 'clawdbot' or model == 'delivery-mirror'):
            continue

        txt = extract_text_content(msg.get('content'), include_thinking=False)
        if not txt.strip():
            continue

        ts = entry.get('timestamp') or ''
        messages.append({
            'role': role,
            'content': txt.strip(),
            'created_at': str(ts),
            'line_index': line_index,
        })

    return session_id, messages


# ---------------------------------------------------------------------------
# ID generation
# ---------------------------------------------------------------------------

def make_episode_id(source_file_path: str) -> str:
    return sha256_hex(f'episode|{source_file_path}')


def make_message_id(
    source_session_id: str,
    source_file_path: str,
    line_index: int,
    role: str,
    created_at: str,
    content: str,
) -> str:
    normalized = normalize_text(content)
    return sha256_hex(
        f'msg|{source_session_id}|{source_file_path}'
        f'|{line_index}|{role}|{created_at}|{normalized}'
    )


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

def _embedding_config() -> tuple[str, int]:
    model = (os.environ.get('OM_EMBEDDING_MODEL') or DEFAULT_EMBEDDING_MODEL).strip()
    if not model:
        model = DEFAULT_EMBEDDING_MODEL

    raw_dim = (os.environ.get('OM_EMBEDDING_DIM') or str(DEFAULT_EMBEDDING_DIM)).strip()
    try:
        dim = int(raw_dim)
    except ValueError as exc:
        raise RuntimeError(
            f'OM_EMBEDDING_DIM must be an integer, got: {raw_dim!r}'
        ) from exc
    if dim <= 0:
        raise RuntimeError(f'OM_EMBEDDING_DIM must be > 0, got: {dim}')

    return model, dim


def _validated_embedding_base_url() -> str:
    base = (
        os.environ.get('EMBEDDER_BASE_URL')
        or os.environ.get('OPENAI_BASE_URL')
        or ''
    ).strip()
    if not base:
        base = 'http://localhost:11434/v1'

    parsed = urllib.parse.urlparse(base)
    if parsed.scheme not in {'http', 'https'} or not parsed.netloc:
        raise RuntimeError('embedding base URL must be absolute http(s) URL')
    if parsed.username or parsed.password:
        raise RuntimeError('embedding base URL must not include credentials')
    if parsed.query or parsed.fragment:
        raise RuntimeError('embedding base URL must not include query/fragment')

    return base.rstrip('/')


def embed_text(content: str, *, embedding_model: str, embedding_dim: int) -> list[float]:
    """Embed text through an OpenAI-compatible /embeddings endpoint."""
    base = _validated_embedding_base_url()
    url = base + '/embeddings'

    payload = {
        'model': embedding_model,
        'input': content,
    }
    body = json.dumps(payload).encode('utf-8')

    headers: dict[str, str] = {'Content-Type': 'application/json'}
    api_key = os.environ.get('EMBEDDER_API_KEY') or os.environ.get('OPENAI_API_KEY')
    if api_key:
        headers['Authorization'] = f'Bearer {api_key}'

    req = urllib.request.Request(url, data=body, headers=headers, method='POST')
    timeout = int(os.environ.get('OM_EMBED_TIMEOUT_SECONDS', '20'))

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode('utf-8', errors='replace')
    except urllib.error.HTTPError as exc:
        details = exc.read().decode('utf-8', errors='replace')
        raise RuntimeError(f'embedding HTTP {exc.code}: {details}') from exc
    except Exception as exc:
        raise RuntimeError(f'embedding request failed: {exc}') from exc

    parsed_resp = json.loads(raw) if raw.strip() else {}
    data = parsed_resp.get('data')
    if not isinstance(data, list) or not data or not isinstance(data[0], dict):
        raise RuntimeError('embedding response missing data[0]')
    emb = data[0].get('embedding')
    if not isinstance(emb, list):
        raise RuntimeError('embedding response missing embedding vector')

    vector = [float(v) for v in emb]
    if len(vector) != embedding_dim:
        raise RuntimeError(
            f'embedding dim mismatch: got={len(vector)} expected={embedding_dim}'
        )
    return vector


# ---------------------------------------------------------------------------
# Neo4j connection (same pattern as om_compressor.py / om_fast_write.py)
# ---------------------------------------------------------------------------

def _is_truthy_env(name: str) -> bool:
    value = os.environ.get(name)
    if value is None:
        return False
    return value.strip().lower() in _TRUTHY_ENV_VALUES


def _is_non_dev_mode() -> bool:
    for key in ('OM_ENV', 'APP_ENV', 'ENVIRONMENT', 'NODE_ENV'):
        value = os.environ.get(key)
        if value and value.strip().lower() in _NON_DEV_ENV_MARKERS:
            return True
    return _is_truthy_env('CI')


def _allow_neo4j_env_fallback() -> bool:
    if not _is_non_dev_mode():
        return True
    return _is_truthy_env(NEO4J_NON_DEV_FALLBACK_OPT_IN_ENV)


def _load_neo4j_env_fallback() -> None:
    if os.environ.get('NEO4J_PASSWORD'):
        return
    if not _allow_neo4j_env_fallback():
        return
    if not NEO4J_ENV_FALLBACK_FILE.exists():
        return

    for raw_line in NEO4J_ENV_FALLBACK_FILE.read_text(
        encoding='utf-8', errors='ignore'
    ).splitlines():
        line = raw_line.strip()
        if not line or line.startswith('#') or '=' not in line:
            continue
        key, value = line.split('=', 1)
        key = key.strip()
        if key in {'NEO4J_PASSWORD', 'NEO4J_USER', 'NEO4J_URI'} and key not in os.environ:
            os.environ[key] = value.strip().strip('"').strip("'")


def neo4j_driver() -> Any:
    try:
        from neo4j import GraphDatabase  # type: ignore
    except Exception as exc:
        raise RuntimeError('neo4j driver is required (pip install neo4j)') from exc

    _load_neo4j_env_fallback()

    uri = os.environ.get('NEO4J_URI', 'bolt://localhost:7687')
    user = os.environ.get('NEO4J_USER', 'neo4j')
    password = os.environ.get('NEO4J_PASSWORD')
    if not password:
        raise RuntimeError('NEO4J_PASSWORD is required')

    return GraphDatabase.driver(uri, auth=(user, password))


# ---------------------------------------------------------------------------
# Neo4j schema / constraints
# ---------------------------------------------------------------------------

def ensure_constraints(session: Any) -> None:
    stmts = [
        'CREATE CONSTRAINT episode_episode_id IF NOT EXISTS '
        'FOR (e:Episode) REQUIRE e.episode_id IS UNIQUE',
        'CREATE CONSTRAINT message_message_id IF NOT EXISTS '
        'FOR (m:Message) REQUIRE m.message_id IS UNIQUE',
    ]
    for stmt in stmts:
        session.run(stmt).consume()


# ---------------------------------------------------------------------------
# Migration: backfill content_hash and graphiti_extracted_at
# ---------------------------------------------------------------------------

def run_migration(session: Any) -> int:
    """Add content_hash to pre-existing Message rows that lack it.

    Also ensures graphiti_extracted_at field exists (NULL) on rows that lack it.
    Returns number of rows patched.
    """
    result = session.run(
        """
        MATCH (m:Message)
        WHERE m.content_hash IS NULL AND m.content IS NOT NULL
        RETURN m.message_id AS message_id,
               m.content AS content
        """
    ).data()

    patched = 0
    for row in result:
        content = str(row.get('content') or '')
        content_hash = sha256_hex(normalize_text(content))
        session.run(
            """
            MATCH (m:Message {message_id: $message_id})
            SET m.content_hash = $content_hash
            SET m.graphiti_extracted_at = coalesce(m.graphiti_extracted_at, NULL)
            """,
            {'message_id': row['message_id'], 'content_hash': content_hash},
        ).consume()
        patched += 1

    return patched


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------

def discover_session_files(sessions_dir: Path, max_files: int | None = None) -> list[Path]:
    """Find .jsonl session files.

    Supports two layouts:
    1. sessions_dir contains *.jsonl directly
    2. sessions_dir is the base .clawdbot dir: agents/*/sessions/*.jsonl
    """
    files: list[Path] = []

    # Layout 1: direct *.jsonl in dir
    direct = sorted(sessions_dir.glob('*.jsonl'))
    if direct:
        files.extend(direct)

    # Layout 2: agents/*/sessions/*.jsonl (look under sessions_dir)
    agents_pattern = sessions_dir / 'agents' / '*' / 'sessions' / '*.jsonl'
    agents_files = sorted(sessions_dir.glob('agents/*/sessions/*.jsonl'))
    if agents_files:
        seen = {f.resolve() for f in files}
        for f in agents_files:
            if f.resolve() not in seen:
                files.append(f)

    # Sort by path for deterministic order.
    files.sort(key=lambda p: str(p))

    if max_files is not None and max_files > 0:
        files = files[:max_files]

    return files


# ---------------------------------------------------------------------------
# Core upsert logic
# ---------------------------------------------------------------------------

def upsert_episode_and_messages(
    session: Any,
    *,
    file_path: Path,
    session_id: str,
    messages: list[dict[str, Any]],
    embedding_model: str,
    embedding_dim: int,
    dry_run: bool,
    stats: ImportStats,
    max_messages: int | None,
) -> None:
    """Upsert one Episode and its Message nodes for a single JSONL file."""
    source_file_path = str(file_path)
    episode_id = make_episode_id(source_file_path)
    ts_now = now_iso()

    timestamps: list[datetime | None] = [parse_iso(m['created_at']) for m in messages]
    valid_timestamps = [t for t in timestamps if t is not None]
    started_at = iso_or_none(min(valid_timestamps)) if valid_timestamps else ts_now
    last_message_at = iso_or_none(max(valid_timestamps)) if valid_timestamps else ts_now

    if not dry_run:
        session.run(
            """
            MERGE (e:Episode {episode_id: $episode_id})
            ON CREATE SET
              e.source_session_id = $source_session_id,
              e.started_at = $started_at,
              e.last_message_at = $last_message_at,
              e.source_file_path = $source_file_path,
              e.message_count = $message_count,
              e.created_at = $created_at
            ON MATCH SET
              e.last_message_at = CASE
                WHEN e.last_message_at IS NULL
                  OR e.last_message_at < $last_message_at
                THEN $last_message_at
                ELSE e.last_message_at
              END,
              e.message_count = $message_count,
              e.source_file_path = $source_file_path
            """,
            {
                'episode_id': episode_id,
                'source_session_id': session_id,
                'started_at': started_at,
                'last_message_at': last_message_at,
                'source_file_path': source_file_path,
                'message_count': len(messages),
                'created_at': ts_now,
            },
        ).consume()

    for msg in messages:
        # Enforce global message limit.
        if max_messages is not None and stats.messages_seen >= max_messages:
            return

        stats.messages_seen += 1

        message_id = make_message_id(
            source_session_id=session_id,
            source_file_path=source_file_path,
            line_index=msg['line_index'],
            role=msg['role'],
            created_at=msg['created_at'],
            content=msg['content'],
        )
        content_hash = sha256_hex(normalize_text(msg['content']))

        if dry_run:
            stats.messages_inserted += 1
            stats.embeddings_computed += 1
            continue

        # Check if the message already exists.
        existing = session.run(
            """
            MATCH (m:Message {message_id: $message_id})
            RETURN m.content_hash AS content_hash,
                   m.embedding_model AS embedding_model,
                   m.embedding_dim AS embedding_dim,
                   m.content_embedding IS NOT NULL AS has_embedding
            """,
            {'message_id': message_id},
        ).single()

        if existing is not None:
            # ON MATCH path: check if we can skip re-embedding.
            existing_hash = existing.get('content_hash')
            existing_model = existing.get('embedding_model')
            existing_dim = existing.get('embedding_dim')
            has_embedding = bool(existing.get('has_embedding'))

            if (
                existing_hash == content_hash
                and has_embedding
                and existing_model == embedding_model
                and existing_dim == embedding_dim
            ):
                # Content unchanged, embedding valid - skip.
                stats.embeddings_skipped += 1
                continue

            # Missing or invalid embedding - recompute and patch.
            try:
                vector = embed_text(
                    normalize_text(msg['content']),
                    embedding_model=embedding_model,
                    embedding_dim=embedding_dim,
                )
            except Exception as exc:
                stats.errors.append({
                    'file': str(file_path),
                    'message_id': message_id,
                    'error': f'embedding_failed: {exc}',
                })
                continue

            stats.embeddings_computed += 1
            stats.messages_updated += 1

            session.run(
                """
                MATCH (m:Message {message_id: $message_id})
                SET m.content_hash = $content_hash,
                    m.content_embedding = $content_embedding,
                    m.embedding_model = $embedding_model,
                    m.embedding_dim = $embedding_dim
                """,
                {
                    'message_id': message_id,
                    'content_hash': content_hash,
                    'content_embedding': vector,
                    'embedding_model': embedding_model,
                    'embedding_dim': embedding_dim,
                },
            ).consume()
            continue

        # ON CREATE path: new message.
        try:
            vector = embed_text(
                normalize_text(msg['content']),
                embedding_model=embedding_model,
                embedding_dim=embedding_dim,
            )
        except Exception as exc:
            stats.errors.append({
                'file': str(file_path),
                'message_id': message_id,
                'error': f'embedding_failed: {exc}',
            })
            continue

        stats.embeddings_computed += 1
        stats.messages_inserted += 1

        created_at = msg['created_at'] or ts_now

        session.run(
            """
            MERGE (m:Message {message_id: $message_id})
            ON CREATE SET
              m.source_session_id = $source_session_id,
              m.role = $role,
              m.content = $content,
              m.created_at = $created_at,
              m.content_hash = $content_hash,
              m.content_embedding = $content_embedding,
              m.embedding_model = $embedding_model,
              m.embedding_dim = $embedding_dim,
              m.graphiti_extracted_at = NULL,
              m.om_extracted = false,
              m.om_extract_attempts = 0,
              m.om_dead_letter = false
            """,
            {
                'message_id': message_id,
                'source_session_id': session_id,
                'role': msg['role'],
                'content': msg['content'],
                'created_at': created_at,
                'content_hash': content_hash,
                'content_embedding': vector,
                'embedding_model': embedding_model,
                'embedding_dim': embedding_dim,
            },
        ).consume()

        # Create relationship.
        session.run(
            """
            MATCH (e:Episode {episode_id: $episode_id})
            MATCH (m:Message {message_id: $message_id})
            MERGE (e)-[:HAS_MESSAGE]->(m)
            """,
            {'episode_id': episode_id, 'message_id': message_id},
        ).consume()


# ---------------------------------------------------------------------------
# Main import orchestration
# ---------------------------------------------------------------------------

def run_import(args: argparse.Namespace) -> ImportStats:
    stats = ImportStats()
    embedding_model, embedding_dim = _embedding_config()

    sessions_dir = Path(args.sessions_dir).expanduser().resolve()
    if not sessions_dir.exists():
        print(f'ERROR: sessions directory not found: {sessions_dir}', file=sys.stderr)
        return stats

    files = discover_session_files(sessions_dir, max_files=args.max_files)
    if not files:
        print(f'No .jsonl files found in {sessions_dir}', file=sys.stderr)
        return stats

    print(f'Discovered {len(files)} session file(s)')
    print(f'Embedding: model={embedding_model} dim={embedding_dim}')
    print(f'Mode: {"DRY-RUN" if args.dry_run else "LIVE WRITE"}')

    if args.dry_run:
        # Dry-run: parse and count without Neo4j.
        for file_path in files:
            stats.files_seen += 1
            try:
                session_id, messages = parse_session_messages(file_path)
                for msg in messages:
                    if args.max_messages and stats.messages_seen >= args.max_messages:
                        break
                    stats.messages_seen += 1
                    stats.messages_inserted += 1
                    stats.embeddings_computed += 1
            except Exception as exc:
                stats.errors.append({
                    'file': str(file_path),
                    'error': str(exc),
                })
        return stats

    # Live write path.
    driver = neo4j_driver()
    database = os.environ.get('NEO4J_DATABASE', 'neo4j')

    with driver:
        with driver.session(database=database) as neo_session:
            ensure_constraints(neo_session)

            # Migration: backfill content_hash on pre-existing rows.
            migrated = run_migration(neo_session)
            if migrated > 0:
                print(f'Migration: backfilled content_hash on {migrated} pre-existing row(s)')

            for file_path in files:
                if args.max_messages and stats.messages_seen >= args.max_messages:
                    break

                stats.files_seen += 1
                try:
                    session_id, messages = parse_session_messages(file_path)
                    if not messages:
                        continue

                    upsert_episode_and_messages(
                        neo_session,
                        file_path=file_path,
                        session_id=session_id,
                        messages=messages,
                        embedding_model=embedding_model,
                        embedding_dim=embedding_dim,
                        dry_run=False,
                        stats=stats,
                        max_messages=args.max_messages,
                    )
                except Exception as exc:
                    stats.errors.append({
                        'file': str(file_path),
                        'error': str(exc),
                    })

    return stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            'Import historical session transcripts into Neo4j '
            'as Episode/Message nodes (FR-1).'
        ),
    )
    parser.add_argument(
        '--sessions-dir',
        required=True,
        help=(
            'Path to sessions directory. Accepts *.jsonl directly '
            'or agents/*/sessions/*.jsonl layout.'
        ),
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        default=False,
        help='Preview without writing to Neo4j.',
    )
    parser.add_argument(
        '--confirm',
        action='store_true',
        default=False,
        help='Required for live writes (safety gate).',
    )
    parser.add_argument(
        '--max-files',
        type=int,
        default=None,
        help='Limit number of session files to process.',
    )
    parser.add_argument(
        '--max-messages',
        type=int,
        default=None,
        help='Limit total messages to process.',
    )
    parser.add_argument(
        '--stats-out',
        type=str,
        default=None,
        help='Write stats JSON to this file path.',
    )
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments. Convenience wrapper for build_parser()."""
    return build_parser().parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    # Safety gate: require --confirm or --dry-run.
    if not args.dry_run and not args.confirm:
        print(
            'ERROR: Live write requires --confirm flag. '
            'Use --dry-run to preview without writes.',
            file=sys.stderr,
        )
        return 2

    stats = run_import(args)

    # Emit structured stats.
    stats_dict = stats.to_dict()
    stats_json = json.dumps(stats_dict, indent=2, ensure_ascii=True)
    print(stats_json)

    # Write stats to file if requested.
    if args.stats_out:
        out_path = Path(args.stats_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(stats_json + '\n', encoding='utf-8')
        print(f'Stats written to: {out_path}')

    # Non-zero exit on errors.
    if stats.errors:
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
