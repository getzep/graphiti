from __future__ import annotations

import hashlib
import hmac
import json
import re
import secrets
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Final, cast
from uuid import uuid4

from .crypto import AesGcmCipher

_PROVIDER_PATTERN = re.compile(r'^[a-z][a-z0-9_-]{0,31}$')
_IDENTIFIER_PATTERN = re.compile(r'^[A-Za-z0-9_-]{1,128}$')
_CONNECTION_STATUSES: Final = frozenset({'active', 'disabled', 'error', 'reauth_required'})
_MAX_SECRET_CHARS = 128 * 1024
_MAX_STATE_TTL_SECONDS = 60 * 60


class OAuthStateError(ValueError):
    """Base class for rejected OAuth state consumption."""


class OAuthStateNotFoundError(OAuthStateError):
    """The state or its browser/provider binding is invalid."""


class OAuthStateExpiredError(OAuthStateError):
    """The state has exceeded its configured lifetime."""


class OAuthStateConsumedError(OAuthStateError):
    """The state was already consumed by a callback."""


@dataclass(frozen=True, slots=True)
class ConnectionTokens:
    access_token: str
    refresh_token: str | None
    provider_context: dict[str, Any]
    expires_at: datetime | None
    token_version: int


@dataclass(frozen=True, slots=True)
class ConsumedOAuthState:
    provider: str
    session_hash: bytes
    pkce_verifier: str
    context: dict[str, Any]
    created_at: datetime
    expires_at: datetime
    consumed_at: datetime


class _UnsetType:
    __slots__ = ()


UNSET: Final = _UnsetType()


def hash_oauth_value(value: str | bytes) -> bytes:
    """Return a fixed-width digest suitable for state and browser-session bindings."""
    if isinstance(value, str):
        value = value.encode('utf-8')
    if not isinstance(value, bytes) or not value:
        raise ValueError('value to hash must be non-empty')
    return hashlib.sha256(value).digest()


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _as_utc(value: datetime, *, field: str) -> datetime:
    if value.tzinfo is None:
        raise ValueError(f'{field} must include a timezone')
    return value.astimezone(timezone.utc)


def _to_iso(value: datetime | None) -> str | None:
    return value.astimezone(timezone.utc).isoformat() if value is not None else None


def _from_iso(value: str | None) -> datetime | None:
    return datetime.fromisoformat(value) if value is not None else None


def _provider(value: str) -> str:
    normalized = value.strip().lower()
    if not _PROVIDER_PATTERN.fullmatch(normalized):
        raise ValueError('provider has an invalid format')
    return normalized


def _identifier(value: str, *, field: str) -> str:
    value = value.strip()
    if not _IDENTIFIER_PATTERN.fullmatch(value):
        raise ValueError(f'{field} has an invalid format')
    return value


def _optional_text(value: str | None, *, field: str, max_chars: int = 1024) -> str | None:
    if value is None:
        return None
    value = value.strip()
    if not value:
        return None
    if len(value) > max_chars or any(ord(character) < 32 for character in value):
        raise ValueError(f'{field} has an invalid format')
    return value


def _secret(value: str, *, field: str) -> str:
    if not isinstance(value, str) or not value or len(value) > _MAX_SECRET_CHARS:
        raise ValueError(f'{field} must be a non-empty string within the size limit')
    return value


def _scopes(values: list[str] | tuple[str, ...] | None) -> list[str]:
    if values is None:
        return []
    if len(values) > 200:
        raise ValueError('scopes contains too many entries')
    normalized: list[str] = []
    for value in values:
        item = _optional_text(value, field='scope', max_chars=256)
        if item and item not in normalized:
            normalized.append(item)
    return normalized


def _context_json(value: dict[str, Any] | None, *, field: str) -> str:
    value = value or {}
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ValueError(f'{field} must be a JSON object with string keys')
    try:
        encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'))
    except (TypeError, ValueError) as exc:
        raise ValueError(f'{field} must contain JSON-compatible values') from exc
    if len(encoded.encode('utf-8')) > 32 * 1024:
        raise ValueError(f'{field} exceeds the size limit')
    return encoded


def _decoded_context(value: str, *, field: str) -> dict[str, Any]:
    try:
        decoded = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(f'{field} is invalid') from exc
    if not isinstance(decoded, dict):
        raise ValueError(f'{field} is invalid')
    return decoded


class ConnectionStore:
    """Encrypted OAuth state stored beside SourceStore in the same SQLite database."""

    def __init__(self, path: Path | str, encryption_key: bytes):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._cipher = AesGcmCipher(encryption_key)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute('PRAGMA foreign_keys = ON')
        connection.execute('PRAGMA journal_mode = WAL')
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS oauth_connections (
                    id TEXT PRIMARY KEY,
                    provider TEXT NOT NULL,
                    display_name TEXT,
                    subject_id TEXT,
                    tenant_id TEXT,
                    scopes_json TEXT NOT NULL DEFAULT '[]',
                    status TEXT NOT NULL DEFAULT 'active',
                    access_token_nonce BLOB NOT NULL,
                    access_token_ciphertext BLOB NOT NULL,
                    refresh_token_nonce BLOB,
                    refresh_token_ciphertext BLOB,
                    provider_context_nonce BLOB,
                    provider_context_ciphertext BLOB,
                    expires_at TEXT,
                    token_version INTEGER NOT NULL DEFAULT 1,
                    last_error TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    CHECK (
                        (refresh_token_nonce IS NULL AND refresh_token_ciphertext IS NULL)
                        OR
                        (refresh_token_nonce IS NOT NULL AND refresh_token_ciphertext IS NOT NULL)
                    ),
                    CHECK (
                        (provider_context_nonce IS NULL AND provider_context_ciphertext IS NULL)
                        OR
                        (
                            provider_context_nonce IS NOT NULL
                            AND provider_context_ciphertext IS NOT NULL
                        )
                    )
                );

                CREATE INDEX IF NOT EXISTS idx_oauth_connections_provider_status
                    ON oauth_connections(provider, status);

                CREATE TABLE IF NOT EXISTS oauth_states (
                    state_hash BLOB PRIMARY KEY,
                    provider TEXT NOT NULL,
                    session_hash BLOB NOT NULL,
                    pkce_verifier_nonce BLOB NOT NULL,
                    pkce_verifier_ciphertext BLOB NOT NULL,
                    context_nonce BLOB,
                    context_ciphertext BLOB,
                    expires_at TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    consumed_at TEXT,
                    CHECK (
                        (context_nonce IS NULL AND context_ciphertext IS NULL)
                        OR
                        (context_nonce IS NOT NULL AND context_ciphertext IS NOT NULL)
                    )
                );

                CREATE INDEX IF NOT EXISTS idx_oauth_states_expires_at
                    ON oauth_states(expires_at);
                """
            )
            # Early development databases may contain the tables without encrypted context
            # columns. Nullable columns preserve those rows; all newly created records write
            # both nonce and ciphertext.
            migrations = {
                'oauth_connections': (
                    ('provider_context_nonce', 'BLOB'),
                    ('provider_context_ciphertext', 'BLOB'),
                ),
                'oauth_states': (
                    ('context_nonce', 'BLOB'),
                    ('context_ciphertext', 'BLOB'),
                ),
            }
            for table, additions in migrations.items():
                columns = {
                    row['name']
                    for row in connection.execute(f'PRAGMA table_info({table})').fetchall()
                }
                for column, column_type in additions:
                    if column not in columns:
                        connection.execute(f'ALTER TABLE {table} ADD COLUMN {column} {column_type}')

    @staticmethod
    def _connection_public(row: sqlite3.Row) -> dict[str, Any]:
        return {
            'id': row['id'],
            'provider': row['provider'],
            'display_name': row['display_name'],
            'subject_id': row['subject_id'],
            'tenant_id': row['tenant_id'],
            'scopes': json.loads(row['scopes_json']),
            'status': row['status'],
            'expires_at': row['expires_at'],
            'token_version': row['token_version'],
            'last_error': row['last_error'],
            'created_at': row['created_at'],
            'updated_at': row['updated_at'],
        }

    @staticmethod
    def _token_aad(connection_id: str, provider: str, token_kind: str) -> bytes:
        return f'graphiti-studio:oauth-token:v1:{provider}:{connection_id}:{token_kind}'.encode()

    @staticmethod
    def _state_aad(state_hash: bytes, provider: str, session_hash: bytes) -> bytes:
        return b'|'.join(
            (
                b'graphiti-studio:oauth-state:v1',
                provider.encode('ascii'),
                state_hash.hex().encode('ascii'),
                session_hash.hex().encode('ascii'),
            )
        )

    @classmethod
    def _state_context_aad(cls, state_hash: bytes, provider: str, session_hash: bytes) -> bytes:
        return cls._state_aad(state_hash, provider, session_hash) + b'|context'

    def create_connection(
        self,
        *,
        provider: str,
        access_token: str,
        refresh_token: str | None = None,
        provider_context: dict[str, Any] | None = None,
        expires_at: datetime | None = None,
        display_name: str | None = None,
        subject_id: str | None = None,
        tenant_id: str | None = None,
        scopes: list[str] | tuple[str, ...] | None = None,
        status: str = 'active',
        connection_id: str | None = None,
        now: datetime | None = None,
    ) -> dict[str, Any]:
        provider = _provider(provider)
        connection_id = _identifier(connection_id or uuid4().hex, field='connection_id')
        access_token = _secret(access_token, field='access_token')
        refresh_token = (
            _secret(refresh_token, field='refresh_token') if refresh_token is not None else None
        )
        if status not in _CONNECTION_STATUSES:
            raise ValueError('status is invalid')
        if expires_at is not None:
            expires_at = _as_utc(expires_at, field='expires_at')
        now = _as_utc(now or _utc_now(), field='now')
        access = self._cipher.encrypt_text(
            access_token,
            associated_data=self._token_aad(connection_id, provider, 'access'),
        )
        refresh = (
            self._cipher.encrypt_text(
                refresh_token,
                associated_data=self._token_aad(connection_id, provider, 'refresh'),
            )
            if refresh_token is not None
            else None
        )
        context_json = _context_json(provider_context, field='provider_context')
        context = (
            self._cipher.encrypt_text(
                context_json,
                associated_data=self._token_aad(connection_id, provider, 'context'),
            )
            if provider_context
            else None
        )
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO oauth_connections (
                    id, provider, display_name, subject_id, tenant_id, scopes_json, status,
                    access_token_nonce, access_token_ciphertext,
                    refresh_token_nonce, refresh_token_ciphertext,
                    provider_context_nonce, provider_context_ciphertext,
                    expires_at, token_version, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?)
                """,
                (
                    connection_id,
                    provider,
                    _optional_text(display_name, field='display_name'),
                    _optional_text(subject_id, field='subject_id'),
                    _optional_text(tenant_id, field='tenant_id'),
                    json.dumps(_scopes(scopes), ensure_ascii=False),
                    status,
                    access.nonce,
                    access.ciphertext,
                    refresh.nonce if refresh else None,
                    refresh.ciphertext if refresh else None,
                    context.nonce if context else None,
                    context.ciphertext if context else None,
                    _to_iso(expires_at),
                    _to_iso(now),
                    _to_iso(now),
                ),
            )
        return self.get_connection(connection_id)

    def get_connection(self, connection_id: str) -> dict[str, Any]:
        connection_id = _identifier(connection_id, field='connection_id')
        with self._connect() as connection:
            row = connection.execute(
                'SELECT * FROM oauth_connections WHERE id = ?', (connection_id,)
            ).fetchone()
        if row is None:
            raise KeyError(connection_id)
        return self._connection_public(row)

    def list_connections(self, *, provider: str | None = None) -> list[dict[str, Any]]:
        with self._connect() as connection:
            if provider is None:
                rows = connection.execute(
                    'SELECT * FROM oauth_connections ORDER BY created_at DESC, id'
                ).fetchall()
            else:
                rows = connection.execute(
                    """
                    SELECT * FROM oauth_connections
                    WHERE provider = ? ORDER BY created_at DESC, id
                    """,
                    (_provider(provider),),
                ).fetchall()
        return [self._connection_public(row) for row in rows]

    def get_connection_tokens(self, connection_id: str) -> ConnectionTokens:
        connection_id = _identifier(connection_id, field='connection_id')
        with self._connect() as connection:
            row = connection.execute(
                'SELECT * FROM oauth_connections WHERE id = ?', (connection_id,)
            ).fetchone()
        if row is None:
            raise KeyError(connection_id)
        access_token = self._cipher.decrypt_text(
            row['access_token_nonce'],
            row['access_token_ciphertext'],
            associated_data=self._token_aad(connection_id, row['provider'], 'access'),
        )
        refresh_token = (
            self._cipher.decrypt_text(
                row['refresh_token_nonce'],
                row['refresh_token_ciphertext'],
                associated_data=self._token_aad(connection_id, row['provider'], 'refresh'),
            )
            if row['refresh_token_ciphertext'] is not None
            else None
        )
        provider_context = (
            _decoded_context(
                self._cipher.decrypt_text(
                    row['provider_context_nonce'],
                    row['provider_context_ciphertext'],
                    associated_data=self._token_aad(connection_id, row['provider'], 'context'),
                ),
                field='provider_context',
            )
            if row['provider_context_ciphertext'] is not None
            else {}
        )
        return ConnectionTokens(
            access_token=access_token,
            refresh_token=refresh_token,
            provider_context=provider_context,
            expires_at=_from_iso(row['expires_at']),
            token_version=row['token_version'],
        )

    def update_connection(
        self,
        connection_id: str,
        *,
        display_name: str | None | _UnsetType = UNSET,
        subject_id: str | None | _UnsetType = UNSET,
        tenant_id: str | None | _UnsetType = UNSET,
        scopes: list[str] | tuple[str, ...] | _UnsetType = UNSET,
        status: str | _UnsetType = UNSET,
        access_token: str | _UnsetType = UNSET,
        refresh_token: str | None | _UnsetType = UNSET,
        expires_at: datetime | None | _UnsetType = UNSET,
        last_error: str | None | _UnsetType = UNSET,
        now: datetime | None = None,
    ) -> dict[str, Any]:
        connection_id = _identifier(connection_id, field='connection_id')
        now = _as_utc(now or _utc_now(), field='now')
        with self._connect() as connection:
            connection.execute('BEGIN IMMEDIATE')
            row = connection.execute(
                'SELECT * FROM oauth_connections WHERE id = ?', (connection_id,)
            ).fetchone()
            if row is None:
                raise KeyError(connection_id)

            updates: list[str] = []
            parameters: list[Any] = []
            token_changed = False

            for column, value in (
                ('display_name', display_name),
                ('subject_id', subject_id),
                ('tenant_id', tenant_id),
                ('last_error', last_error),
            ):
                if value is not UNSET:
                    updates.append(f'{column} = ?')
                    parameters.append(_optional_text(cast(str | None, value), field=column))

            if scopes is not UNSET:
                updates.append('scopes_json = ?')
                parameters.append(
                    json.dumps(
                        _scopes(cast(list[str] | tuple[str, ...], scopes)), ensure_ascii=False
                    )
                )
            if status is not UNSET:
                if status not in _CONNECTION_STATUSES:
                    raise ValueError('status is invalid')
                updates.append('status = ?')
                parameters.append(status)
            if access_token is not UNSET:
                encrypted = self._cipher.encrypt_text(
                    _secret(cast(str, access_token), field='access_token'),
                    associated_data=self._token_aad(connection_id, row['provider'], 'access'),
                )
                updates.extend(('access_token_nonce = ?', 'access_token_ciphertext = ?'))
                parameters.extend((encrypted.nonce, encrypted.ciphertext))
                token_changed = True
            if refresh_token is not UNSET:
                if refresh_token is None:
                    updates.extend(
                        ('refresh_token_nonce = NULL', 'refresh_token_ciphertext = NULL')
                    )
                else:
                    encrypted = self._cipher.encrypt_text(
                        _secret(cast(str, refresh_token), field='refresh_token'),
                        associated_data=self._token_aad(connection_id, row['provider'], 'refresh'),
                    )
                    updates.extend(('refresh_token_nonce = ?', 'refresh_token_ciphertext = ?'))
                    parameters.extend((encrypted.nonce, encrypted.ciphertext))
                token_changed = True
            if expires_at is not UNSET:
                if expires_at is not None:
                    expires_at = _as_utc(cast(datetime, expires_at), field='expires_at')
                updates.append('expires_at = ?')
                parameters.append(_to_iso(expires_at))
                token_changed = True

            if updates:
                if token_changed:
                    updates.append('token_version = token_version + 1')
                updates.append('updated_at = ?')
                parameters.append(_to_iso(now))
                parameters.append(connection_id)
                connection.execute(
                    f'UPDATE oauth_connections SET {", ".join(updates)} WHERE id = ?',
                    parameters,
                )
        return self.get_connection(connection_id)

    def delete_connection(self, connection_id: str) -> None:
        connection_id = _identifier(connection_id, field='connection_id')
        with self._connect() as connection:
            cursor = connection.execute(
                'DELETE FROM oauth_connections WHERE id = ?', (connection_id,)
            )
        if cursor.rowcount == 0:
            raise KeyError(connection_id)

    def create_state(
        self,
        *,
        provider: str,
        session_hash: bytes,
        pkce_verifier: str,
        context: dict[str, Any] | None = None,
        state: str | None = None,
        ttl_seconds: int = 600,
        now: datetime | None = None,
    ) -> str:
        provider = _provider(provider)
        if not isinstance(session_hash, bytes) or len(session_hash) != hashlib.sha256().digest_size:
            raise ValueError('session_hash must be a SHA-256 digest')
        pkce_verifier = _secret(pkce_verifier, field='pkce_verifier')
        if state is not None:
            state = _secret(state, field='state')
        context_json = _context_json(context, field='context')
        if isinstance(ttl_seconds, bool) or not 1 <= ttl_seconds <= _MAX_STATE_TTL_SECONDS:
            raise ValueError('ttl_seconds must be between 1 and 3600')
        now = _as_utc(now or _utc_now(), field='now')
        expires_at = now + timedelta(seconds=ttl_seconds)

        for _ in range(1 if state is not None else 3):
            candidate = state or secrets.token_urlsafe(32)
            state_hash = hash_oauth_value(candidate)
            encrypted = self._cipher.encrypt_text(
                pkce_verifier,
                associated_data=self._state_aad(state_hash, provider, session_hash),
            )
            encrypted_context = (
                self._cipher.encrypt_text(
                    context_json,
                    associated_data=self._state_context_aad(state_hash, provider, session_hash),
                )
                if context
                else None
            )
            try:
                with self._connect() as connection:
                    connection.execute(
                        """
                        INSERT INTO oauth_states (
                            state_hash, provider, session_hash,
                            pkce_verifier_nonce, pkce_verifier_ciphertext,
                            context_nonce, context_ciphertext,
                            expires_at, created_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            state_hash,
                            provider,
                            session_hash,
                            encrypted.nonce,
                            encrypted.ciphertext,
                            encrypted_context.nonce if encrypted_context else None,
                            encrypted_context.ciphertext if encrypted_context else None,
                            _to_iso(expires_at),
                            _to_iso(now),
                        ),
                    )
                return candidate
            except sqlite3.IntegrityError:
                if state is not None:
                    raise ValueError('OAuth state already exists') from None
                continue
        raise RuntimeError('could not allocate a unique OAuth state')

    def consume_state(
        self,
        *,
        state: str,
        provider: str,
        session_hash: bytes,
        now: datetime | None = None,
    ) -> ConsumedOAuthState:
        provider = _provider(provider)
        if not isinstance(session_hash, bytes) or len(session_hash) != hashlib.sha256().digest_size:
            raise ValueError('session_hash must be a SHA-256 digest')
        state_hash = hash_oauth_value(state)
        now = _as_utc(now or _utc_now(), field='now')

        with self._connect() as connection:
            connection.execute('BEGIN IMMEDIATE')
            row = connection.execute(
                'SELECT * FROM oauth_states WHERE state_hash = ?', (state_hash,)
            ).fetchone()
            if row is None or not hmac.compare_digest(row['provider'], provider):
                raise OAuthStateNotFoundError('OAuth state is invalid')
            if not hmac.compare_digest(row['session_hash'], session_hash):
                raise OAuthStateNotFoundError('OAuth state is invalid')
            if row['consumed_at'] is not None:
                raise OAuthStateConsumedError('OAuth state was already consumed')
            expires_at = _from_iso(row['expires_at'])
            if expires_at is None or expires_at <= now:
                raise OAuthStateExpiredError('OAuth state has expired')
            consumed_at = _to_iso(now)
            cursor = connection.execute(
                """
                UPDATE oauth_states SET consumed_at = ?
                WHERE state_hash = ? AND consumed_at IS NULL
                """,
                (consumed_at, state_hash),
            )
            if cursor.rowcount != 1:
                raise OAuthStateConsumedError('OAuth state was already consumed')

        pkce_verifier = self._cipher.decrypt_text(
            row['pkce_verifier_nonce'],
            row['pkce_verifier_ciphertext'],
            associated_data=self._state_aad(state_hash, provider, session_hash),
        )
        context = (
            _decoded_context(
                self._cipher.decrypt_text(
                    row['context_nonce'],
                    row['context_ciphertext'],
                    associated_data=self._state_context_aad(state_hash, provider, session_hash),
                ),
                field='context',
            )
            if row['context_ciphertext'] is not None
            else {}
        )
        created_at = _from_iso(row['created_at'])
        if created_at is None:
            raise OAuthStateError('OAuth state has invalid timestamps')
        return ConsumedOAuthState(
            provider=provider,
            session_hash=session_hash,
            pkce_verifier=pkce_verifier,
            context=context,
            created_at=created_at,
            expires_at=expires_at,
            consumed_at=now,
        )

    def delete_expired_states(self, *, now: datetime | None = None) -> int:
        now = _as_utc(now or _utc_now(), field='now')
        with self._connect() as connection:
            cursor = connection.execute(
                'DELETE FROM oauth_states WHERE expires_at <= ?', (_to_iso(now),)
            )
        return cursor.rowcount
