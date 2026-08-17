import sqlite3
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

import pytest

from graph_service.connections import (
    AesGcmCipher,
    ConnectionStore,
    DecryptionError,
    OAuthStateConsumedError,
    OAuthStateExpiredError,
    OAuthStateNotFoundError,
    hash_oauth_value,
)
from graph_service.sources.store import SourceStore

KEY = b'connection-store-test-key-32b!!!'


def test_aes_gcm_uses_random_nonces_and_authenticates_associated_data():
    cipher = AesGcmCipher(KEY)

    first = cipher.encrypt_text('sensitive-token', associated_data=b'connection:one')
    second = cipher.encrypt_text('sensitive-token', associated_data=b'connection:one')

    assert first.nonce != second.nonce
    assert first.ciphertext != second.ciphertext
    assert b'sensitive-token' not in first.ciphertext
    assert (
        cipher.decrypt_text(
            first.nonce,
            first.ciphertext,
            associated_data=b'connection:one',
        )
        == 'sensitive-token'
    )
    with pytest.raises(DecryptionError):
        cipher.decrypt_text(
            first.nonce,
            first.ciphertext,
            associated_data=b'connection:two',
        )


def test_connection_crud_coexists_with_source_store_and_never_exposes_tokens(tmp_path):
    path = tmp_path / 'shared.db'
    source_store = SourceStore(path)
    source = source_store.create_source(
        kind='local', name='Local', group_id='neo4j', config={}, enabled=True
    )
    store = ConnectionStore(path, KEY)
    expires_at = datetime(2026, 8, 17, 10, tzinfo=timezone.utc)

    public = store.create_connection(
        connection_id='connection-one',
        provider='feishu',
        display_name='工作账号',
        subject_id='subject-1',
        tenant_id='tenant-1',
        scopes=['drive:read', 'drive:read'],
        access_token='access-secret',
        refresh_token='refresh-secret',
        provider_context={'client_id': 'dynamic-client-secret-id'},
        expires_at=expires_at,
    )

    assert public['id'] == 'connection-one'
    assert public['scopes'] == ['drive:read']
    assert public['token_version'] == 1
    assert {'access_token', 'refresh_token', 'ciphertext', 'nonce'}.isdisjoint(public)
    assert store.get_connection('connection-one') == public
    assert store.list_connections(provider='feishu') == [public]
    assert source_store.get_source(source['id'])['name'] == 'Local'

    tokens = store.get_connection_tokens('connection-one')
    assert tokens.access_token == 'access-secret'
    assert tokens.refresh_token == 'refresh-secret'
    assert tokens.provider_context == {'client_id': 'dynamic-client-secret-id'}
    assert tokens.expires_at == expires_at

    with sqlite3.connect(path) as connection:
        raw = connection.execute(
            """
            SELECT access_token_nonce, access_token_ciphertext,
                   refresh_token_nonce, refresh_token_ciphertext,
                   provider_context_nonce, provider_context_ciphertext
            FROM oauth_connections WHERE id = 'connection-one'
            """
        ).fetchone()
    stored_bytes = b''.join(bytes(value) for value in raw if value is not None)
    assert b'access-secret' not in stored_bytes
    assert b'refresh-secret' not in stored_bytes
    assert b'dynamic-client-secret-id' not in stored_bytes

    metadata_update = store.update_connection(
        'connection-one', display_name='新名称', status='error', last_error='需要重试'
    )
    assert metadata_update['display_name'] == '新名称'
    assert metadata_update['token_version'] == 1

    token_update = store.update_connection(
        'connection-one',
        status='active',
        access_token='new-access-secret',
        refresh_token=None,
        expires_at=expires_at + timedelta(hours=1),
        last_error=None,
    )
    assert token_update['token_version'] == 2
    updated_tokens = store.get_connection_tokens('connection-one')
    assert updated_tokens.access_token == 'new-access-secret'
    assert updated_tokens.refresh_token is None
    assert updated_tokens.expires_at == expires_at + timedelta(hours=1)

    context_update = store.update_connection(
        'connection-one', provider_context={'client_id': 'rotated-client-id'}
    )
    assert context_update['token_version'] == 3
    assert store.get_connection_tokens('connection-one').provider_context == {
        'client_id': 'rotated-client-id'
    }

    store.delete_connection('connection-one')
    with pytest.raises(KeyError):
        store.get_connection('connection-one')
    with pytest.raises(KeyError):
        store.delete_connection('connection-one')


def test_state_is_hashed_pkce_is_encrypted_and_consumption_is_bound(tmp_path):
    path = tmp_path / 'state.db'
    store = ConnectionStore(path, KEY)
    session_hash = hash_oauth_value('browser-session')
    created_at = datetime(2026, 8, 17, 10, tzinfo=timezone.utc)

    state = store.create_state(
        state='fixed-browser-state-value-for-test',
        provider='feishu',
        session_hash=session_hash,
        pkce_verifier='pkce-secret-verifier',
        context={
            'client_id': 'state-client-secret-id',
            'redirect_uri': 'https://studio.example.test/api/oauth/feishu/callback',
        },
        ttl_seconds=600,
        now=created_at,
    )

    with sqlite3.connect(path) as connection:
        row = connection.execute(
            """
            SELECT state_hash, session_hash,
                   pkce_verifier_nonce, pkce_verifier_ciphertext
            FROM oauth_states
            """
        ).fetchone()
    assert state == 'fixed-browser-state-value-for-test'
    assert row[0] == hash_oauth_value(state)
    assert row[0] != state.encode()
    assert row[1] == session_hash
    persisted = b''.join(bytes(value) for value in row)
    assert state.encode() not in persisted
    assert b'pkce-secret-verifier' not in persisted
    with sqlite3.connect(path) as connection:
        database_bytes = b''.join(
            bytes(value) if isinstance(value, bytes) else str(value).encode()
            for row_value in connection.execute('SELECT * FROM oauth_states').fetchall()
            for value in row_value
            if value is not None
        )
    assert b'state-client-secret-id' not in database_bytes
    assert b'studio.example.test' not in database_bytes

    with pytest.raises(OAuthStateNotFoundError):
        store.consume_state(
            state=state,
            provider='meego',
            session_hash=session_hash,
            now=created_at + timedelta(seconds=1),
        )
    with pytest.raises(OAuthStateNotFoundError):
        store.consume_state(
            state=state,
            provider='feishu',
            session_hash=hash_oauth_value('other-browser'),
            now=created_at + timedelta(seconds=1),
        )

    consumed = store.consume_state(
        state=state,
        provider='feishu',
        session_hash=session_hash,
        now=created_at + timedelta(seconds=2),
    )
    assert consumed.provider == 'feishu'
    assert consumed.pkce_verifier == 'pkce-secret-verifier'
    assert consumed.context['client_id'] == 'state-client-secret-id'
    assert consumed.session_hash == session_hash
    assert consumed.created_at == created_at
    assert consumed.expires_at == created_at + timedelta(seconds=600)

    with pytest.raises(OAuthStateConsumedError):
        store.consume_state(
            state=state,
            provider='feishu',
            session_hash=session_hash,
            now=created_at + timedelta(seconds=3),
        )


def test_state_expiry_and_cleanup(tmp_path):
    store = ConnectionStore(tmp_path / 'state.db', KEY)
    session_hash = hash_oauth_value('browser-session')
    created_at = datetime(2026, 8, 17, 10, tzinfo=timezone.utc)
    state = store.create_state(
        provider='feishu',
        session_hash=session_hash,
        pkce_verifier='pkce-verifier',
        ttl_seconds=10,
        now=created_at,
    )

    with pytest.raises(OAuthStateExpiredError):
        store.consume_state(
            state=state,
            provider='feishu',
            session_hash=session_hash,
            now=created_at + timedelta(seconds=10),
        )
    assert store.delete_expired_states(now=created_at + timedelta(seconds=10)) == 1
    with pytest.raises(OAuthStateNotFoundError):
        store.consume_state(
            state=state,
            provider='feishu',
            session_hash=session_hash,
            now=created_at + timedelta(seconds=11),
        )


def test_state_can_only_be_consumed_once_across_connections(tmp_path):
    store = ConnectionStore(tmp_path / 'state.db', KEY)
    session_hash = hash_oauth_value('browser-session')
    created_at = datetime(2026, 8, 17, 10, tzinfo=timezone.utc)
    state = store.create_state(
        provider='feishu',
        session_hash=session_hash,
        pkce_verifier='pkce-verifier',
        now=created_at,
    )

    def consume(index: int) -> str:
        try:
            store.consume_state(
                state=state,
                provider='feishu',
                session_hash=session_hash,
                now=created_at + timedelta(seconds=index + 1),
            )
            return 'consumed'
        except OAuthStateConsumedError:
            return 'already-consumed'

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(consume, range(8)))

    assert results.count('consumed') == 1
    assert results.count('already-consumed') == 7
