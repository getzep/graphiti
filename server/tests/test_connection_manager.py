from datetime import datetime, timedelta, timezone
from urllib.parse import parse_qs, urlparse

import pytest

from graph_service.config import Settings
from graph_service.connections.manager import (
    ConnectionManager,
    ConnectionManagerError,
    load_or_create_encryption_key,
    normalize_public_base_url,
)
from graph_service.connections.providers import OAuthIdentity, OAuthStart, OAuthTokens
from graph_service.connections.store import ConnectionStore


class FakeProvider:
    configured = True

    async def start(self, redirect_uri: str, state: str) -> OAuthStart:
        return OAuthStart(
            authorization_url=f'https://provider.example/authorize?state={state}',
            code_verifier='pkce-verifier',
            client_id='dynamic-client-id',
            client_secret='dynamic-client-secret',
            token_url='https://provider.example/token',
            metadata={'tenant': 'test'},
        )

    async def exchange(self, **kwargs):
        assert kwargs['code'] == 'one-time-code'
        assert kwargs['code_verifier'] == 'pkce-verifier'
        assert kwargs['client_id'] == 'dynamic-client-id'
        assert kwargs['client_secret'] == 'dynamic-client-secret'
        return OAuthTokens(
            access_token='oauth-access-token',
            refresh_token='oauth-refresh-token',
            expires_in=3600,
            scope='files.read offline_access',
        )

    async def identity(self, access_token: str) -> OAuthIdentity:
        assert access_token == 'oauth-access-token'
        return OAuthIdentity(account_id='user-1', account_name='测试账号', tenant_id='tenant-1')

    async def refresh(self, refresh_token: str, client_id: str) -> OAuthTokens:
        assert refresh_token == 'oauth-refresh-token'
        assert client_id == 'dynamic-client-id'
        return OAuthTokens(
            access_token='refreshed-token', refresh_token='rotated-token', expires_in=3600
        )

    async def resources(self, access_token: str, *, parent_id: str = '', page_token: str = ''):
        assert access_token in {'oauth-access-token', 'refreshed-token'}
        return {'items': [], 'parent_id': parent_id, 'next_page': page_token}


def _settings(tmp_path) -> Settings:
    return Settings(
        _env_file=None,
        source_state_path=tmp_path / 'state.db',
        upload_root=tmp_path / 'uploads',
        falkordb_port=6379,
    )


def test_local_key_is_persistent_and_public_url_is_restricted(tmp_path):
    settings = _settings(tmp_path)
    first = load_or_create_encryption_key(settings)
    second = load_or_create_encryption_key(settings)
    assert first == second
    assert len(first) == 32
    assert normalize_public_base_url(None, 'http://localhost:8000/') == 'http://localhost:8000'
    with pytest.raises(ConnectionManagerError):
        normalize_public_base_url(None, 'http://studio.example.test/')
    assert (
        normalize_public_base_url('https://studio.example.test', 'http://localhost:8000')
        == 'https://studio.example.test'
    )


@pytest.mark.asyncio
async def test_oauth_state_callback_persists_encrypted_connection_and_rejects_replay(tmp_path):
    settings = _settings(tmp_path)
    store = ConnectionStore(settings.source_state_path, b'manager-test-key-32-bytes-long!!')
    manager = ConnectionManager(settings, store)
    fake = FakeProvider()
    manager.providers['feishu'] = fake  # type: ignore[assignment]

    authorization_url = await manager.start(
        'feishu',
        public_base_url='http://localhost:8000',
        browser_session='browser-session-value-with-enough-entropy',
    )
    state = parse_qs(urlparse(authorization_url).query)['state'][0]
    connection = await manager.complete(
        'feishu',
        state=state,
        browser_session='browser-session-value-with-enough-entropy',
        code='one-time-code',
        oauth_error=None,
    )

    assert connection['provider'] == 'feishu'
    assert connection['display_name'] == '测试账号'
    assert 'access_token' not in connection
    assert await manager.get_access_token(connection['id'], 'feishu') == 'oauth-access-token'
    assert await manager.resources(connection['id'], parent_id='folder-1', page='next') == {
        'items': [],
        'parent_id': 'folder-1',
        'next_page': 'next',
    }
    with pytest.raises(ConnectionManagerError):
        await manager.complete(
            'feishu',
            state=state,
            browser_session='browser-session-value-with-enough-entropy',
            code='one-time-code',
            oauth_error=None,
        )


@pytest.mark.asyncio
async def test_expiring_connection_refreshes_with_encrypted_provider_context(tmp_path):
    settings = _settings(tmp_path)
    store = ConnectionStore(settings.source_state_path, b'manager-test-key-32-bytes-long!!')
    manager = ConnectionManager(settings, store)
    manager.providers['feishu'] = FakeProvider()  # type: ignore[assignment]
    connection = store.create_connection(
        provider='feishu',
        access_token='old-token',
        refresh_token='oauth-refresh-token',
        provider_context={'client_id': 'dynamic-client-id'},
        expires_at=datetime.now(timezone.utc) + timedelta(seconds=10),
    )

    assert await manager.get_access_token(connection['id'], 'feishu') == 'refreshed-token'
    tokens = store.get_connection_tokens(connection['id'])
    assert tokens.refresh_token == 'rotated-token'
    assert tokens.provider_context == {'client_id': 'dynamic-client-id'}
