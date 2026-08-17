from __future__ import annotations

import asyncio
import base64
import binascii
import os
import secrets
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx

from graph_service.config import Settings

from .providers import (
    FeishuOAuthProvider,
    MeegoOAuthProvider,
    OAuthProviderError,
)
from .store import ConnectionStore, hash_oauth_value

_LOCAL_HOSTS = frozenset({'localhost', '127.0.0.1', '::1'})


class ConnectionManagerError(RuntimeError):
    """A credential-free connection error safe to expose through the Studio API."""


def _safe_provider_error(exc: BaseException, fallback: str) -> str:
    return str(exc) if isinstance(exc, OAuthProviderError) else fallback


def _decode_encryption_key(value: str) -> bytes:
    value = value.strip()
    if len(value) == 64:
        try:
            key = bytes.fromhex(value)
        except ValueError:
            key = b''
        if len(key) == 32:
            return key
    try:
        key = base64.urlsafe_b64decode(value + '=' * (-len(value) % 4))
    except (binascii.Error, ValueError):
        key = b''
    if len(key) != 32:
        raise ConnectionManagerError(
            'OAUTH_TOKEN_ENCRYPTION_KEY 必须是 32 字节密钥的 Base64URL 或 64 位十六进制'
        )
    return key


def load_or_create_encryption_key(settings: Settings) -> bytes:
    """Load the configured key, or create a persistent key for local single-node use."""
    if settings.oauth_token_encryption_key:
        return _decode_encryption_key(settings.oauth_token_encryption_key)

    state_path = Path(settings.source_state_path)
    key_path = state_path.with_suffix(f'{state_path.suffix}.oauth-key')
    key_path.parent.mkdir(parents=True, exist_ok=True)
    generated = secrets.token_bytes(32)
    encoded = base64.urlsafe_b64encode(generated).rstrip(b'=')
    try:
        descriptor = os.open(key_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        try:
            saved = key_path.read_bytes()
        except OSError as exc:
            raise ConnectionManagerError('无法读取本地 OAuth 加密密钥') from exc
        return _decode_encryption_key(saved.decode('ascii'))
    try:
        os.write(descriptor, encoded)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return generated


def normalize_public_base_url(configured: str | None, request_base_url: str) -> str:
    value = (configured or request_base_url).strip().rstrip('/')
    parsed = urlparse(value)
    if (
        parsed.scheme not in {'http', 'https'}
        or not parsed.hostname
        or parsed.username
        or parsed.password
        or parsed.path not in {'', '/'}
        or parsed.query
        or parsed.fragment
    ):
        raise ConnectionManagerError('OAUTH_PUBLIC_BASE_URL 必须是站点根地址')
    if parsed.scheme != 'https' and parsed.hostname not in _LOCAL_HOSTS:
        raise ConnectionManagerError('非本机 OAuth 回调必须使用 HTTPS')
    if configured is None and parsed.hostname not in _LOCAL_HOSTS:
        raise ConnectionManagerError('非本机部署必须显式配置 OAUTH_PUBLIC_BASE_URL')
    return f'{parsed.scheme}://{parsed.netloc}'


class ConnectionManager:
    def __init__(self, settings: Settings, store: ConnectionStore):
        self.settings = settings
        self.store = store
        self.providers = {
            'feishu': FeishuOAuthProvider(settings),
            'meego': MeegoOAuthProvider(settings),
        }
        self._refresh_locks: dict[str, asyncio.Lock] = {}

    def provider_status(self) -> dict[str, bool]:
        return {name: provider.configured for name, provider in self.providers.items()}

    def _provider(self, provider_name: str):
        provider = self.providers.get(provider_name)
        if provider is None:
            raise ConnectionManagerError('不支持的 OAuth 服务')
        if not provider.configured:
            raise ConnectionManagerError(f'{provider_name} OAuth 服务未就绪')
        return provider

    def validate_connection(self, connection_id: str, provider_name: str) -> dict[str, Any]:
        try:
            connection = self.store.get_connection(connection_id)
        except (KeyError, ValueError) as exc:
            raise ConnectionManagerError('连接账号不存在') from exc
        if connection['provider'] != provider_name:
            raise ConnectionManagerError('连接账号与数据源类型不匹配')
        if connection['status'] != 'active':
            raise ConnectionManagerError('连接账号需要重新授权')
        return connection

    async def start(
        self,
        provider_name: str,
        *,
        public_base_url: str,
        browser_session: str,
    ) -> str:
        provider = self._provider(provider_name)
        redirect_uri = f'{public_base_url}/api/oauth/{provider_name}/callback'
        state = secrets.token_urlsafe(32)
        try:
            start = await provider.start(redirect_uri, state)
            self.store.create_state(
                state=state,
                provider=provider_name,
                session_hash=hash_oauth_value(browser_session),
                pkce_verifier=start.code_verifier,
                context={
                    'client_id': start.client_id,
                    'client_secret': start.client_secret,
                    'token_url': start.token_url,
                    'redirect_uri': redirect_uri,
                    'metadata': start.metadata,
                },
                ttl_seconds=self.settings.oauth_state_ttl_seconds,
            )
        except (OAuthProviderError, httpx.HTTPError) as exc:
            raise ConnectionManagerError(
                _safe_provider_error(exc, f'{provider_name} 授权服务暂时不可用')
            ) from exc
        return start.authorization_url

    async def complete(
        self,
        provider_name: str,
        *,
        state: str,
        browser_session: str,
        code: str | None,
        oauth_error: str | None,
    ) -> dict[str, Any]:
        provider = self._provider(provider_name)
        try:
            consumed = self.store.consume_state(
                state=state,
                provider=provider_name,
                session_hash=hash_oauth_value(browser_session),
            )
        except (KeyError, ValueError) as exc:
            raise ConnectionManagerError('授权请求已失效，请重新连接') from exc
        if oauth_error:
            raise ConnectionManagerError('你已取消授权，账号未连接')
        if not code:
            raise ConnectionManagerError('授权回调缺少授权码，请重新连接')

        context = consumed.context
        try:
            tokens = await provider.exchange(
                code=code,
                redirect_uri=str(context.get('redirect_uri') or ''),
                code_verifier=consumed.pkce_verifier,
                client_id=str(context.get('client_id') or ''),
                client_secret=str(context.get('client_secret') or ''),
                token_url=str(context.get('token_url') or ''),
            )
            identity = await provider.identity(tokens.access_token)
        except (OAuthProviderError, httpx.HTTPError) as exc:
            raise ConnectionManagerError(
                _safe_provider_error(exc, f'{provider_name} 授权失败，请重试')
            ) from exc

        expires_at = (
            datetime.now(timezone.utc) + timedelta(seconds=tokens.expires_in)
            if tokens.expires_in > 0
            else None
        )
        provider_context = {
            'client_id': str(context.get('client_id') or ''),
            'token_url': str(context.get('token_url') or ''),
            'metadata': context.get('metadata')
            if isinstance(context.get('metadata'), dict)
            else {},
        }
        existing = next(
            (
                item
                for item in self.store.list_connections(provider=provider_name)
                if item.get('subject_id') == identity.account_id
                and item.get('tenant_id') == identity.tenant_id
            ),
            None,
        )
        if existing:
            return self.store.update_connection(
                str(existing['id']),
                access_token=tokens.access_token,
                refresh_token=tokens.refresh_token or None,
                provider_context=provider_context,
                expires_at=expires_at,
                display_name=identity.account_name,
                subject_id=identity.account_id,
                tenant_id=identity.tenant_id,
                scopes=tokens.scope.split(),
                status='active',
                last_error=None,
            )
        return self.store.create_connection(
            provider=provider_name,
            access_token=tokens.access_token,
            refresh_token=tokens.refresh_token or None,
            provider_context=provider_context,
            expires_at=expires_at,
            display_name=identity.account_name,
            subject_id=identity.account_id,
            tenant_id=identity.tenant_id,
            scopes=tokens.scope.split(),
        )

    async def get_access_token(self, connection_id: str, provider_name: str) -> str:
        self.validate_connection(connection_id, provider_name)
        tokens = self.store.get_connection_tokens(connection_id)
        refresh_before = datetime.now(timezone.utc) + timedelta(seconds=120)
        if tokens.expires_at is None or tokens.expires_at > refresh_before:
            return tokens.access_token

        lock = self._refresh_locks.setdefault(connection_id, asyncio.Lock())
        async with lock:
            self.validate_connection(connection_id, provider_name)
            tokens = self.store.get_connection_tokens(connection_id)
            if tokens.expires_at is None or tokens.expires_at > refresh_before:
                return tokens.access_token
            if not tokens.refresh_token:
                self.store.update_connection(
                    connection_id,
                    status='reauth_required',
                    last_error='授权已过期，请重新连接',
                )
                raise ConnectionManagerError('连接账号需要重新授权')
            provider = self._provider(provider_name)
            try:
                refreshed = await provider.refresh(
                    tokens.refresh_token,
                    str(tokens.provider_context.get('client_id') or ''),
                )
            except (OAuthProviderError, httpx.HTTPError) as exc:
                self.store.update_connection(
                    connection_id,
                    status='reauth_required',
                    last_error='授权刷新失败，请重新连接',
                )
                raise ConnectionManagerError(
                    _safe_provider_error(exc, '授权刷新失败，请重新连接')
                ) from exc
            expires_at = (
                datetime.now(timezone.utc) + timedelta(seconds=refreshed.expires_in)
                if refreshed.expires_in > 0
                else None
            )
            self.store.update_connection(
                connection_id,
                access_token=refreshed.access_token,
                refresh_token=refreshed.refresh_token or tokens.refresh_token,
                expires_at=expires_at,
                status='active',
                last_error=None,
            )
            return refreshed.access_token

    async def resources(
        self,
        connection_id: str,
        *,
        parent_id: str = '',
        page: str = '',
    ) -> dict[str, Any]:
        try:
            connection = self.store.get_connection(connection_id)
        except (KeyError, ValueError) as exc:
            raise ConnectionManagerError('连接账号不存在') from exc
        provider_name = str(connection['provider'])
        token = await self.get_access_token(connection_id, provider_name)
        provider = self._provider(provider_name)
        try:
            return await provider.resources(token, parent_id=parent_id, page_token=page)
        except (OAuthProviderError, httpx.HTTPError) as exc:
            raise ConnectionManagerError(
                _safe_provider_error(exc, '读取远程资源失败，请稍后重试')
            ) from exc

    async def meego_call(
        self,
        connection_id: str,
        resource: str,
        method: str,
        fallback: str,
        arguments: dict[str, Any],
    ) -> Any:
        token = await self.get_access_token(connection_id, 'meego')
        provider = self.providers['meego']
        try:
            return await provider.business_call(
                token,
                resource=resource,
                method=method,
                fallback=fallback,
                arguments=arguments,
            )
        except (OAuthProviderError, httpx.HTTPError) as exc:
            raise ConnectionManagerError(
                _safe_provider_error(exc, '读取 MeeGo 数据失败，请稍后重试')
            ) from exc

    async def meego_views(
        self,
        connection_id: str,
        *,
        project_key: str,
        query: str = '',
    ) -> list[dict[str, str]]:
        token = await self.get_access_token(connection_id, 'meego')
        provider = self.providers['meego']
        try:
            return await provider.search_views(
                token,
                project_key=project_key,
                query=query,
                view_scope='story',
            )
        except (OAuthProviderError, httpx.HTTPError) as exc:
            raise ConnectionManagerError(
                _safe_provider_error(exc, '读取 MeeGo 需求视图失败，请稍后重试')
            ) from exc
