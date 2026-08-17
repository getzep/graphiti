from __future__ import annotations

import base64
import hashlib
import json
import re
import secrets
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlencode, urlparse

import httpx

from graph_service.config import Settings


class OAuthProviderError(RuntimeError):
    """A sanitized OAuth/provider failure safe to show in Studio."""


@dataclass(slots=True)
class OAuthTokens:
    access_token: str
    refresh_token: str = ''
    expires_in: int = 0
    scope: str = ''


@dataclass(slots=True)
class OAuthStart:
    authorization_url: str
    code_verifier: str
    client_id: str
    client_secret: str = ''
    token_url: str = ''
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class OAuthIdentity:
    account_id: str
    account_name: str
    tenant_id: str = ''
    metadata: dict[str, Any] = field(default_factory=dict)


def generate_pkce() -> tuple[str, str]:
    verifier = secrets.token_urlsafe(64)[:96]
    digest = hashlib.sha256(verifier.encode()).digest()
    challenge = base64.urlsafe_b64encode(digest).rstrip(b'=').decode()
    return verifier, challenge


def _integer(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _json_object(response: httpx.Response, operation: str) -> dict[str, Any]:
    try:
        payload = response.json()
    except (json.JSONDecodeError, ValueError) as exc:
        raise OAuthProviderError(f'{operation}返回了无效响应') from exc
    if not isinstance(payload, dict):
        raise OAuthProviderError(f'{operation}返回了无效响应')
    return payload


def _check_feishu(payload: dict[str, Any], operation: str) -> dict[str, Any]:
    code = payload.get('code', 0)
    if code not in (0, '0', None):
        raise OAuthProviderError(f'{operation}失败：{payload.get("msg") or code}')
    data = payload.get('data')
    return data if isinstance(data, dict) else payload


class FeishuOAuthProvider:
    name = 'feishu'

    def __init__(self, settings: Settings):
        self.settings = settings

    @property
    def configured(self) -> bool:
        return bool(self.settings.feishu_app_id and self.settings.feishu_app_secret)

    async def start(self, redirect_uri: str, state: str) -> OAuthStart:
        if not self.configured:
            raise OAuthProviderError('管理员尚未配置飞书 OAuth 应用')
        verifier, challenge = generate_pkce()
        scopes = ' '.join(self.settings.feishu_oauth_scopes.split())
        query = urlencode(
            {
                'client_id': self.settings.feishu_app_id,
                'response_type': 'code',
                'redirect_uri': redirect_uri,
                'scope': scopes,
                'state': state,
                'code_challenge': challenge,
                'code_challenge_method': 'S256',
                'prompt': 'consent',
            }
        )
        return OAuthStart(
            authorization_url=f'{self.settings.feishu_authorize_url}?{query}',
            code_verifier=verifier,
            client_id=str(self.settings.feishu_app_id),
            client_secret=str(self.settings.feishu_app_secret),
            token_url=self.settings.feishu_token_url,
            metadata={'scope': scopes},
        )

    async def exchange(
        self,
        *,
        code: str,
        redirect_uri: str,
        code_verifier: str,
        client_id: str,
        client_secret: str,
        token_url: str,
    ) -> OAuthTokens:
        body = {
            'grant_type': 'authorization_code',
            'client_id': client_id,
            'client_secret': client_secret,
            'code': code,
            'redirect_uri': redirect_uri,
            'code_verifier': code_verifier,
        }
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(token_url, json=body)
        if response.status_code >= 400:
            raise OAuthProviderError('飞书授权码交换失败，请重新授权')
        data = _check_feishu(_json_object(response, '飞书授权'), '飞书授权')
        access_token = str(data.get('access_token') or '')
        if not access_token:
            raise OAuthProviderError('飞书授权响应缺少 access_token')
        return OAuthTokens(
            access_token=access_token,
            refresh_token=str(data.get('refresh_token') or ''),
            expires_in=_integer(data.get('expires_in')),
            scope=str(data.get('scope') or ''),
        )

    async def refresh(self, refresh_token: str, client_id: str) -> OAuthTokens:
        if not self.settings.feishu_app_secret:
            raise OAuthProviderError('管理员尚未配置飞书 OAuth 应用密钥')
        body = {
            'grant_type': 'refresh_token',
            'client_id': client_id,
            'client_secret': self.settings.feishu_app_secret,
            'refresh_token': refresh_token,
        }
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(self.settings.feishu_token_url, json=body)
        if response.status_code >= 400:
            raise OAuthProviderError('飞书授权已过期，请重新连接')
        data = _check_feishu(_json_object(response, '刷新飞书授权'), '刷新飞书授权')
        access_token = str(data.get('access_token') or '')
        if not access_token:
            raise OAuthProviderError('刷新飞书授权失败，请重新连接')
        return OAuthTokens(
            access_token=access_token,
            refresh_token=str(data.get('refresh_token') or refresh_token),
            expires_in=_integer(data.get('expires_in')),
            scope=str(data.get('scope') or ''),
        )

    async def identity(self, access_token: str) -> OAuthIdentity:
        headers = {'Authorization': f'Bearer {access_token}'}
        url = f'{self.settings.feishu_base_url.rstrip("/")}/authen/v1/user_info'
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.get(url, headers=headers)
        response.raise_for_status()
        data = _check_feishu(_json_object(response, '读取飞书用户信息'), '读取飞书用户信息')
        account_id = str(data.get('open_id') or data.get('union_id') or '')
        if not account_id:
            raise OAuthProviderError('飞书用户信息缺少用户标识')
        return OAuthIdentity(
            account_id=account_id,
            account_name=str(data.get('name') or data.get('en_name') or '飞书账号'),
            tenant_id=str(data.get('tenant_key') or ''),
            metadata={'avatar_url': str(data.get('avatar_url') or '')},
        )

    async def resources(
        self, access_token: str, *, parent_id: str = '', page_token: str = ''
    ) -> dict[str, Any]:
        params: dict[str, Any] = {
            'page_size': 200,
            'order_by': 'EditedTime',
            'direction': 'DESC',
        }
        if parent_id:
            params['folder_token'] = parent_id
        if page_token and parent_id:
            params['page_token'] = page_token
        headers = {'Authorization': f'Bearer {access_token}'}
        url = f'{self.settings.feishu_base_url.rstrip("/")}/drive/v1/files'
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.get(url, headers=headers, params=params)
        response.raise_for_status()
        data = _check_feishu(_json_object(response, '列举飞书云空间'), '列举飞书云空间')
        items = []
        for item in data.get('files') or []:
            if not isinstance(item, dict):
                continue
            resource_id = str(item.get('token') or '')
            resource_type = str(item.get('type') or '')
            if not resource_id:
                continue
            selectable = resource_type in {'folder', 'docx', 'file'}
            items.append(
                {
                    'id': resource_id,
                    'name': str(item.get('name') or resource_id),
                    'type': resource_type,
                    'selectable': selectable,
                    'has_children': resource_type == 'folder',
                    'metadata': {
                        'url': item.get('url'),
                        'modified_time': item.get('modified_time'),
                    },
                }
            )
        return {
            'items': items,
            'next_page': str(data.get('next_page_token') or '') if data.get('has_more') else '',
            'parent_id': parent_id,
            'root_selectable': not parent_id,
        }


def normalize_meego_host(value: str) -> str:
    raw = value.strip()
    if '://' not in raw:
        raw = f'https://{raw}'
    parsed = urlparse(raw)
    if (
        parsed.scheme != 'https'
        or not parsed.hostname
        or parsed.username
        or parsed.password
        or parsed.path not in {'', '/'}
        or parsed.query
        or parsed.fragment
    ):
        raise OAuthProviderError('MeeGo Host 必须是安全的 HTTPS 域名')
    return parsed.netloc


def _validate_https_endpoint(value: Any, field_name: str) -> str:
    endpoint = str(value or '')
    parsed = urlparse(endpoint)
    if parsed.scheme != 'https' or not parsed.hostname:
        raise OAuthProviderError(f'MeeGo OAuth Discovery 缺少安全的 {field_name}')
    return endpoint


def _mcp_data(payload: dict[str, Any]) -> Any:
    error = payload.get('error')
    if error:
        message = error.get('message') if isinstance(error, dict) else None
        raise OAuthProviderError(f'MeeGo 请求失败：{message or "未知错误"}')
    result = payload.get('result') or {}
    if not isinstance(result, dict):
        return None
    if result.get('isError'):
        raise OAuthProviderError('MeeGo 请求失败，请检查账号权限')
    for entry in result.get('content') or []:
        if not isinstance(entry, dict) or entry.get('type') != 'text':
            continue
        text = str(entry.get('text') or '')
        if not text or text.casefold().startswith(('log_id:', 'logid:')):
            continue
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return text
    return None


def find_meego_projects(value: Any) -> list[dict[str, Any]]:
    projects: list[dict[str, Any]] = []
    seen: set[str] = set()

    def visit(node: Any) -> None:
        if isinstance(node, list):
            for child in node:
                visit(child)
            return
        if not isinstance(node, dict):
            return
        project_key = node.get('project_key') or node.get('projectKey')
        if project_key and str(project_key) not in seen:
            key = str(project_key)
            seen.add(key)
            projects.append(
                {
                    'id': key,
                    'name': str(
                        node.get('name')
                        or node.get('project_name')
                        or node.get('simple_name')
                        or key
                    ),
                    'type': 'project',
                    'selectable': True,
                    'has_children': False,
                    'metadata': {
                        key_name: node.get(key_name)
                        for key_name in ('simple_name', 'description')
                        if node.get(key_name) is not None
                    },
                }
            )
        for child in node.values():
            if isinstance(child, dict | list):
                visit(child)

    visit(value)
    return projects


def find_meego_work_item_types(value: Any) -> list[dict[str, Any]]:
    types: list[dict[str, Any]] = []
    seen: set[str] = set()

    def visit(node: Any) -> None:
        if isinstance(node, list):
            for child in node:
                visit(child)
            return
        if not isinstance(node, dict):
            return
        type_key = node.get('type_key') or node.get('work_item_type_key')
        if type_key and str(type_key) not in seen and not node.get('is_disabled'):
            key = str(type_key)
            seen.add(key)
            types.append(
                {
                    'id': key,
                    'name': str(node.get('name') or node.get('type_name') or key),
                    'type': 'work_item_type',
                    'selectable': True,
                    'has_children': False,
                    'metadata': {'work_item_type_key': key},
                }
            )
        for child in node.values():
            if isinstance(child, dict | list):
                visit(child)

    visit(value)
    return types


class MeegoOAuthProvider:
    name = 'meego'
    _HOST_PATTERN = re.compile(r'^[A-Za-z0-9.-]+(?::[0-9]{1,5})?$')

    def __init__(self, settings: Settings):
        self.settings = settings
        self.host = normalize_meego_host(settings.meego_host)

    @property
    def configured(self) -> bool:
        return bool(self.host)

    async def _discovery(self) -> dict[str, str]:
        if not self._HOST_PATTERN.fullmatch(self.host):
            raise OAuthProviderError('MeeGo Host 格式无效')
        url = f'https://{self.host}/.well-known/oauth-authorization-server'
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.get(url)
        response.raise_for_status()
        payload = _json_object(response, 'MeeGo OAuth Discovery')
        return {
            'authorization_endpoint': _validate_https_endpoint(
                payload.get('authorization_endpoint'), 'authorization_endpoint'
            ),
            'token_endpoint': _validate_https_endpoint(
                payload.get('token_endpoint'), 'token_endpoint'
            ),
            'registration_endpoint': _validate_https_endpoint(
                payload.get('registration_endpoint'), 'registration_endpoint'
            ),
        }

    async def start(self, redirect_uri: str, state: str) -> OAuthStart:
        discovery = await self._discovery()
        registration = {
            'client_name': 'Graphiti Studio',
            'grant_types': ['authorization_code', 'refresh_token'],
            'token_endpoint_auth_method': 'none',
            'redirect_uris': [redirect_uri],
            'response_types': ['code'],
        }
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(discovery['registration_endpoint'], json=registration)
        if response.status_code not in {200, 201}:
            raise OAuthProviderError('MeeGo OAuth 客户端注册失败')
        registered = _json_object(response, 'MeeGo OAuth 客户端注册')
        if not registered.get('client_id') and isinstance(registered.get('data'), dict):
            registered = registered['data']
        client_id = str(registered.get('client_id') or '')
        if not client_id:
            raise OAuthProviderError('MeeGo OAuth 客户端注册响应缺少 client_id')
        verifier, challenge = generate_pkce()
        query = urlencode(
            {
                'response_type': 'code',
                'client_id': client_id,
                'redirect_uri': redirect_uri,
                'code_challenge': challenge,
                'code_challenge_method': 'S256',
                'state': state,
                'channel': 'graphiti-studio',
            }
        )
        return OAuthStart(
            authorization_url=f'{discovery["authorization_endpoint"]}?{query}',
            code_verifier=verifier,
            client_id=client_id,
            client_secret=str(registered.get('client_secret') or ''),
            token_url=discovery['token_endpoint'],
            metadata={'host': self.host},
        )

    async def exchange(
        self,
        *,
        code: str,
        redirect_uri: str,
        code_verifier: str,
        client_id: str,
        client_secret: str,
        token_url: str,
    ) -> OAuthTokens:
        del client_secret
        data = {
            'grant_type': 'authorization_code',
            'code': code,
            'redirect_uri': redirect_uri,
            'client_id': client_id,
            'code_verifier': code_verifier,
        }
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(token_url, data=data)
        if response.status_code >= 400:
            raise OAuthProviderError('MeeGo 授权码交换失败，请重新授权')
        payload = _json_object(response, 'MeeGo 授权')
        access_token = str(payload.get('access_token') or '')
        if not access_token:
            raise OAuthProviderError('MeeGo 授权响应缺少 access_token')
        return OAuthTokens(
            access_token=access_token,
            refresh_token=str(payload.get('refresh_token') or ''),
            expires_in=_integer(payload.get('expires_in')),
            scope=str(payload.get('scope') or ''),
        )

    async def refresh(self, refresh_token: str, client_id: str) -> OAuthTokens:
        discovery = await self._discovery()
        data = {
            'grant_type': 'refresh_token',
            'refresh_token': refresh_token,
            'client_id': client_id,
        }
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(discovery['token_endpoint'], data=data)
        if response.status_code >= 400:
            raise OAuthProviderError('MeeGo 授权已过期，请重新连接')
        payload = _json_object(response, '刷新 MeeGo 授权')
        access_token = str(payload.get('access_token') or '')
        if not access_token:
            raise OAuthProviderError('刷新 MeeGo 授权失败，请重新连接')
        return OAuthTokens(
            access_token=access_token,
            refresh_token=str(payload.get('refresh_token') or refresh_token),
            expires_in=_integer(payload.get('expires_in')),
            scope=str(payload.get('scope') or ''),
        )

    async def _mcp_request(
        self, access_token: str, method: str, params: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        url = f'https://{self.host}/mcp_server/v1'
        request = {
            'jsonrpc': '2.0',
            'id': secrets.randbelow(2**31 - 1) + 1,
            'method': method,
        }
        if params is not None:
            request['params'] = params
        headers = {'Authorization': f'Bearer {access_token}'}
        async with httpx.AsyncClient(timeout=60) as client:
            response = await client.post(url, headers=headers, json=request)
        if response.status_code >= 400:
            raise OAuthProviderError('MeeGo 请求失败，请重新授权或检查账号权限')
        payload = _json_object(response, 'MeeGo MCP')
        error = payload.get('error')
        if error:
            raise OAuthProviderError('MeeGo 请求失败，请检查账号权限')
        return payload

    async def _resolve_tool(
        self,
        access_token: str,
        *,
        resource: str,
        method: str,
        fallback: str,
    ) -> str:
        payload = await self._mcp_request(access_token, 'tools/list')
        result = payload.get('result')
        tools = result.get('tools') if isinstance(result, dict) else None
        if not isinstance(tools, list):
            raise OAuthProviderError('MeeGo 未返回可用工具列表')

        fallback_available = False
        for tool in tools:
            if not isinstance(tool, dict):
                continue
            name = str(tool.get('name') or '')
            if name == fallback:
                fallback_available = True
            metadata = tool.get('metadata') or tool.get('_meta') or {}
            if not isinstance(metadata, dict):
                continue
            if metadata.get('resource') == resource and metadata.get('method') == method and name:
                return name
            nested = metadata.get('metadata')
            if (
                isinstance(nested, dict)
                and nested.get('resource') == resource
                and nested.get('method') == method
                and name
            ):
                return name
        if fallback_available:
            return fallback
        raise OAuthProviderError(f'MeeGo 当前账号未提供所需的 {resource} 能力')

    async def business_call(
        self,
        access_token: str,
        *,
        resource: str,
        method: str,
        fallback: str,
        arguments: dict[str, Any],
    ) -> Any:
        tool = await self._resolve_tool(
            access_token,
            resource=resource,
            method=method,
            fallback=fallback,
        )
        payload = await self._mcp_request(
            access_token,
            'tools/call',
            {'name': tool, 'arguments': arguments},
        )
        return _mcp_data(payload)

    async def identity(self, access_token: str) -> OAuthIdentity:
        projects = find_meego_projects(
            await self.business_call(
                access_token,
                resource='project',
                method='search',
                fallback='search_project_info',
                arguments={'page_num': 1},
            )
        )
        suffix = f' · {projects[0]["name"]}' if projects else ''
        token_fingerprint = hashlib.sha256(access_token.encode()).hexdigest()[:20]
        return OAuthIdentity(
            account_id=f'{self.host}:{token_fingerprint}',
            account_name=f'MeeGo 账号{suffix}',
            tenant_id=self.host,
            metadata={'host': self.host},
        )

    async def resources(
        self, access_token: str, *, parent_id: str = '', page_token: str = ''
    ) -> dict[str, Any]:
        if parent_id:
            value = await self.business_call(
                access_token,
                resource='workitem',
                method='meta-types',
                fallback='list_workitem_types',
                arguments={'project_key': parent_id},
            )
            return {
                'items': find_meego_work_item_types(value),
                'next_page': '',
                'parent_id': parent_id,
                'root_selectable': False,
            }
        page = max(1, _integer(page_token, 1))
        value = await self.business_call(
            access_token,
            resource='project',
            method='search',
            fallback='search_project_info',
            arguments={'page_num': page},
        )
        projects = find_meego_projects(value)
        return {
            'items': projects,
            'next_page': str(page + 1) if len(projects) >= 50 else '',
            'parent_id': '',
            'root_selectable': False,
        }
