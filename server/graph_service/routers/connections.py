from __future__ import annotations

import html
import json
import secrets
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from fastapi.responses import HTMLResponse, RedirectResponse, Response

from graph_service.config import Settings, get_settings
from graph_service.connections.manager import (
    ConnectionManager,
    ConnectionManagerError,
    normalize_public_base_url,
)
from graph_service.sources.store import SourceStore

router = APIRouter(prefix='/api', tags=['connections'])
SESSION_COOKIE = 'graphiti_oauth_session'


def get_connection_manager(request: Request) -> ConnectionManager:
    return request.app.state.connection_manager


def get_source_store(request: Request) -> SourceStore:
    return request.app.state.source_store


ManagerDep = Annotated[ConnectionManager, Depends(get_connection_manager)]
StoreDep = Annotated[SourceStore, Depends(get_source_store)]
SettingsDep = Annotated[Settings, Depends(get_settings)]


def _public_base_url(request: Request, settings: Settings) -> str:
    try:
        return normalize_public_base_url(
            settings.oauth_public_base_url,
            str(request.base_url),
        )
    except ConnectionManagerError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


def _safe_callback_html(
    *,
    origin: str,
    provider: str,
    connection_id: str | None = None,
    error: str | None = None,
) -> HTMLResponse:
    nonce = secrets.token_urlsafe(18)
    payload = {
        'type': 'graphiti.oauth.complete' if connection_id else 'graphiti.oauth.error',
        'provider': provider,
    }
    if connection_id:
        payload['connection_id'] = connection_id
    else:
        payload['message'] = error or '授权失败，请关闭窗口后重试'
    payload_json = (
        json.dumps(payload, ensure_ascii=False)
        .replace('&', '\\u0026')
        .replace('<', '\\u003c')
        .replace('>', '\\u003e')
    )
    origin_json = (
        json.dumps(origin).replace('&', '\\u0026').replace('<', '\\u003c').replace('>', '\\u003e')
    )
    title = '账号已连接' if connection_id else '授权未完成'
    detail = html.escape(
        str('正在返回 Graphiti Studio…' if connection_id else payload['message']), quote=True
    )
    body = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>{title}</title><style nonce="{nonce}">
body{{font:16px system-ui;background:#0b0f10;color:#f5f3ed;display:grid;place-items:center;min-height:100vh;margin:0}}
main{{max-width:30rem;padding:2rem;border:1px solid #354047;border-radius:1rem;background:#141a1d}}p{{color:#a8b2b9}}
</style></head><body><main><h1>{title}</h1><p>{detail}</p></main>
<script nonce="{nonce}">const payload={payload_json};const origin={origin_json};
if(window.opener)window.opener.postMessage(payload,origin);setTimeout(()=>window.close(),500);</script>
</body></html>"""
    return HTMLResponse(
        body,
        headers={
            'Cache-Control': 'no-store, max-age=0',
            'Pragma': 'no-cache',
            'Referrer-Policy': 'no-referrer',
            'Content-Security-Policy': (
                f"default-src 'none'; style-src 'nonce-{nonce}'; script-src 'nonce-{nonce}'; "
                "base-uri 'none'; frame-ancestors 'none'; form-action 'none'"
            ),
            'X-Content-Type-Options': 'nosniff',
        },
    )


@router.get('/oauth/providers')
async def oauth_providers(manager: ManagerDep):
    return manager.provider_status()


@router.get('/oauth/{provider}/start')
async def oauth_start(
    provider: str,
    request: Request,
    manager: ManagerDep,
    settings: SettingsDep,
):
    public_base_url = _public_base_url(request, settings)
    browser_session = request.cookies.get(SESSION_COOKIE)
    if not browser_session or len(browser_session) < 32:
        browser_session = secrets.token_urlsafe(32)
    try:
        authorization_url = await manager.start(
            provider,
            public_base_url=public_base_url,
            browser_session=browser_session,
        )
    except ConnectionManagerError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    response = RedirectResponse(authorization_url, status_code=status.HTTP_302_FOUND)
    response.set_cookie(
        SESSION_COOKIE,
        browser_session,
        max_age=24 * 60 * 60,
        httponly=True,
        secure=settings.oauth_cookie_secure or public_base_url.startswith('https://'),
        samesite='lax',
        path='/api/oauth',
    )
    response.headers['Cache-Control'] = 'no-store'
    response.headers['Referrer-Policy'] = 'no-referrer'
    return response


@router.get('/oauth/{provider}/callback', response_class=HTMLResponse)
async def oauth_callback(
    provider: str,
    request: Request,
    manager: ManagerDep,
    settings: SettingsDep,
    state_value: str = Query(alias='state', min_length=20, max_length=256),
    code: str | None = Query(default=None, max_length=4096),
    error: str | None = Query(default=None, max_length=256),
):
    public_base_url = _public_base_url(request, settings)
    browser_session = request.cookies.get(SESSION_COOKIE)
    if not browser_session:
        return _safe_callback_html(
            origin=public_base_url,
            provider=provider,
            error='授权会话已失效，请重新连接',
        )
    try:
        connection = await manager.complete(
            provider,
            state=state_value,
            browser_session=browser_session,
            code=code,
            oauth_error=error,
        )
    except ConnectionManagerError as exc:
        return _safe_callback_html(
            origin=public_base_url,
            provider=provider,
            error=str(exc),
        )
    return _safe_callback_html(
        origin=public_base_url,
        provider=provider,
        connection_id=str(connection['id']),
    )


@router.get('/connections')
async def list_connections(manager: ManagerDep):
    return manager.store.list_connections()


@router.get('/connections/{connection_id}/resources')
async def list_connection_resources(
    connection_id: str,
    manager: ManagerDep,
    parent_id: str = Query(default='', max_length=512),
    page: str = Query(default='', max_length=512),
):
    try:
        return await manager.resources(connection_id, parent_id=parent_id, page=page)
    except (ConnectionManagerError, KeyError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get('/connections/{connection_id}/meego/views')
async def list_meego_views(
    connection_id: str,
    manager: ManagerDep,
    project_key: str = Query(min_length=1, max_length=128),
    query: str = Query(default='', max_length=128),
):
    try:
        return {
            'items': await manager.meego_views(
                connection_id,
                project_key=project_key,
                query=query,
            )
        }
    except (ConnectionManagerError, KeyError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.delete('/connections/{connection_id}', status_code=status.HTTP_204_NO_CONTENT)
async def delete_connection(
    connection_id: str,
    manager: ManagerDep,
    store: StoreDep,
):
    if store.sources_using_connection(connection_id):
        raise HTTPException(status_code=409, detail='该账号仍被数据源使用，请先删除或迁移数据源')
    try:
        manager.store.delete_connection(connection_id)
    except (KeyError, ValueError) as exc:
        raise HTTPException(status_code=404, detail='连接账号不存在') from exc
    return Response(status_code=status.HTTP_204_NO_CONTENT)
