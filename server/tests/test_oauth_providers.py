from unittest.mock import AsyncMock

import pytest

from graph_service.config import Settings
from graph_service.connections.providers import (
    FeishuOAuthProvider,
    MeegoOAuthProvider,
    OAuthProviderError,
    find_meego_user,
    find_meego_views,
)


def _provider() -> MeegoOAuthProvider:
    return MeegoOAuthProvider(Settings(_env_file=None, falkordb_port=6379))


@pytest.mark.asyncio
async def test_feishu_uses_current_oauth_token_endpoint():
    provider = FeishuOAuthProvider(
        Settings(
            _env_file=None,
            falkordb_port=6379,
            feishu_app_id='client-id',
            feishu_app_secret='client-secret',
        )
    )

    start = await provider.start('http://localhost:8000/api/oauth/feishu/callback', 'state')

    assert start.token_url == 'https://open.feishu.cn/open-apis/authen/v2/oauth/token'


def test_meego_current_user_has_stable_identity():
    user = find_meego_user(
        [
            {
                'user_key': 'user-123',
                'name_cn': '测试用户',
                'email': 'user@example.test',
            }
        ]
    )

    assert user == {
        'id': 'user-123',
        'name': '测试用户',
        'email': 'user@example.test',
        'avatar_url': '',
    }


def test_meego_views_are_normalized_and_deduplicated():
    views = find_meego_views(
        {
            'data': {
                'view_list': [
                    {'view_id': 'view-1', 'view_name': '需求池'},
                    {'view_id': 'view-1', 'view_name': '重复结果'},
                    {'viewId': 'view-2', 'name': '本周计划'},
                ]
            }
        }
    )

    assert views == [
        {'id': 'view-1', 'name': '需求池'},
        {'id': 'view-2', 'name': '本周计划'},
    ]


@pytest.mark.asyncio
async def test_meego_searches_default_visible_story_views():
    provider = _provider()
    provider.business_call = AsyncMock(  # type: ignore[method-assign]
        return_value={'view_list': [{'view_id': 'view-1', 'view_name': '需求池'}]}
    )

    assert await provider.search_views('access-token', project_key='demo') == [
        {'id': 'view-1', 'name': '需求池'}
    ]
    provider.business_call.assert_awaited_once_with(
        'access-token',
        resource='view',
        method='search',
        fallback='search_view_by_title',
        arguments={'key_word': ' ', 'project_key': 'demo', 'view_scope': 'story'},
    )


@pytest.mark.asyncio
async def test_meego_discovers_metadata_named_tool_before_calling_it():
    provider = _provider()
    provider._mcp_request = AsyncMock(  # type: ignore[method-assign]
        side_effect=[
            {
                'result': {
                    'tools': [
                        {
                            'name': 'tenant_specific_project_search',
                            '_meta': {'resource': 'project', 'method': 'search'},
                        }
                    ]
                }
            },
            {'result': {'content': [{'type': 'text', 'text': '{"list": []}'}]}},
        ]
    )

    result = await provider.business_call(
        'access-token',
        resource='project',
        method='search',
        fallback='search_project_info',
        arguments={'page_num': 1},
    )

    assert result == {'list': []}
    assert provider._mcp_request.await_args_list[0].args[1] == 'tools/list'
    assert provider._mcp_request.await_args_list[1].args[2]['name'] == (
        'tenant_specific_project_search'
    )


@pytest.mark.asyncio
async def test_meego_uses_official_fallback_only_if_advertised():
    provider = _provider()
    provider._mcp_request = AsyncMock(  # type: ignore[method-assign]
        side_effect=[
            {'result': {'tools': [{'name': 'search_project_info'}]}},
            {'result': {'content': [{'type': 'text', 'text': '[]'}]}},
        ]
    )
    assert (
        await provider.business_call(
            'access-token',
            resource='project',
            method='search',
            fallback='search_project_info',
            arguments={},
        )
        == []
    )

    provider._mcp_request = AsyncMock(  # type: ignore[method-assign]
        return_value={'result': {'tools': [{'name': 'unrelated'}]}}
    )
    with pytest.raises(OAuthProviderError, match='未提供'):
        await provider.business_call(
            'access-token',
            resource='project',
            method='search',
            fallback='search_project_info',
            arguments={},
        )
