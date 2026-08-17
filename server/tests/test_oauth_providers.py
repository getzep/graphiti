from unittest.mock import AsyncMock

import pytest

from graph_service.config import Settings
from graph_service.connections.providers import MeegoOAuthProvider, OAuthProviderError


def _provider() -> MeegoOAuthProvider:
    return MeegoOAuthProvider(Settings(_env_file=None, falkordb_port=6379))


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
