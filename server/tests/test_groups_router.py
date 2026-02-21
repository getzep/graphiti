import pytest

from graph_service.dto import ResolveGroupIdRequest
from graph_service.routers.groups import resolve_group


@pytest.mark.asyncio
async def test_resolve_group_returns_group_id():
    response = await resolve_group(ResolveGroupIdRequest(scope='user', key='github_login:octocat'))
    assert response.group_id.startswith('graphiti_user_')
