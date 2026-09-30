#!/usr/bin/env python3
"""Unit tests for the unknown-argument refusal on MCP tool calls.

MCPServer validates tool arguments against a pydantic model that ignores extra
keys, so a misspelled argument (`group_id` where the tool takes `group_ids`) was
dropped silently and the call ran with the default scope. These tests drive the
real `mcp` instance through `call_tool`, the same path the wire uses, and assert
the call is refused with an error that names the argument and the closest valid
one. The tool bodies never run (the refusal comes first), so no database or
mocked client is needed.
"""

import sys
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from mcp.server.mcpserver.exceptions import ToolError

# Add the src directory to the path (mirrors the other unit tests)
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import graphiti_mcp_server as server  # noqa: E402
from utils.tool_args import closest_argument, unknown_argument_message  # noqa: E402


@pytest.fixture
def no_client(monkeypatch):
    """A tool body that ran would need a client; make reaching it observable."""
    service = AsyncMock()
    monkeypatch.setattr(server, 'graphiti_service', service)
    return service


async def test_search_nodes_refuses_group_id_and_suggests_group_ids(no_client):
    with pytest.raises(ToolError) as exc:
        await server.mcp.call_tool('search_nodes', {'query': 'x', 'group_id': 'proj'})

    message = str(exc.value)
    assert "'group_id'" in message
    assert "'group_ids'" in message
    no_client.get_client.assert_not_called()


async def test_search_memory_facts_refuses_group_id_and_suggests_group_ids(no_client):
    with pytest.raises(ToolError) as exc:
        await server.mcp.call_tool('search_memory_facts', {'query': 'x', 'group_id': 'proj'})

    message = str(exc.value)
    assert "'group_id'" in message
    assert "'group_ids'" in message
    no_client.get_client.assert_not_called()


async def test_add_memory_refuses_group_ids_and_suggests_group_id(no_client):
    with pytest.raises(ToolError) as exc:
        await server.mcp.call_tool(
            'add_memory', {'name': 'n', 'episode_body': 'b', 'group_ids': 'proj'}
        )

    message = str(exc.value)
    assert "'group_ids'" in message
    assert "'group_id'" in message
    no_client.get_client.assert_not_called()


async def test_unknown_argument_without_a_close_match_names_only_the_argument(no_client):
    with pytest.raises(ToolError) as exc:
        await server.mcp.call_tool('search_nodes', {'query': 'x', 'zzzzzzzz': 1})

    message = str(exc.value)
    assert "'zzzzzzzz'" in message
    assert 'Did you mean' not in message
    assert 'group_ids' in message  # valid names listed so the caller can self-correct


async def test_every_unknown_argument_is_reported(no_client):
    with pytest.raises(ToolError) as exc:
        await server.mcp.call_tool(
            'search_nodes', {'query': 'x', 'group_id': 'proj', 'max_node': 5}
        )

    message = str(exc.value)
    assert "'group_id'" in message and "'group_ids'" in message
    assert "'max_node'" in message and "'max_nodes'" in message


async def test_known_arguments_still_reach_the_tool(monkeypatch):
    """The refusal must not break a well-formed call."""
    seen: dict = {}

    async def fake_get_client():
        seen['reached'] = True
        raise RuntimeError('stop here: the tool body was reached')

    service = AsyncMock()
    service.get_client = fake_get_client
    monkeypatch.setattr(server, 'graphiti_service', service)

    result = await server.mcp.call_tool('search_nodes', {'query': 'x', 'group_ids': 'proj'})

    assert seen.get('reached') is True
    assert 'unknown' not in str(result).lower()


async def test_every_registered_tool_refuses_an_unknown_argument(no_client):
    tools = await server.mcp.list_tools()
    assert tools
    for tool in tools:
        with pytest.raises(ToolError) as exc:
            await server.mcp.call_tool(tool.name, {'definitely_not_an_arg': 1})
        assert "'definitely_not_an_arg'" in str(exc.value), tool.name


def test_closest_argument_picks_the_nearest_valid_name():
    assert closest_argument('group_id', ['query', 'group_ids', 'max_nodes']) == 'group_ids'
    assert closest_argument('group_ids', ['name', 'group_id', 'source']) == 'group_id'
    assert closest_argument('zzzzzzzz', ['query', 'group_ids']) is None


def test_unknown_argument_message_shape():
    message = unknown_argument_message('search_nodes', ['group_id'], ['query', 'group_ids'])
    assert message.startswith("Unknown argument 'group_id' for search_nodes")
    assert "Did you mean 'group_ids'?" in message
