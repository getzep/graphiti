"""Live end-to-end tests for the Graphiti MCP server against FalkorDB + a real LLM.

These tests start the MCP server as a subprocess over stdio, backed by FalkorDB
(the default database) and a real OpenAI model, and exercise the tools end to end:
add_memory -> wait for async processing -> search_nodes / search_memory_facts ->
get_episodes / get_status -> delete_episode -> clear_graph.

They are skipped automatically unless an OpenAI API key is available AND FalkorDB
is reachable:

- Local: the project-root ``.env`` is loaded, so an ``OPENAI_API_KEY`` there is
  picked up. Start FalkorDB first, e.g.::

      docker run -d --name falkordb -p 6379:6379 falkordb/falkordb:latest
      cd mcp_server && uv run pytest tests/test_live_falkordb_int.py

- CI: ``OPENAI_API_KEY`` comes from the GitHub environment and FalkorDB runs as a
  container (see .github/workflows/mcp-server-tests.yml).

The model is taken from ``MODEL_NAME`` (CI pins a lighter, broadly-available model
for speed/reliability; gpt-5.5 is the server's runtime default but needs a
key with fast gpt-5.5 access), falling back to the server's configured default.
"""

import asyncio
import json
import os
import socket
import time
from contextlib import AsyncExitStack, suppress
from pathlib import Path
from typing import Any

import pytest
from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# Load the project-root .env for local runs (CI provides the env directly).
_REPO_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(_REPO_ROOT / '.env')

MCP_SERVER_DIR = Path(__file__).resolve().parent.parent
FALKORDB_URI = os.environ.get('FALKORDB_URI', 'redis://localhost:6379')

pytestmark = [
    pytest.mark.integration,
    pytest.mark.requires_falkordb,
    pytest.mark.requires_openai,
]


def _falkordb_reachable(uri: str) -> bool:
    """Best-effort TCP check that a FalkorDB (Redis) endpoint is reachable."""
    host, port = 'localhost', 6379
    if '://' in uri:
        rest = uri.split('://', 1)[1].split('@')[-1]  # strip scheme + any credentials
        hostport = rest.split('/')[0]
        if ':' in hostport:
            host_part, port_part = hostport.rsplit(':', 1)
            host = host_part or host
            with suppress(ValueError):
                port = int(port_part)
        elif hostport:
            host = hostport
    try:
        with socket.create_connection((host, port), timeout=2):
            return True
    except OSError:
        return False


# Skip the whole module unless prerequisites are present, so the suite is a no-op
# locally without setup and on fork PRs (which have no secrets).
if not os.environ.get('OPENAI_API_KEY'):
    pytest.skip('OPENAI_API_KEY not set; skipping live MCP tests', allow_module_level=True)
if not _falkordb_reachable(FALKORDB_URI):
    pytest.skip(
        f'FalkorDB not reachable at {FALKORDB_URI}; skipping live MCP tests',
        allow_module_level=True,
    )


def _unique_group_id() -> str:
    # Alphanumeric only (no '_'/'-'): keeps the RediSearch fulltext group filter
    # valid without relying on group_id escaping, so the suite is backend-portable.
    return f'livetest{int(time.time())}{os.getpid()}'


def _raise_on_error(tool: str, resp: Any) -> None:
    """Surface a tool's error response immediately instead of masking it as empty.

    Otherwise a real server/LLM/schema error only shows up as a generic
    'not processed within the timeout' after the full poll, hiding the cause.
    """
    if isinstance(resp, dict) and resp.get('error'):
        raise AssertionError(f'{tool} returned an error: {resp["error"]}')


class LiveMCPClient:
    """Spawns the MCP server over stdio (FalkorDB backend) and calls its tools."""

    def __init__(self, group_id: str):
        self.group_id = group_id
        self._stack = AsyncExitStack()
        self.session: ClientSession | None = None

    async def __aenter__(self) -> 'LiveMCPClient':
        env = {
            **os.environ,
            'GRAPHITI_GROUP_ID': self.group_id,
            'FALKORDB_URI': FALKORDB_URI,
        }
        params = StdioServerParameters(
            command='uv',
            args=['run', str(MCP_SERVER_DIR / 'main.py'), '--transport', 'stdio'],
            env=env,
            cwd=str(MCP_SERVER_DIR),
        )
        read, write = await self._stack.enter_async_context(stdio_client(params))
        self.session = await self._stack.enter_async_context(ClientSession(read, write))
        await self.session.initialize()
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self._stack.aclose()

    async def call(self, tool: str, arguments: dict[str, Any]) -> Any:
        assert self.session is not None
        result = await self.session.call_tool(tool, arguments)
        if not result.content:
            return None
        text = getattr(result.content[0], 'text', None)
        if text is None:
            return None
        try:
            return json.loads(text)
        except (json.JSONDecodeError, TypeError):
            return text

    async def list_tool_names(self) -> set[str]:
        assert self.session is not None
        result = await self.session.list_tools()
        return {tool.name for tool in result.tools}

    async def wait_for_episodes(
        self, expected: int = 1, timeout: float = 180.0, poll: float = 3.0
    ) -> list[dict[str, Any]]:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            resp = await self.call(
                'get_episodes', {'group_ids': [self.group_id], 'max_episodes': 50}
            )
            _raise_on_error('get_episodes', resp)
            episodes = resp.get('episodes', []) if isinstance(resp, dict) else []
            if len(episodes) >= expected:
                return episodes
            await asyncio.sleep(poll)
        return []

    async def search_until(
        self,
        tool: str,
        arguments: dict[str, Any],
        key: str,
        attempts: int = 6,
        poll: float = 3.0,
    ) -> list[Any]:
        """Call a search tool, retrying until ``resp[key]`` is non-empty (index lag)."""
        results: list[Any] = []
        for _ in range(attempts):
            resp = await self.call(tool, arguments)
            _raise_on_error(tool, resp)
            results = resp.get(key, []) if isinstance(resp, dict) else []
            if results:
                return results
            await asyncio.sleep(poll)
        return results


async def test_server_lists_core_tools():
    async with LiveMCPClient(_unique_group_id()) as client:
        tools = await client.list_tool_names()
        for expected in (
            'add_memory',
            'search_nodes',
            'search_memory_facts',
            'get_episodes',
            'get_entity_edge',
            'delete_entity_edge',
            'delete_episode',
            'clear_graph',
            'get_status',
            # parity tools (this branch)
            'summarize_saga',
            'build_communities',
            'add_triplet',
            'get_episode_entities',
        ):
            assert expected in tools, f'missing tool {expected}; got {sorted(tools)}'


async def test_end_to_end_add_search_delete_clear():
    group = _unique_group_id()
    async with LiveMCPClient(group) as client:
        try:
            # 1. Add an episode with clearly-extractable entities and a relationship.
            add = await client.call(
                'add_memory',
                {
                    'name': 'Live Test Episode',
                    'episode_body': (
                        'Alice is a software engineer at Acme Corporation. '
                        'She works on the Graphiti project.'
                    ),
                    'source': 'text',
                    'source_description': 'live integration test',
                    'group_id': group,
                },
            )
            assert isinstance(add, dict) and 'message' in add, (
                f'unexpected add_memory response: {add}'
            )

            # 2. Wait for async processing to persist the episode (and its graph).
            episodes = await client.wait_for_episodes(expected=1)
            assert episodes, 'episode was not processed within the timeout'
            assert any(e.get('group_id') == group for e in episodes)
            episode_uuid = episodes[0]['uuid']

            # 3. At least one entity was extracted and is searchable.
            nodes = await client.search_until(
                'search_nodes',
                {'query': 'Alice Acme Graphiti', 'group_ids': [group], 'max_nodes': 10},
                key='nodes',
            )
            assert nodes, 'expected at least one extracted entity node'

            # 4. At least one fact (edge) was extracted and is searchable.
            facts = await client.search_until(
                'search_memory_facts',
                {'query': 'Alice Acme Corporation', 'group_ids': [group], 'max_facts': 10},
                key='facts',
            )
            assert facts, 'expected at least one extracted fact'

            # 5. Status reports a healthy FalkorDB connection.
            status = await client.call('get_status', {})
            assert isinstance(status, dict) and status.get('status') == 'ok', f'status: {status}'

            # 6. Delete the episode, then clear the test group.
            deleted = await client.call('delete_episode', {'uuid': episode_uuid})
            assert isinstance(deleted, dict) and 'message' in deleted, f'delete_episode: {deleted}'

            cleared = await client.call('clear_graph', {'group_ids': [group]})
            assert isinstance(cleared, dict) and 'message' in cleared, f'clear_graph: {cleared}'
        finally:
            # Always remove this run's data, even if an assertion above failed, so a
            # long-lived local FalkorDB doesn't accumulate orphaned test groups.
            with suppress(Exception):
                await client.call('clear_graph', {'group_ids': [group]})
