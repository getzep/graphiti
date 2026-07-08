"""Live-graph relevance checks for the tuned MCP search tools (opt-in).

Proves the behavioral claims that unit tests cannot: against a real, populated
graph, (a) no superseded/expired facts are returned by default, and (b) every
returned fact/node carries a rerank ``score``.

Opt-in: set ``GRAPHITI_INT_GROUP_ID`` to the group to query. The server is
spawned as a subprocess over stdio (same bootstrap as
``test_live_falkordb_int.py``), so it also needs a reachable FalkorDB and a
working embedder/LLM config (matching whatever populated the target graph).

Example::

    cd mcp_server
    GRAPHITI_INT_GROUP_ID=brent_atvenu uv run pytest tests/test_search_relevance_int.py -v
"""

import json
import os
import socket
from contextlib import AsyncExitStack, suppress
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest
from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

_REPO_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(_REPO_ROOT / '.env')

MCP_SERVER_DIR = Path(__file__).resolve().parent.parent
FALKORDB_URI = os.environ.get('FALKORDB_URI', 'redis://localhost:6379')

pytestmark = [
    pytest.mark.integration,
    pytest.mark.requires_falkordb,
]


def _falkordb_reachable(uri: str) -> bool:
    host, port = 'localhost', 6379
    if '://' in uri:
        rest = uri.split('://', 1)[1].split('@')[-1]
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


_GROUP = os.environ.get('GRAPHITI_INT_GROUP_ID')
if not _GROUP:
    pytest.skip(
        'set GRAPHITI_INT_GROUP_ID to run live-graph relevance checks',
        allow_module_level=True,
    )
if not _falkordb_reachable(FALKORDB_URI):
    pytest.skip(
        f'FalkorDB not reachable at {FALKORDB_URI}; skipping live relevance checks',
        allow_module_level=True,
    )


class _Client:
    """Minimal stdio MCP client for the running server."""

    def __init__(self) -> None:
        self._stack = AsyncExitStack()
        self.session: ClientSession | None = None

    async def __aenter__(self) -> '_Client':
        env = {**os.environ, 'FALKORDB_URI': FALKORDB_URI}
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


async def test_no_expired_edges_returned_int():
    """Every returned fact must be live (invalid_at null or in the future) and scored."""
    async with _Client() as client:
        resp = await client.call(
            'search_memory_facts',
            {'query': 'Rails 6.1 upgrade', 'group_ids': [_GROUP], 'max_facts': 10},
        )
        assert isinstance(resp, dict) and not resp.get('error'), f'unexpected response: {resp}'
        now = datetime.now(timezone.utc)
        facts = resp.get('facts', [])
        for fact in facts:
            assert 'score' in fact, f'fact missing score: {fact.get("uuid")}'
            inv = fact.get('invalid_at')
            if inv:
                assert datetime.fromisoformat(inv) > now, f'stale fact returned: {fact["uuid"]}'


async def test_nodes_carry_score_int():
    """Node search results carry a rerank score."""
    async with _Client() as client:
        resp = await client.call(
            'search_nodes',
            {'query': 'Rails upgrade', 'group_ids': [_GROUP], 'max_nodes': 10},
        )
        assert isinstance(resp, dict) and not resp.get('error'), f'unexpected response: {resp}'
        for node in resp.get('nodes', []):
            assert 'score' in node, f'node missing score: {node.get("uuid")}'
