import sqlite3

import pytest
from mcp.server.fastmcp import FastMCP
from starlette.testclient import TestClient

from graphiti_mcp_server import create_http_application
from vaka_wiki_mcp import PublishedWikiResolver, create_vaka_wiki_mcp


def _database(path, *, wiki_id='a' * 32, published_group_id='published_group'):
    connection = sqlite3.connect(path)
    connection.execute('CREATE TABLE wikis (id TEXT PRIMARY KEY, published_group_id TEXT)')
    connection.execute(
        'INSERT INTO wikis (id, published_group_id) VALUES (?, ?)',
        (wiki_id, published_group_id),
    )
    connection.commit()
    connection.close()


def test_resolver_returns_only_the_published_namespace(tmp_path):
    path = tmp_path / 'state.db'
    _database(path)

    assert PublishedWikiResolver(path).resolve('a' * 32) == 'published_group'


def test_resolver_rejects_unknown_or_unpublished_wiki(tmp_path):
    path = tmp_path / 'state.db'
    _database(path, published_group_id=None)
    resolver = PublishedWikiResolver(path)

    with pytest.raises(ValueError, match='尚未发布'):
        resolver.resolve('a' * 32)
    with pytest.raises(ValueError, match='不存在'):
        resolver.resolve('b' * 32)


@pytest.mark.asyncio
async def test_wiki_mcp_exposes_only_three_read_tools(tmp_path):
    path = tmp_path / 'state.db'
    _database(path)

    async def get_client():
        raise AssertionError('tool listing must not initialize Graphiti')

    server = create_vaka_wiki_mcp(get_client, PublishedWikiResolver(path))
    tools = await server.list_tools()

    assert {tool.name for tool in tools} == {'item.search', 'item.get', 'link.traverse'}


def test_http_application_routes_wiki_id_to_read_only_mcp(tmp_path):
    path = tmp_path / 'state.db'
    _database(path)

    async def get_client():
        raise AssertionError('routing must not initialize Graphiti')

    admin = FastMCP('Admin')
    wiki = create_vaka_wiki_mcp(get_client, PublishedWikiResolver(path))
    wiki.settings.transport_security.enable_dns_rebinding_protection = False
    application = create_http_application(admin, wiki)
    url = f'/wiki/{"a" * 32}/mcp'
    headers = {
        'Accept': 'application/json, text/event-stream',
        'Content-Type': 'application/json',
    }
    initialize = {
        'jsonrpc': '2.0',
        'id': 1,
        'method': 'initialize',
        'params': {
            'protocolVersion': '2025-06-18',
            'capabilities': {},
            'clientInfo': {'name': 'test', 'version': '1'},
        },
    }

    with TestClient(application) as client:
        response = client.post(url, headers=headers, json=initialize)
        assert response.status_code == 200
        listed = client.post(
            url,
            headers=headers,
            json={'jsonrpc': '2.0', 'id': 2, 'method': 'tools/list', 'params': {}},
        )

    assert listed.status_code == 200
    assert all(name in listed.text for name in ('item.search', 'item.get', 'link.traverse'))
