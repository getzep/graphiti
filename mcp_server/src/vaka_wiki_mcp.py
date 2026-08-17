"""Read-only MCP surface for one published Vaka Wiki."""

from __future__ import annotations

import os
import re
import sqlite3
from pathlib import Path
from typing import TYPE_CHECKING, Any

from graphiti_core.edges import EntityEdge
from graphiti_core.errors import NodeNotFoundError
from graphiti_core.nodes import EntityNode
from graphiti_core.search.search_config_recipes import NODE_HYBRID_SEARCH_RRF
from mcp.server.fastmcp import Context, FastMCP

if TYPE_CHECKING:
    from graphiti_core import Graphiti

WIKI_ID_PATTERN = re.compile(r'^[a-f0-9]{32}$')


class PublishedWikiResolver:
    """Resolve a Wiki to its atomically published Graphiti namespace."""

    def __init__(self, path: Path | str | None = None):
        default_path = Path(__file__).parents[2] / 'server' / 'data' / 'source_state.db'
        configured_path = path or os.getenv('VAKA_STATE_PATH') or os.getenv('SOURCE_STATE_PATH')
        self.path = Path(configured_path) if configured_path else default_path

    def resolve(self, wiki_id: str) -> str:
        if not WIKI_ID_PATTERN.fullmatch(wiki_id):
            raise ValueError('Wiki ID 格式无效')
        if not self.path.exists():
            raise ValueError('Vaka 控制库不存在')
        connection = sqlite3.connect(f'{self.path.resolve().as_uri()}?mode=ro', uri=True, timeout=5)
        try:
            row = connection.execute(
                'SELECT published_group_id FROM wikis WHERE id = ?', (wiki_id,)
            ).fetchone()
        except sqlite3.OperationalError as exc:
            raise ValueError('Vaka 控制库尚未初始化') from exc
        finally:
            connection.close()
        if row is None:
            raise ValueError('Wiki 不存在')
        if not row[0]:
            raise ValueError('Wiki 尚未发布')
        return str(row[0])


def _wiki_id(ctx: Context) -> str:
    request = ctx.request_context.request
    if request is None:
        raise ValueError('Wiki MCP 只支持 HTTP 传输')
    path_params = getattr(request, 'path_params', {})
    wiki_id = path_params.get('wiki_id')
    if not wiki_id:
        raise ValueError('MCP URL 中缺少 Wiki ID')
    return str(wiki_id)


def _node_result(node: EntityNode) -> dict[str, Any]:
    return {
        'id': node.uuid,
        'name': node.name,
        'summary': node.summary,
        'types': node.labels,
        'attributes': node.attributes,
    }


def create_vaka_wiki_mcp(
    get_client,
    resolver: PublishedWikiResolver | None = None,
) -> FastMCP:
    """Create the small, read-only MCP surface used by Wiki consumers."""
    published = resolver or PublishedWikiResolver()
    server = FastMCP(
        'Vaka Wiki',
        instructions=(
            'Read the published knowledge of the Wiki bound to this MCP URL. '
            'This server is read-only and exposes search, item lookup, and link traversal.'
        ),
        stateless_http=True,
    )

    @server.tool(name='item.search')
    async def item_search(query: str, ctx: Context, max_items: int = 10) -> dict[str, Any]:
        """Search entities in the current published Wiki."""
        wiki_id = _wiki_id(ctx)
        group_id = published.resolve(wiki_id)
        client: Graphiti = await get_client()
        results = await client.search_(
            query=query,
            config=NODE_HYBRID_SEARCH_RRF,
            group_ids=[group_id],
        )
        max_items = max(1, min(max_items, 50))
        nodes = [node for node in (results.nodes or []) if node.group_id == group_id][:max_items]
        return {'wiki_id': wiki_id, 'items': [_node_result(node) for node in nodes]}

    @server.tool(name='item.get')
    async def item_get(item_id: str, ctx: Context) -> dict[str, Any]:
        """Get one entity from the current published Wiki."""
        wiki_id = _wiki_id(ctx)
        group_id = published.resolve(wiki_id)
        client: Graphiti = await get_client()
        try:
            node = await EntityNode.get_by_uuid(client.driver, item_id)
        except NodeNotFoundError as exc:
            raise ValueError('Item 不存在') from exc
        if node.group_id != group_id:
            raise ValueError('Item 不存在')
        return {'wiki_id': wiki_id, 'item': _node_result(node)}

    @server.tool(name='link.traverse')
    async def link_traverse(
        item_id: str,
        ctx: Context,
        max_links: int = 20,
    ) -> dict[str, Any]:
        """Return neighboring links and entities for one published Wiki item."""
        wiki_id = _wiki_id(ctx)
        group_id = published.resolve(wiki_id)
        client: Graphiti = await get_client()
        try:
            item = await EntityNode.get_by_uuid(client.driver, item_id)
        except NodeNotFoundError as exc:
            raise ValueError('Item 不存在') from exc
        if item.group_id != group_id:
            raise ValueError('Item 不存在')

        max_links = max(1, min(max_links, 100))
        edges = [
            edge
            for edge in await EntityEdge.get_by_node_uuid(client.driver, item_id)
            if edge.group_id == group_id
        ][:max_links]
        neighbor_ids = {
            edge.target_node_uuid if edge.source_node_uuid == item_id else edge.source_node_uuid
            for edge in edges
        }
        neighbors = await EntityNode.get_by_uuids(
            client.driver, list(neighbor_ids), group_id=group_id
        )
        neighbors = [node for node in neighbors if node.group_id == group_id]
        return {
            'wiki_id': wiki_id,
            'item': _node_result(item),
            'neighbors': [_node_result(node) for node in neighbors],
            'links': [
                {
                    'id': edge.uuid,
                    'type': edge.name,
                    'fact': edge.fact,
                    'source_item_id': edge.source_node_uuid,
                    'target_item_id': edge.target_node_uuid,
                }
                for edge in edges
            ],
        }

    return server
