"""
Copyright 2024, Zep Software, Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import logging
from typing import Any

from pydantic import PrivateAttr

from graphiti_core.driver.graph_operations.graph_operations import GraphOperationsInterface
from graphiti_core.vector_store.milvus_utils import (
    COLLECTION_COMMUNITY_NODES,
    COLLECTION_ENTITY_EDGES,
    COLLECTION_ENTITY_NODES,
    COLLECTION_EPISODIC_NODES,
    community_node_to_milvus_dict,
    entity_edge_to_milvus_dict,
    entity_node_to_milvus_dict,
    episodic_node_to_milvus_dict,
)

logger = logging.getLogger(__name__)


class MilvusGraphOperationsInterface(GraphOperationsInterface):
    """GraphOperationsInterface implementation that keeps Milvus collections in sync.

    Overrides save/delete/embedding-load methods to mirror data into Milvus via a
    shared VectorStoreClient. All other methods (get_by_uuid, get_by_group_ids,
    graph traversal, etc.) raise NotImplementedError to fall back to the graph database.
    """

    _vs_client: Any = PrivateAttr()

    def __init__(self, *, vs_client: Any, **data: Any) -> None:
        super().__init__(**data)
        object.__setattr__(self, '_vs_client', vs_client)

    # -----------------
    # Node: Save/Delete
    # -----------------

    async def node_save(self, node: Any, driver: Any) -> None:
        data = entity_node_to_milvus_dict(node)
        await self._vs_client.upsert(
            collection_name=self._vs_client.collection_name(COLLECTION_ENTITY_NODES),
            data=[data],
        )

    async def node_save_bulk(
        self,
        _cls: Any,
        driver: Any,
        transaction: Any,
        nodes: list[Any],
        batch_size: int = 100,
    ) -> None:
        if not nodes:
            return
        data = [entity_node_to_milvus_dict(n) for n in nodes]
        for i in range(0, len(data), batch_size):
            batch = data[i : i + batch_size]
            await self._vs_client.upsert(
                collection_name=self._vs_client.collection_name(COLLECTION_ENTITY_NODES),
                data=batch,
            )

    async def node_delete(self, node: Any, driver: Any) -> None:
        await self._vs_client.delete(
            collection_name=self._vs_client.collection_name(COLLECTION_ENTITY_NODES),
            filter_expr=f'uuid == "{node.uuid}"',
        )

    async def node_delete_by_uuids(
        self,
        _cls: Any,
        driver: Any,
        uuids: list[str],
        group_id: str | None = None,
        batch_size: int = 100,
    ) -> None:
        if not uuids:
            return
        uuids_str = ', '.join(f'"{u}"' for u in uuids)
        await self._vs_client.delete(
            collection_name=self._vs_client.collection_name(COLLECTION_ENTITY_NODES),
            filter_expr=f'uuid in [{uuids_str}]',
        )

    async def node_delete_by_group_id(
        self,
        _cls: Any,
        driver: Any,
        group_id: str,
        batch_size: int = 100,
    ) -> None:
        await self._vs_client.delete(
            collection_name=self._vs_client.collection_name(COLLECTION_ENTITY_NODES),
            filter_expr=f'group_id == "{group_id}"',
        )

    # -----------------
    # Node: Embeddings
    # -----------------

    async def node_load_embeddings(self, node: Any, driver: Any) -> None:
        results = await self._vs_client.query(
            collection_name=self._vs_client.collection_name(COLLECTION_ENTITY_NODES),
            filter_expr=f'uuid == "{node.uuid}"',
            output_fields=['name_embedding'],
        )
        if results and results[0].get('name_embedding'):
            node.name_embedding = results[0]['name_embedding']

    async def node_load_embeddings_bulk(
        self,
        driver: Any,
        nodes: list[Any],
        batch_size: int = 100,
    ) -> dict[str, list[float]]:
        if not nodes:
            return {}
        uuids = [n.uuid for n in nodes]
        uuids_str = ', '.join(f'"{u}"' for u in uuids)
        results = await self._vs_client.query(
            collection_name=self._vs_client.collection_name(COLLECTION_ENTITY_NODES),
            filter_expr=f'uuid in [{uuids_str}]',
            output_fields=['uuid', 'name_embedding'],
        )
        return {r['uuid']: r['name_embedding'] for r in results if r.get('name_embedding')}

    # -----------------
    # Edge: Save/Delete
    # -----------------

    async def edge_save(self, edge: Any, driver: Any) -> None:
        data = entity_edge_to_milvus_dict(edge)
        await self._vs_client.upsert(
            collection_name=self._vs_client.collection_name(COLLECTION_ENTITY_EDGES),
            data=[data],
        )

    async def edge_save_bulk(
        self,
        _cls: Any,
        driver: Any,
        transaction: Any,
        edges: list[Any],
        batch_size: int = 100,
    ) -> None:
        if not edges:
            return
        data = [entity_edge_to_milvus_dict(e) for e in edges]
        for i in range(0, len(data), batch_size):
            batch = data[i : i + batch_size]
            await self._vs_client.upsert(
                collection_name=self._vs_client.collection_name(COLLECTION_ENTITY_EDGES),
                data=batch,
            )

    async def edge_delete(self, edge: Any, driver: Any) -> None:
        await self._vs_client.delete(
            collection_name=self._vs_client.collection_name(COLLECTION_ENTITY_EDGES),
            filter_expr=f'uuid == "{edge.uuid}"',
        )

    async def edge_delete_by_uuids(
        self,
        _cls: Any,
        driver: Any,
        uuids: list[str],
        group_id: str | None = None,
    ) -> None:
        if not uuids:
            return
        uuids_str = ', '.join(f'"{u}"' for u in uuids)
        await self._vs_client.delete(
            collection_name=self._vs_client.collection_name(COLLECTION_ENTITY_EDGES),
            filter_expr=f'uuid in [{uuids_str}]',
        )

    # -----------------
    # Edge: Embeddings
    # -----------------

    async def edge_load_embeddings(self, edge: Any, driver: Any) -> None:
        results = await self._vs_client.query(
            collection_name=self._vs_client.collection_name(COLLECTION_ENTITY_EDGES),
            filter_expr=f'uuid == "{edge.uuid}"',
            output_fields=['fact_embedding'],
        )
        if results and results[0].get('fact_embedding'):
            edge.fact_embedding = results[0]['fact_embedding']

    async def edge_load_embeddings_bulk(
        self,
        driver: Any,
        edges: list[Any],
        batch_size: int = 100,
    ) -> dict[str, list[float]]:
        if not edges:
            return {}
        uuids = [e.uuid for e in edges]
        uuids_str = ', '.join(f'"{u}"' for u in uuids)
        results = await self._vs_client.query(
            collection_name=self._vs_client.collection_name(COLLECTION_ENTITY_EDGES),
            filter_expr=f'uuid in [{uuids_str}]',
            output_fields=['uuid', 'fact_embedding'],
        )
        return {r['uuid']: r['fact_embedding'] for r in results if r.get('fact_embedding')}

    # --------------------------
    # EpisodicNode: Save/Delete
    # --------------------------

    async def episodic_node_save(self, node: Any, driver: Any) -> None:
        data = episodic_node_to_milvus_dict(node)
        await self._vs_client.upsert(
            collection_name=self._vs_client.collection_name(COLLECTION_EPISODIC_NODES),
            data=[data],
        )

    async def episodic_node_save_bulk(
        self,
        _cls: Any,
        driver: Any,
        transaction: Any,
        nodes: list[Any],
        batch_size: int = 100,
    ) -> None:
        if not nodes:
            return
        data = [episodic_node_to_milvus_dict(n) for n in nodes]
        for i in range(0, len(data), batch_size):
            batch = data[i : i + batch_size]
            await self._vs_client.upsert(
                collection_name=self._vs_client.collection_name(COLLECTION_EPISODIC_NODES),
                data=batch,
            )

    async def episodic_node_delete(self, node: Any, driver: Any) -> None:
        await self._vs_client.delete(
            collection_name=self._vs_client.collection_name(COLLECTION_EPISODIC_NODES),
            filter_expr=f'uuid == "{node.uuid}"',
        )

    async def episodic_node_delete_by_uuids(
        self,
        _cls: Any,
        driver: Any,
        uuids: list[str],
        group_id: str | None = None,
        batch_size: int = 100,
    ) -> None:
        if not uuids:
            return
        uuids_str = ', '.join(f'"{u}"' for u in uuids)
        await self._vs_client.delete(
            collection_name=self._vs_client.collection_name(COLLECTION_EPISODIC_NODES),
            filter_expr=f'uuid in [{uuids_str}]',
        )

    async def episodic_node_delete_by_group_id(
        self,
        _cls: Any,
        driver: Any,
        group_id: str,
        batch_size: int = 100,
    ) -> None:
        await self._vs_client.delete(
            collection_name=self._vs_client.collection_name(COLLECTION_EPISODIC_NODES),
            filter_expr=f'group_id == "{group_id}"',
        )

    # -----------------------
    # CommunityNode: Save/Delete
    # -----------------------

    async def community_node_save(self, node: Any, driver: Any) -> None:
        data = community_node_to_milvus_dict(node)
        await self._vs_client.upsert(
            collection_name=self._vs_client.collection_name(COLLECTION_COMMUNITY_NODES),
            data=[data],
        )

    async def community_node_save_bulk(
        self,
        _cls: Any,
        driver: Any,
        transaction: Any,
        nodes: list[Any],
        batch_size: int = 100,
    ) -> None:
        if not nodes:
            return
        data = [community_node_to_milvus_dict(n) for n in nodes]
        for i in range(0, len(data), batch_size):
            batch = data[i : i + batch_size]
            await self._vs_client.upsert(
                collection_name=self._vs_client.collection_name(COLLECTION_COMMUNITY_NODES),
                data=batch,
            )

    async def community_node_delete(self, node: Any, driver: Any) -> None:
        await self._vs_client.delete(
            collection_name=self._vs_client.collection_name(COLLECTION_COMMUNITY_NODES),
            filter_expr=f'uuid == "{node.uuid}"',
        )

    async def community_node_delete_by_uuids(
        self,
        _cls: Any,
        driver: Any,
        uuids: list[str],
        group_id: str | None = None,
        batch_size: int = 100,
    ) -> None:
        if not uuids:
            return
        uuids_str = ', '.join(f'"{u}"' for u in uuids)
        await self._vs_client.delete(
            collection_name=self._vs_client.collection_name(COLLECTION_COMMUNITY_NODES),
            filter_expr=f'uuid in [{uuids_str}]',
        )

    async def community_node_delete_by_group_id(
        self,
        _cls: Any,
        driver: Any,
        group_id: str,
        batch_size: int = 100,
    ) -> None:
        await self._vs_client.delete(
            collection_name=self._vs_client.collection_name(COLLECTION_COMMUNITY_NODES),
            filter_expr=f'group_id == "{group_id}"',
        )

    async def community_node_load_name_embedding(
        self,
        node: Any,
        driver: Any,
    ) -> None:
        results = await self._vs_client.query(
            collection_name=self._vs_client.collection_name(COLLECTION_COMMUNITY_NODES),
            filter_expr=f'uuid == "{node.uuid}"',
            output_fields=['name_embedding'],
        )
        if results and results[0].get('name_embedding'):
            node.name_embedding = results[0]['name_embedding']

    # -----------------
    # Maintenance
    # -----------------

    async def clear_data(
        self,
        driver: Any,
        group_ids: list[str] | None = None,
    ) -> None:
        all_suffixes = [
            COLLECTION_ENTITY_NODES,
            COLLECTION_ENTITY_EDGES,
            COLLECTION_EPISODIC_NODES,
            COLLECTION_COMMUNITY_NODES,
        ]
        if group_ids:
            for group_id in group_ids:
                for suffix in all_suffixes:
                    col_name = self._vs_client.collection_name(suffix)
                    await self._vs_client.delete(
                        collection_name=col_name,
                        filter_expr=f'group_id == "{group_id}"',
                    )
        else:
            await self._vs_client.reset_collections()
